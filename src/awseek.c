/* SPDX-License-Identifier: GPL-3.0-or-later */
/* .awseek seekable variant (docs/format.md §7). */
#include <stdlib.h>
#include <string.h>

#include "awseek.h"
#include "header.h"

/* ---- encoder ------------------------------------------------------------ */

taw_status taw_awseek_encode(const uint8_t *src, size_t n,
    const taw_params *p, const taw_model_choice *m, taw_buf *out)
{
	const size_t bs = p->block_size;
	const uint64_t nblocks = ((uint64_t)n + bs - 1) / bs;
	if (nblocks > UINT32_MAX)
		return TAW_E_ARG;

	taw_header h = {
		.kind = TAW_KIND_AWSEEK,
		.profile = m->profile,
		.quant = m->quant,
		.level = (uint8_t)p->level,
		.block_size = p->block_size,
		.model_size = m->model_size,
		.model_crc = taw_crc32c(0, m->model, m->model_size),
	};
	taw_header_write(out, &h);
	taw_buf_put(out, m->model, m->model_size);

	taw_pred pr;
	taw_status st =
	    taw_pred_init(&pr, h.profile, h.quant, m->model, m->model_size);
	if (st != TAW_OK)
		return st;

	taw_buf index = {0};
	taw_buf_put(&index, TAW_INDEX_MAGIC, 4);
	taw_buf_put_u32(&index, (uint32_t)nblocks);
	taw_buf_put_u64(&index, (uint64_t)n);

	for (size_t off = 0; off < n && !out->oom; off += bs) {
		size_t len = n - off < bs ? n - off : bs;
		uint64_t at = out->n;
		taw_pred_reset(&pr);
		taw_rc_enc e;
		taw_rc_enc_init(&e, out);
		taw_pred_encode(&pr, &e, src + off, len);
		taw_rc_enc_flush(&e);
		uint64_t coded = out->n - at;
		if (coded > UINT32_MAX) {
			taw_buf_free(&index);
			taw_pred_free(&pr);
			return TAW_E_ARG;
		}
		taw_buf_put_u64(&index, at);
		taw_buf_put_u32(&index, (uint32_t)coded);
		taw_buf_put_u32(&index, taw_crc32c(0, src + off, len));
	}
	taw_pred_free(&pr);
	if (index.oom) {
		taw_buf_free(&index);
		return TAW_E_NOMEM;
	}

	uint64_t index_off = out->n;
	taw_buf_put(out, index.p, index.n);
	taw_buf_put_u64(out, index_off);
	taw_buf_put_u32(out, taw_crc32c(0, index.p, index.n));
	taw_buf_put(out, TAW_FOOTER_MAGIC, 4);
	taw_buf_free(&index);
	return out->oom ? TAW_E_NOMEM : TAW_OK;
}

/* ---- random-access reader ----------------------------------------------- */

typedef struct {
	uint64_t offset;
	uint32_t coded_len, crc;
} seek_entry;

struct taw_seek {
	taw_source src;
	taw_header h;
	uint8_t *model;
	taw_pred pr;
	uint64_t total;
	uint32_t nblocks;
	seek_entry *ent;
	uint8_t *coded;   /* scratch for one coded block */
	size_t coded_cap;
	uint8_t *block;   /* last decoded block */
	int64_t cached;   /* its index, or -1 */
};

static taw_status mem_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	memcpy(dst, (const uint8_t *)ctx + off, n);
	return TAW_OK;
}

void taw_source_memory(taw_source *s, const void *p, size_t n)
{
	*s = (taw_source){.read_at = mem_read_at, .ctx = (void *)p, .size = n};
}

static taw_status src_read(const taw_source *s, uint64_t off, void *dst,
    uint64_t n)
{
	if (off > s->size || n > s->size - off || n > SIZE_MAX)
		return TAW_E_FORMAT;
	if (n == 0)
		return TAW_OK;
	return s->read_at(s->ctx, off, dst, (size_t)n);
}

static taw_status seek_load(taw_seek *s)
{
	const uint64_t size = s->src.size;
	uint8_t hb[TAW_HEADER_SIZE];
	taw_status st;

	if (size < TAW_HEADER_SIZE + TAW_FOOTER_SIZE)
		return TAW_E_FORMAT;
	if ((st = src_read(&s->src, 0, hb, sizeof hb)) != TAW_OK)
		return st;
	if ((st = taw_header_parse(hb, sizeof hb, &s->h)) != TAW_OK)
		return st;
	if (s->h.kind != TAW_KIND_AWSEEK)
		return TAW_E_ARG;

	const uint64_t data_start = TAW_HEADER_SIZE + s->h.model_size;
	if (s->h.model_size > size - TAW_HEADER_SIZE - TAW_FOOTER_SIZE)
		return TAW_E_FORMAT;
	s->model = malloc(s->h.model_size ? (size_t)s->h.model_size : 1);
	if (!s->model)
		return TAW_E_NOMEM;
	if ((st = src_read(&s->src, TAW_HEADER_SIZE, s->model,
		 s->h.model_size)) != TAW_OK)
		return st;
	if (taw_crc32c(0, s->model, (size_t)s->h.model_size) != s->h.model_crc)
		return TAW_E_CHECKSUM;
	if ((st = taw_pred_init(&s->pr, s->h.profile, s->h.quant, s->model,
		 (size_t)s->h.model_size)) != TAW_OK)
		return st;

	uint8_t ft[TAW_FOOTER_SIZE];
	const uint64_t footer_off = size - TAW_FOOTER_SIZE;
	if ((st = src_read(&s->src, footer_off, ft, sizeof ft)) != TAW_OK)
		return st;
	if (memcmp(ft + 12, TAW_FOOTER_MAGIC, 4) != 0)
		return TAW_E_FORMAT;
	const uint64_t index_off = taw_rd64(ft);
	if (index_off < data_start || index_off > footer_off ||
	    footer_off - index_off < TAW_INDEX_HEAD)
		return TAW_E_FORMAT;
	const uint64_t index_len = footer_off - index_off;
	if (index_len > SIZE_MAX)
		return TAW_E_NOMEM;
	uint8_t *ix = malloc((size_t)index_len);
	if (!ix)
		return TAW_E_NOMEM;
	if ((st = src_read(&s->src, index_off, ix, index_len)) != TAW_OK)
		goto out;
	st = TAW_E_CHECKSUM;
	if (taw_crc32c(0, ix, (size_t)index_len) != taw_rd32(ft + 8))
		goto out;
	st = TAW_E_FORMAT;
	if (memcmp(ix, TAW_INDEX_MAGIC, 4) != 0)
		goto out;
	s->nblocks = taw_rd32(ix + 4);
	s->total = taw_rd64(ix + 8);
	const uint64_t bs = s->h.block_size;
	if (s->total / bs + (s->total % bs != 0) != s->nblocks)
		goto out;
	if (index_len != TAW_INDEX_HEAD + (uint64_t)s->nblocks * TAW_INDEX_ENTRY)
		goto out;
	st = TAW_E_NOMEM;
	s->ent = calloc(s->nblocks ? s->nblocks : 1, sizeof *s->ent);
	if (!s->ent)
		goto out;
	st = TAW_E_FORMAT;
	for (uint32_t i = 0; i < s->nblocks; i++) {
		const uint8_t *e = ix + TAW_INDEX_HEAD + (size_t)i * TAW_INDEX_ENTRY;
		seek_entry *en = &s->ent[i];
		en->offset = taw_rd64(e);
		en->coded_len = taw_rd32(e + 8);
		en->crc = taw_rd32(e + 12);
		if (en->offset < data_start || en->offset > index_off ||
		    en->coded_len > index_off - en->offset)
			goto out;
	}
	st = TAW_OK;
out:
	free(ix);
	return st;
}

taw_status taw_seek_open(const taw_source *src, taw_seek **out)
{
	if (!src || !src->read_at || !out)
		return TAW_E_ARG;
	taw_seek *s = calloc(1, sizeof *s);
	if (!s)
		return TAW_E_NOMEM;
	s->src = *src;
	s->cached = -1;
	taw_status st = seek_load(s);
	if (st != TAW_OK) {
		taw_seek_close(s);
		return st;
	}
	*out = s;
	return TAW_OK;
}

uint64_t taw_seek_size(const taw_seek *s)
{
	return s->total;
}

static size_t block_raw_len(const taw_seek *s, uint32_t i)
{
	uint64_t start = (uint64_t)i * s->h.block_size;
	uint64_t left = s->total - start;
	return left < s->h.block_size ? (size_t)left : s->h.block_size;
}

static taw_status load_block(taw_seek *s, uint32_t i)
{
	if (s->cached == i)
		return TAW_OK;
	s->cached = -1;
	const seek_entry *en = &s->ent[i];
	if (en->coded_len > s->coded_cap) {
		uint8_t *c = realloc(s->coded, en->coded_len);
		if (!c)
			return TAW_E_NOMEM;
		s->coded = c;
		s->coded_cap = en->coded_len;
	}
	if (!s->block) {
		s->block = malloc(s->h.block_size);
		if (!s->block)
			return TAW_E_NOMEM;
	}
	taw_status st = src_read(&s->src, en->offset, s->coded, en->coded_len);
	if (st != TAW_OK)
		return st;
	size_t len = block_raw_len(s, i);
	taw_pred_reset(&s->pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, s->coded, en->coded_len);
	taw_pred_decode(&s->pr, &d, s->block, len);
	if (taw_crc32c(0, s->block, len) != en->crc)
		return TAW_E_CHECKSUM;
	s->cached = i;
	return TAW_OK;
}

taw_status taw_seek_read(taw_seek *s, uint64_t off, void *dst, size_t n)
{
	if (!s || (!dst && n))
		return TAW_E_ARG;
	if (off > s->total || n > s->total - off)
		return TAW_E_RANGE;
	uint8_t *d = dst;
	const uint64_t bs = s->h.block_size;
	while (n) {
		uint32_t i = (uint32_t)(off / bs);
		size_t in = (size_t)(off % bs);
		taw_status st = load_block(s, i);
		if (st != TAW_OK)
			return st;
		size_t k = block_raw_len(s, i) - in;
		if (k > n)
			k = n;
		memcpy(d, s->block + in, k);
		d += k;
		off += k;
		n -= k;
	}
	return TAW_OK;
}

void taw_seek_close(taw_seek *s)
{
	if (!s)
		return;
	taw_pred_free(&s->pr);
	free(s->model);
	free(s->ent);
	free(s->coded);
	free(s->block);
	free(s);
}
