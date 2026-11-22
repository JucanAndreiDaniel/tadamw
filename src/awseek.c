/* SPDX-License-Identifier: GPL-3.0-or-later */
/* .awseek seekable variant (docs/format.md §7). */
#include <stdlib.h>
#include <string.h>

#include "awseek.h"
#include "header.h"

/* ---- sources --------------------------------------------------------------- */

static taw_status mem_read_at(void *ctx, uint64_t off, void *dst, size_t n)
{
	memcpy(dst, (const uint8_t *)ctx + off, n);
	return TAW_OK;
}

void taw_source_memory(taw_source *s, const void *p, size_t n)
{
	*s = (taw_source){.read_at = mem_read_at, .ctx = (void *)p, .size = n};
}

const uint8_t *taw_source_view(const taw_source *s)
{
	return s->read_at == mem_read_at ? s->ctx : nullptr;
}

/* ---- directory ------------------------------------------------------------- */

/* A directory as the encoder takes it or the reader parsed it: 0, or
 * TAW_E_ARG (encoder) / TAW_E_FORMAT (reader) via the caller. */
static bool files_valid(const taw_file *f, size_t n, uint64_t total)
{
	uint64_t end = 0;
	for (size_t i = 0; i < n; i++) {
		size_t len = f[i].name ? strlen(f[i].name) : 0;
		if (len == 0 || len > TAW_NAME_MAX || f[i].name[0] != '/')
			return false;
		if (f[i].offset < end || f[i].offset > total ||
		    f[i].size > total - f[i].offset)
			return false;
		end = f[i].offset + f[i].size;
		for (size_t j = 0; j < i; j++)
			if (!strcmp(f[i].name, f[j].name))
				return false;
	}
	return true;
}

#if TAW_HAVE_ENCODER
static void files_put(taw_buf *b, const taw_file *f, size_t n)
{
	taw_buf_put_u32(b, (uint32_t)n);
	for (size_t i = 0; i < n; i++) {
		size_t len = strlen(f[i].name);
		taw_buf_put_u64(b, f[i].offset);
		taw_buf_put_u64(b, f[i].size);
		taw_buf_put_u8(b, (uint8_t)len);
		taw_buf_put_u8(b, (uint8_t)(len >> 8));
		taw_buf_put(b, f[i].name, len);
	}
}

/* ---- encoder --------------------------------------------------------------- */

typedef struct {
	taw_buf out;      /* coded bytes, or a copy of a stored block */
	const uint8_t *raw_view; /* stored block of a memory source */
	size_t raw_len;
	uint32_t crc;
	taw_status st;
} block_result;

typedef struct {
	const taw_source *in;
	const uint8_t *view;
	uint64_t total, bs;
	const taw_model_choice *m;
	taw_pred *pred;   /* [threads] */
	bool *pred_ok;
	uint8_t **buf;    /* [threads] */
	block_result *res; /* [blocks] */
	/* writer */
	const taw_sink *sink;
	uint64_t at;
	taw_buf index;
} enc_job;

static void encode_block(void *arg, size_t i, unsigned w)
{
	enc_job *J = arg;
	block_result *r = &J->res[i];
	const uint64_t off = (uint64_t)i * J->bs;
	const size_t len = (size_t)(J->total - off < J->bs ? J->total - off : J->bs);
	const uint8_t *src = J->view ? J->view + off : nullptr;
	if (!src) {
		if (!J->buf[w] && !(J->buf[w] = malloc((size_t)J->bs))) {
			r->st = TAW_E_NOMEM;
			return;
		}
		if ((r->st = J->in->read_at(J->in->ctx, off, J->buf[w], len)) !=
		    TAW_OK)
			return;
		src = J->buf[w];
	}
	if (!J->pred_ok[w]) {
		r->st = taw_pred_init(&J->pred[w], J->m->profile, J->m->quant,
		    J->m->model, J->m->model_size);
		if (r->st != TAW_OK)
			return;
		J->pred_ok[w] = true;
	} else {
		taw_pred_reset(&J->pred[w]);
	}
	r->raw_len = len;
	r->crc = taw_crc32c(0, src, len);
	taw_rc_enc e;
	taw_rc_enc_init(&e, &r->out);
	taw_pred_encode(&J->pred[w], &e, src, len);
	taw_rc_enc_flush(&e);
	if (r->out.oom) {
		r->st = TAW_E_NOMEM;
		return;
	}
	if (r->out.n >= len) { /* stored: coded_len == raw_len */
		r->out.n = 0;
		if (J->view) {
			r->raw_view = src;
		} else {
			taw_buf_put(&r->out, src, len);
			if (r->out.oom)
				r->st = TAW_E_NOMEM;
		}
	}
}

static taw_status write_block(void *arg, size_t i)
{
	enc_job *J = arg;
	block_result *r = &J->res[i];
	taw_status st = r->st;
	const uint8_t *p = r->raw_view ? r->raw_view : r->out.p;
	const size_t n = r->raw_view ? r->raw_len : r->out.n;
	if (st == TAW_OK && n > UINT32_MAX)
		st = TAW_E_ARG;
	if (st == TAW_OK && n)
		st = J->sink->write(J->sink->ctx, p, n);
	if (st == TAW_OK) {
		taw_buf_put_u64(&J->index, J->at);
		taw_buf_put_u32(&J->index, (uint32_t)n);
		taw_buf_put_u32(&J->index, r->crc);
		J->at += n;
	}
	taw_buf_free(&r->out);
	return st;
}

static taw_status sink_put(const taw_sink *s, const taw_buf *b)
{
	if (b->oom)
		return TAW_E_NOMEM;
	return b->n ? s->write(s->ctx, b->p, b->n) : TAW_OK;
}

taw_status taw_seek_encode(const taw_source *src, const taw_params *params,
    const taw_file *files, size_t nfiles, const taw_sink *out)
{
	if (!src || !src->read_at || !out || !out->write || (nfiles && !files))
		return TAW_E_ARG;
	taw_params p;
	taw_params in = params ? *params : (taw_params){0};
	in.seekable = true;
	taw_status st = taw_params_resolve(&in, &p);
	if (st != TAW_OK)
		return st;
	const uint64_t total = src->size, bs = p.block_size;
	const uint64_t nblocks = (total + bs - 1) / bs;
	if (nblocks > UINT32_MAX || nfiles > UINT32_MAX ||
	    !files_valid(files, nfiles, total))
		return TAW_E_ARG;
	const unsigned threads = p.threads ? p.threads : 1;

	taw_model_choice m;
	if ((st = taw_model_fit_source(src, bs, &p, &m)) != TAW_OK)
		return st;

	taw_buf head = {0};
	taw_header h = {
		.kind = TAW_KIND_AWSEEK,
		.profile = m.profile,
		.quant = m.quant,
		.level = (uint8_t)p.level,
		.flags = nfiles ? TAW_FLAG_FILES : 0,
		.block_size = p.block_size,
		.model_size = m.model_size,
		.model_crc = taw_crc32c(0, m.model, m.model_size),
	};
	taw_header_write(&head, &h);
	taw_buf_put(&head, m.model, m.model_size);

	enc_job J = {.in = src, .view = taw_source_view(src), .total = total,
		.bs = bs, .m = &m, .sink = out, .at = head.n};
	unsigned tw = threads < nblocks ? threads : (unsigned)(nblocks ? nblocks : 1);
	J.pred = calloc(tw, sizeof *J.pred);
	J.pred_ok = calloc(tw, sizeof *J.pred_ok);
	J.buf = calloc(tw, sizeof *J.buf);
	J.res = calloc(nblocks ? (size_t)nblocks : 1, sizeof *J.res);
	taw_buf_put(&J.index, TAW_INDEX_MAGIC, 4);
	taw_buf_put_u32(&J.index, (uint32_t)nblocks);
	taw_buf_put_u64(&J.index, total);
	if (!J.pred || !J.pred_ok || !J.buf || !J.res)
		st = TAW_E_NOMEM;
	if (st == TAW_OK)
		st = sink_put(out, &head);
	if (st == TAW_OK)
		st = taw_parallel(tw, (size_t)nblocks, 2 * (size_t)tw, encode_block,
		    write_block, &J);
	if (st == TAW_OK) {
		if (nfiles)
			files_put(&J.index, files, nfiles);
		/* the index starts where the blocks end; the footer follows */
		const uint32_t crc =
		    J.index.oom ? 0 : taw_crc32c(0, J.index.p, J.index.n);
		taw_buf_put_u64(&J.index, J.at);
		taw_buf_put_u32(&J.index, crc);
		taw_buf_put(&J.index, TAW_FOOTER_MAGIC, 4);
		st = sink_put(out, &J.index);
	}
	for (uint64_t i = 0; J.res && i < nblocks; i++)
		taw_buf_free(&J.res[i].out);
	for (unsigned w = 0; w < tw && J.pred; w++) {
		if (J.pred_ok[w])
			taw_pred_free(&J.pred[w]);
		free(J.buf[w]);
	}
	free(J.pred);
	free(J.pred_ok);
	free(J.buf);
	free(J.res);
	taw_buf_free(&J.index);
	taw_buf_free(&head);
	free(m.model);
	return st;
}

#endif

#if TAW_HAVE_DECODER
/* ---- random-access reader ------------------------------------------------- */

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
	taw_file *files;  /* names point into names */
	size_t nfiles;
	char *names;
	uint8_t *coded;   /* scratch for one coded block */
	size_t coded_cap;
	uint8_t *block;   /* last decoded block */
	int64_t cached;   /* its index, or -1 */
};

static taw_status src_read(const taw_source *s, uint64_t off, void *dst,
    uint64_t n)
{
	if (off > s->size || n > s->size - off || n > SIZE_MAX)
		return TAW_E_FORMAT;
	if (n == 0)
		return TAW_OK;
	return s->read_at(s->ctx, off, dst, (size_t)n);
}

/* The directory at d (n bytes, exactly), after the block entries. */
static taw_status parse_files(taw_seek *s, const uint8_t *d, size_t n)
{
	if (n < 4)
		return TAW_E_FORMAT;
	uint32_t count = taw_rd32(d);
	size_t at = 4, names = 0;
	/* first pass: sizes */
	for (uint32_t i = 0; i < count; i++) {
		if (n - at < 18)
			return TAW_E_FORMAT;
		size_t len = taw_rd16(d + at + 16);
		if (len > n - at - 18)
			return TAW_E_FORMAT;
		names += len + 1;
		at += 18 + len;
	}
	if (at != n)
		return TAW_E_FORMAT;
	s->files = calloc(count ? count : 1, sizeof *s->files);
	s->names = malloc(names ? names : 1);
	if (!s->files || !s->names)
		return TAW_E_NOMEM;
	at = 4;
	char *np = s->names;
	for (uint32_t i = 0; i < count; i++) {
		size_t len = taw_rd16(d + at + 16);
		if (memchr(d + at + 18, 0, len))
			return TAW_E_FORMAT;
		memcpy(np, d + at + 18, len);
		np[len] = 0;
		s->files[i] = (taw_file){.name = np, .offset = taw_rd64(d + at),
			.size = taw_rd64(d + at + 8)};
		np += len + 1;
		at += 18 + len;
	}
	s->nfiles = count;
	return files_valid(s->files, count, s->total) ? TAW_OK : TAW_E_FORMAT;
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
	const uint64_t entries_end =
	    TAW_INDEX_HEAD + (uint64_t)s->nblocks * TAW_INDEX_ENTRY;
	const bool has_files = s->h.flags & TAW_FLAG_FILES;
	if (has_files ? index_len < entries_end : index_len != entries_end)
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
	if (has_files)
		st = parse_files(s, ix + entries_end,
		    (size_t)(index_len - entries_end));
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

size_t taw_seek_files(const taw_seek *s, const taw_file **files)
{
	*files = s->files;
	return s->nfiles;
}

long taw_seek_find(const taw_seek *s, const char *name)
{
	for (size_t i = 0; i < s->nfiles; i++)
		if (!strcmp(s->files[i].name, name))
			return (long)i;
	return -1;
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
	const size_t len = block_raw_len(s, i);
	if (!s->block) {
		s->block = malloc(s->h.block_size);
		if (!s->block)
			return TAW_E_NOMEM;
	}
	taw_status st;
	if (en->coded_len == len) { /* stored */
		if ((st = src_read(&s->src, en->offset, s->block, len)) != TAW_OK)
			return st;
	} else {
		if (en->coded_len > s->coded_cap) {
			uint8_t *c = realloc(s->coded, en->coded_len);
			if (!c)
				return TAW_E_NOMEM;
			s->coded = c;
			s->coded_cap = en->coded_len;
		}
		if ((st = src_read(&s->src, en->offset, s->coded,
			 en->coded_len)) != TAW_OK)
			return st;
		taw_pred_reset(&s->pr);
		taw_rc_dec d;
		taw_rc_dec_init(&d, s->coded, en->coded_len);
		taw_pred_decode(&s->pr, &d, s->block, len);
	}
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
	free(s->files);
	free(s->names);
	free(s->coded);
	free(s->block);
	free(s);
}

#endif
