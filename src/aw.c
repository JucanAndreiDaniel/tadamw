/* .aw stream variant (docs/format.md §6). */
#include <stdlib.h>
#include <string.h>

#include "aw.h"
#include "header.h"

/* Encoder chunk size; the format allows up to TAW_CHUNK_MAX. */
constexpr size_t AW_CHUNK = 1u << 20;

taw_status taw_aw_encode(const uint8_t *src, size_t n, const taw_params *p,
    taw_buf *out)
{
	taw_header h = {
		.kind = TAW_KIND_AW,
		.profile = TAW_PROFILE_INT_ORDER0,
		.quant = TAW_QUANT_NONE,
		.level = (uint8_t)p->level,
		.model_crc = taw_crc32c(0, nullptr, 0),
	};
	taw_header_write(out, &h);

	taw_pred pr;
	taw_status st = taw_pred_init(&pr, h.profile, h.quant, nullptr, 0);
	if (st != TAW_OK)
		return st;

	taw_buf coded = {0};
	for (size_t off = 0; off < n; off += AW_CHUNK) {
		size_t len = n - off < AW_CHUNK ? n - off : AW_CHUNK;
		coded.n = 0;
		taw_rc_enc e;
		taw_rc_enc_init(&e, &coded);
		taw_pred_encode(&pr, &e, src + off, len);
		taw_rc_enc_flush(&e);
		if (coded.oom || coded.n > UINT32_MAX)
			break;
		taw_buf_put_u32(out, (uint32_t)len);
		taw_buf_put_u32(out, (uint32_t)coded.n);
		taw_buf_put_u32(out, taw_crc32c(0, src + off, len));
		taw_buf_put(out, coded.p, coded.n);
	}
	bool oom = coded.oom || coded.n > UINT32_MAX;
	taw_buf_free(&coded);
	if (oom)
		return TAW_E_NOMEM;

	taw_buf_put_u32(out, 0);
	taw_buf_put_u32(out, 0);
	taw_buf_put_u32(out, taw_crc32c(0, src, n));
	taw_buf_put_u64(out, (uint64_t)n);
	return out->oom ? TAW_E_NOMEM : TAW_OK;
}

taw_status taw_aw_decode(const uint8_t *src, size_t n, taw_buf *out)
{
	taw_header h;
	const uint8_t *model;
	taw_status st = taw_open_common(src, n, TAW_KIND_AW, &h, &model);
	if (st != TAW_OK)
		return st;
	taw_pred pr;
	st = taw_pred_init(&pr, h.profile, h.quant, model, (size_t)h.model_size);
	if (st != TAW_OK)
		return st;

	const uint8_t *p = model + h.model_size, *end = src + n;
	for (;;) {
		if ((size_t)(end - p) < 12)
			return TAW_E_FORMAT;
		uint32_t raw_len = taw_rd32(p);
		uint32_t coded_len = taw_rd32(p + 4);
		uint32_t crc = taw_rd32(p + 8);
		p += 12;
		if (raw_len == 0) {
			if (coded_len != 0 || (size_t)(end - p) != 8)
				return TAW_E_FORMAT;
			if (taw_rd64(p) != out->n)
				return TAW_E_FORMAT;
			if (crc != taw_crc32c(0, out->p, out->n))
				return TAW_E_CHECKSUM;
			return TAW_OK;
		}
		if (raw_len > TAW_CHUNK_MAX || coded_len > (size_t)(end - p))
			return TAW_E_FORMAT;
		uint8_t *dst = taw_buf_extend(out, raw_len);
		if (!dst)
			return TAW_E_NOMEM;
		taw_rc_dec d;
		taw_rc_dec_init(&d, p, coded_len);
		taw_pred_decode(&pr, &d, dst, raw_len);
		if (crc != taw_crc32c(0, dst, raw_len))
			return TAW_E_CHECKSUM;
		p += coded_len;
	}
}
