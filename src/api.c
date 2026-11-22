/* Public entry points: dispatch on variant, own the output buffers. */
#include <stdlib.h>

#include "aw.h"
#include "awseek.h"
#include "header.h"

static taw_status finish(taw_status st, taw_buf *b, void **out, size_t *out_n)
{
	if (st != TAW_OK) {
		taw_buf_free(b);
		return st;
	}
	if (b->oom) {
		taw_buf_free(b);
		return TAW_E_NOMEM;
	}
	*out = b->p ? b->p : malloc(1);
	*out_n = b->n;
	return *out ? TAW_OK : TAW_E_NOMEM;
}

taw_status taw_compress(const void *src, size_t n, const taw_params *params,
    void **out, size_t *out_n)
{
	if ((!src && n) || !out || !out_n)
		return TAW_E_ARG;
	taw_params p;
	taw_status st = taw_params_resolve(params, &p);
	if (st != TAW_OK)
		return st;
	taw_buf b = {0};
	st = p.seekable ? taw_awseek_encode(src, n, &p, &b)
			: taw_aw_encode(src, n, &p, &b);
	return finish(st, &b, out, out_n);
}

static taw_status decompress_seekable(const void *src, size_t n, void **out,
    size_t *out_n)
{
	taw_source in;
	taw_source_memory(&in, src, n);
	taw_seek *s;
	taw_status st = taw_seek_open(&in, &s);
	if (st != TAW_OK)
		return st;
	uint64_t total = taw_seek_size(s);
	uint8_t *dst = total < SIZE_MAX ? malloc(total ? (size_t)total : 1)
					: nullptr;
	if (!dst) {
		taw_seek_close(s);
		return TAW_E_NOMEM;
	}
	st = taw_seek_read(s, 0, dst, (size_t)total);
	taw_seek_close(s);
	if (st != TAW_OK) {
		free(dst);
		return st;
	}
	*out = dst;
	*out_n = (size_t)total;
	return TAW_OK;
}

taw_status taw_decompress(const void *src, size_t n, void **out,
    size_t *out_n)
{
	if (!src || !out || !out_n)
		return TAW_E_ARG;
	taw_info info;
	taw_status st = taw_read_info(src, n, &info);
	if (st != TAW_OK)
		return st;
	if (info.kind == TAW_KIND_SEEKABLE)
		return decompress_seekable(src, n, out, out_n);
	taw_buf b = {0};
	st = taw_aw_decode(src, n, &b);
	return finish(st, &b, out, out_n);
}

taw_status taw_read_info(const void *src, size_t n, taw_info *info)
{
	if (!src || !info)
		return TAW_E_ARG;
	taw_header h;
	taw_status st = taw_header_parse(src, n, &h);
	if (st != TAW_OK)
		return st;
	*info = (taw_info){
		.kind = (taw_kind)h.kind,
		.arith_profile = h.profile,
		.quant = h.quant,
		.level = h.level,
		.block_size = h.block_size,
		.model_size = h.model_size,
	};
	return TAW_OK;
}
