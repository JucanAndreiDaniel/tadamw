/* Public entry points: dispatch on variant, own the output buffers. */
#include <stdlib.h>

#include "aw.h"
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
	if (p.seekable)
		return TAW_E_UNSUPPORTED; /* .awseek: next step */
	taw_buf b = {0};
	st = taw_aw_encode(src, n, &p, &b);
	return finish(st, &b, out, out_n);
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
	if (info.kind != TAW_KIND_STREAM)
		return TAW_E_UNSUPPORTED;
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
