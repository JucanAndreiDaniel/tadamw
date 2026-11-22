/* Binary range coder; the per-bit steps are inline in taw_internal.h. */
#include "taw_internal.h"

void taw_rc_enc_init(taw_rc_enc *e, taw_buf *out)
{
	e->x1 = 0;
	e->x2 = 0xFFFFFFFFu;
	e->out = out;
}

void taw_rc_enc_flush(taw_rc_enc *e)
{
	taw_buf_put_u8(e->out, (uint8_t)(e->x1 >> 24));
	taw_buf_put_u8(e->out, (uint8_t)(e->x1 >> 16));
	taw_buf_put_u8(e->out, (uint8_t)(e->x1 >> 8));
	taw_buf_put_u8(e->out, (uint8_t)e->x1);
}

void taw_rc_dec_init(taw_rc_dec *d, const uint8_t *src, size_t n)
{
	d->x1 = 0;
	d->x2 = 0xFFFFFFFFu;
	d->p = src;
	d->end = src + n;
	d->x = 0;
	for (int i = 0; i < 4; i++)
		d->x = (d->x << 8) | (d->p < d->end ? *d->p++ : 0u);
}
