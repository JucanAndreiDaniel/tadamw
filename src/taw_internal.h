/* Internal interfaces shared by the libtaw sources and the unit tests. */
#ifndef TAW_INTERNAL_H
#define TAW_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "taw/taw.h"

/* ---- growable byte buffer ---------------------------------------------- */

typedef struct {
	uint8_t *p;
	size_t n, cap;
	bool oom;
} taw_buf;

void taw_buf_put(taw_buf *b, const void *src, size_t n);
void taw_buf_put_u8(taw_buf *b, uint8_t v);
/* Append n uninitialized bytes; returns them, or nullptr on OOM. */
uint8_t *taw_buf_extend(taw_buf *b, size_t n);
void taw_buf_put_u32(taw_buf *b, uint32_t v);
void taw_buf_put_u64(taw_buf *b, uint64_t v);
void taw_buf_free(taw_buf *b);

static inline uint16_t taw_rd16(const uint8_t *p)
{
	return (uint16_t)(p[0] | p[1] << 8);
}
static inline uint32_t taw_rd32(const uint8_t *p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
	    (uint32_t)p[3] << 24;
}
static inline uint64_t taw_rd64(const uint8_t *p)
{
	return (uint64_t)taw_rd32(p) | (uint64_t)taw_rd32(p + 4) << 32;
}
static inline void taw_wr32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}
static inline void taw_wr64(uint8_t *p, uint64_t v)
{
	taw_wr32(p, (uint32_t)v);
	taw_wr32(p + 4, (uint32_t)(v >> 32));
}

/* ---- CRC-32C ------------------------------------------------------------ */

/* Continue a CRC-32C; start with crc = 0. */
uint32_t taw_crc32c(uint32_t crc, const void *data, size_t n);

/* ---- binary range coder (docs/format.md §4) ----------------------------- */

typedef struct {
	uint32_t x1, x2;
	taw_buf *out;
} taw_rc_enc;

typedef struct {
	uint32_t x1, x2, x;
	const uint8_t *p, *end;
} taw_rc_dec;

void taw_rc_enc_init(taw_rc_enc *e, taw_buf *out);
void taw_rc_enc_flush(taw_rc_enc *e);
void taw_rc_dec_init(taw_rc_dec *d, const uint8_t *src, size_t n);

static inline void taw_rc_enc_bit(taw_rc_enc *e, int bit, uint32_t p1)
{
	uint32_t xmid =
	    e->x1 + (uint32_t)(((uint64_t)(e->x2 - e->x1) * p1) >> 16);
	if (bit)
		e->x2 = xmid;
	else
		e->x1 = xmid + 1;
	while (((e->x1 ^ e->x2) & 0xFF000000u) == 0) {
		taw_buf_put_u8(e->out, (uint8_t)(e->x2 >> 24));
		e->x1 <<= 8;
		e->x2 = (e->x2 << 8) | 0xFFu;
	}
}

static inline int taw_rc_dec_bit(taw_rc_dec *d, uint32_t p1)
{
	uint32_t xmid =
	    d->x1 + (uint32_t)(((uint64_t)(d->x2 - d->x1) * p1) >> 16);
	int bit = d->x <= xmid;
	if (bit)
		d->x2 = xmid;
	else
		d->x1 = xmid + 1;
	while (((d->x1 ^ d->x2) & 0xFF000000u) == 0) {
		d->x1 <<= 8;
		d->x2 = (d->x2 << 8) | 0xFFu;
		d->x = (d->x << 8) | (d->p < d->end ? *d->p++ : 0u);
	}
	return bit;
}

/* ---- predictor / arithmetic profiles (docs/format.md §2) ---------------- */

typedef struct {
	uint8_t profile;
	uint32_t c0;     /* partial byte with leading 1 bit, 1..255 */
	uint16_t p[256]; /* profile 0 state */
} taw_pred;

/* Is (profile, quant) a pair this build can decode? */
bool taw_pred_supported(uint8_t profile, uint8_t quant);

/* Set up a predictor from a validated header's profile and model section. */
taw_status taw_pred_init(taw_pred *pr, uint8_t profile, uint8_t quant,
    const uint8_t *model, size_t model_size);

/* Return to the start-of-coding-unit state; the model is kept. */
void taw_pred_reset(taw_pred *pr);

void taw_pred_encode(taw_pred *pr, taw_rc_enc *e, const uint8_t *src,
    size_t n);
void taw_pred_decode(taw_pred *pr, taw_rc_dec *d, uint8_t *dst, size_t n);

#endif
