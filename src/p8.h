/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 8, PAQ-style adaptive logistic mixing (format.md §2.3,
 * §5.2). The context models, squash/stretch and APM stages are the profile 1
 * ones; the mixer is different.
 *
 * The frozen first layer (exactly profile 1's, with the same row formats)
 * turns the 25 context features into H expert outputs in [-8, 8], or, when
 * H = 0, the context features themselves are the experts. The output layer
 * has a frozen binary32 part (the initial weights from the archive) plus a
 * live int32 *delta* per expert, starting at zero, updated after every bit
 * by a deterministic integer logistic gradient step with a stored rate.
 *
 * With all rates zero this is bit-identical to profile 1's frozen mixer;
 * the integer delta is a pure PAQ-style adaptive correction. */
#ifndef TAW_P8_H
#define TAW_P8_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"
#include "taw_internal.h"

/* Division by 2^n, truncated toward zero (symmetric, so positive and
 * negative gradient steps of equal magnitude cancel). */
static inline int64_t p8_divpow2(int64_t v, unsigned n)
{
	return v / ((int64_t)1 << n);
}

enum {
	P8_MODEL_HEAD = 12,
	P8_MAXE = TAW_P8_MAXE, /* experts + bias (26 at most) */
	P8_MIX_SHIFT = 12,     /* delta term: trunc((dw * expert) / 2^12) */
	P8_USHIFT_MIN = 4,
	P8_USHIFT_MAX = 31,
};

typedef struct p8_mix {
	p1_shape s;      /* geometry, sel1 (frozen layer), sel2 (adaptive) */
	uint8_t ushift;  /* gradient-step shift from the head */
	uint8_t q1;      /* quant of the frozen layer: F16 or Q8 */
	uint8_t q2;      /* quant of the initial output weights: F16 or Q8 */
	unsigned E;      /* experts + bias: hidden ? H + 1 : 26 */
	uint32_t nset1, nset2;
	float *w1;       /* [nset1][H][P1_F] frozen first layer */
	float *w2;       /* [nset2][E] frozen output weights, dequantized */
	int32_t *dw;     /* [nset2][E] live adaptive deltas, clamp to int16 */
	uint8_t rate[P8_MAXE];
} p8_mix;

/* Head and model size. */
void p8_head_write(uint8_t out[P8_MODEL_HEAD], const p1_shape *s,
    uint8_t q1, uint8_t q2, uint8_t ushift);
taw_status p8_head_parse(const uint8_t *p, size_t n, p1_shape *s,
    uint8_t *q1, uint8_t *q2, uint8_t *ushift);
uint64_t p8_model_size(const p1_shape *s, uint8_t q1, uint8_t q2);

/* Parse + allocate + reset. */
taw_status p8_load(p8_mix *m, const uint8_t *model, size_t size);
void p8_free(p8_mix *m);
void p8_reset(p8_mix *m);

/* Expert vector: e[0..E-2] are h[j]*256 (or st[i] when H = 0) as exact
 * integers, e[E-1] = 256 (the bias). h[] holds the binary32 hidden values
 * for the frozen output layer. */
void p8_features(const p8_mix *m, const float x[P1_XPAD],
    const p1_sel_ctx *sc, int32_t e[P8_MAXE], float h[P1_HMAX]);

/* Mixer (frozen base + adaptive integer delta) and its gradient step. */
int32_t p8_eval(const p8_mix *m, const float x[P1_XPAD],
    const int32_t e[P8_MAXE], const float h[P1_HMAX],
    const p1_sel_ctx *sc);
void p8_update(p8_mix *m, const int32_t e[P8_MAXE], const p1_sel_ctx *sc,
    int bit, uint32_t p_mix);

#endif
