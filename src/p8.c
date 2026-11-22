/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 8 model section, loading and the adaptive mixer
 * (format.md §2.3, §5.2). */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "half.h"
#include "p8.h"
#include "taw_internal.h"

/* ---- model head ---------------------------------------------------------------- */

void p8_head_write(uint8_t out[P8_MODEL_HEAD], const p1_shape *s,
    uint8_t q1, uint8_t q2, uint8_t ushift)
{
	memset(out, 0, P8_MODEL_HEAD);
	out[0] = s->lines_log2;
	out[1] = s->hist_log2;
	out[2] = s->match_log2;
	out[3] = s->sel1;
	out[4] = s->hidden;
	out[5] = s->sel2;
	out[6] = q1;
	out[7] = q2;
	out[8] = ushift;
}

taw_status p8_head_parse(const uint8_t *p, size_t n, p1_shape *s,
    uint8_t *q1, uint8_t *q2, uint8_t *ushift)
{
	if (n < P8_MODEL_HEAD)
		return TAW_E_FORMAT;
	*s = (p1_shape){
		.lines_log2 = p[0],
		.hist_log2 = p[1],
		.match_log2 = p[2],
		.sel1 = p[3],
		.hidden = p[4],
		.sel2 = p[5],
	};
	*q1 = p[6];
	*q2 = p[7];
	*ushift = p[8];
	if (p[9] || p[10] || p[11])
		return TAW_E_FORMAT;
	if (s->lines_log2 < P1_LINES_MIN || s->lines_log2 > P1_LINES_MAX ||
	    s->hist_log2 < P1_HIST_MIN || s->hist_log2 > P1_HIST_MAX ||
	    s->match_log2 < P1_MATCH_MIN || s->match_log2 > P1_MATCH_MAX)
		return TAW_E_FORMAT;
	if (s->hidden > P1_HMAX || s->sel2 >= P1_SEL_COUNT ||
	    s->sel1 >= P1_SEL_COUNT)
		return TAW_E_FORMAT;
	if (*q1 != TAW_QUANT_NONE && *q1 != TAW_QUANT_F16 && *q1 != TAW_QUANT_Q8)
		return TAW_E_FORMAT;
	if (*q2 != TAW_QUANT_Q8 && *q2 != TAW_QUANT_F16)
		return TAW_E_FORMAT;
	if ((s->hidden == 0) != (*q1 == TAW_QUANT_NONE) ||
	    (s->hidden == 0 && s->sel1 != P1_SEL_NONE))
		return TAW_E_FORMAT;
	if (*ushift < P8_USHIFT_MIN || *ushift > P8_USHIFT_MAX)
		return TAW_E_FORMAT;
	return TAW_OK;
}

uint64_t p8_model_size(const p1_shape *s, uint8_t q1, uint8_t q2)
{
	const unsigned E = s->hidden ? s->hidden + 1 : P1_F;
	const unsigned w2row = q2 == TAW_QUANT_Q8 ? E + 2 : 2 * E;
	uint64_t n = P8_MODEL_HEAD + (uint64_t)p1_sel_size[s->sel2] * w2row + E;
	if (s->hidden) {
		uint64_t row = q1 == TAW_QUANT_Q8 ? P1_F + 2 : 2 * P1_F;
		n += (uint64_t)p1_sel_size[s->sel1] * s->hidden * row;
	}
	return n;
}

/* ---- loading -------------------------------------------------------------------- */

static bool half_finite(uint16_t h)
{
	return (h & 0x7C00u) != 0x7C00u;
}

/* One row in the profile 1 weight format (format.md §5.1). */
static taw_status read_row(const uint8_t **pp, uint8_t quant, unsigned len,
    float *dst)
{
	const uint8_t *p = *pp;
	if (quant == TAW_QUANT_Q8) {
		uint16_t sh = taw_rd16(p);
		if (!half_finite(sh))
			return TAW_E_FORMAT;
		const float scale = taw_h2f(sh);
		p += 2;
		for (unsigned i = 0; i < len; i++) {
			int8_t q = (int8_t)p[i];
			if (q == -128)
				return TAW_E_FORMAT;
			dst[i] = (float)q * scale;
		}
		p += len;
	} else {
		for (unsigned i = 0; i < len; i++) {
			uint16_t h = taw_rd16(p + 2 * i);
			if (!half_finite(h))
				return TAW_E_FORMAT;
			dst[i] = taw_h2f(h);
		}
		p += 2 * len;
	}
	*pp = p;
	return TAW_OK;
}

taw_status p8_load(p8_mix *m, const uint8_t *model, size_t size)
{
	memset(m, 0, sizeof *m);
	uint8_t q1, q2, ushift;
	p1_shape s;
	taw_status st = p8_head_parse(model, size, &s, &q1, &q2, &ushift);
	if (st != TAW_OK)
		return st;
	if ((uint64_t)size != p8_model_size(&s, q1, q2))
		return TAW_E_FORMAT;
	m->s = s;
	m->q1 = q1;
	m->q2 = q2;
	m->ushift = ushift;
	m->E = s.hidden ? s.hidden + 1 : P1_F;
	m->nset1 = s.hidden ? p1_sel_size[s.sel1] : 0;
	m->nset2 = p1_sel_size[s.sel2];
	m->w1 = calloc((size_t)m->nset1 * s.hidden * P1_F, sizeof *m->w1);
	m->w2 = malloc((size_t)m->nset2 * m->E * sizeof *m->w2);
	m->dw = calloc((size_t)m->nset2 * m->E, sizeof *m->dw);
	if ((m->nset1 && !m->w1) || !m->w2 || !m->dw) {
		p8_free(m);
		return TAW_E_NOMEM;
	}
	const uint8_t *p = model + P8_MODEL_HEAD;
	for (size_t set = 0; set < m->nset1; set++)
		for (unsigned j = 0; j < s.hidden; j++) {
			st = read_row(&p, q1, P1_F,
			    m->w1 + (set * s.hidden + j) * P1_F);
			if (st != TAW_OK) {
				p8_free(m);
				return st;
			}
		}
	for (size_t set = 0; set < m->nset2; set++) {
		st = read_row(&p, q2, m->E, m->w2 + set * m->E);
		if (st != TAW_OK) {
			p8_free(m);
			return st;
		}
	}
	memcpy(m->rate, p, m->E);
	p8_reset(m);
	return TAW_OK;
}

void p8_free(p8_mix *m)
{
	free(m->w1);
	free(m->w2);
	free(m->dw);
	memset(m, 0, sizeof *m);
}

void p8_reset(p8_mix *m)
{
	memset(m->dw, 0, (size_t)m->nset2 * m->E * sizeof *m->dw);
}

/* ---- features and mixer --------------------------------------------------------- */

/* Same dot product shape as profile 1 (format.md §2.2.8). */
static inline float dot4(const float *w, size_t ws, const float *x,
    unsigned n)
{
	float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
	for (unsigned i = 0; i < n; i++) {
		switch (i & 3) {
		case 0: s0 += w[i * ws] * x[i]; break;
		case 1: s1 += w[i * ws] * x[i]; break;
		case 2: s2 += w[i * ws] * x[i]; break;
		default: s3 += w[i * ws] * x[i]; break;
		}
	}
	return (s0 + s1) + (s2 + s3);
}

void p8_features(const p8_mix *m, const float x[P1_XPAD],
    const p1_sel_ctx *sc, int32_t e[P8_MAXE], float h[P1_HMAX])
{
	const unsigned H = m->s.hidden;
	if (!H) {
		for (unsigned i = 0; i < P1_NCTX; i++)
			e[i] = (int32_t)(x[i] * 256.0f +
			    (x[i] >= 0.0f ? 0.5f : -0.5f));
	} else {
		const float *w1 = m->w1 + (size_t)p1_sel_value(m->s.sel1, sc) *
		    H * P1_F;
		for (unsigned j = 0; j < H; j++) {
			float a = dot4(w1 + j * P1_F, 1, x, P1_F);
			h[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
			e[j] = (int32_t)(h[j] * 256.0f +
			    (h[j] >= 0.0f ? 0.5f : -0.5f));
		}
	}
	e[m->E - 1] = 256;
}

static inline int32_t clamp16(int64_t v)
{
	return (int32_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v);
}

static inline size_t out_set(const p8_mix *m, const p1_sel_ctx *sc)
{
	return (size_t)p1_sel_value(m->s.sel2, sc) * m->E;
}

int32_t p8_eval(const p8_mix *m, const float x[P1_XPAD],
    const int32_t e[P8_MAXE], const float h[P1_HMAX],
    const p1_sel_ctx *sc)
{
	const unsigned H = m->s.hidden;
	const size_t off = out_set(m, sc);
	const float *w2 = m->w2 + off;
	const float lim = 2047.0f / 256.0f;
	/* The frozen base is profile 1's mixer, operation for operation. */
	float z = H ? dot4(w2, 1, h, H) + w2[H] : dot4(w2, 1, x, P1_F);
	z = z > lim ? lim : z < -lim ? -lim : z;
	const int32_t base = (int32_t)(z * 256.0f);
	/* The adaptive part is an exact integer correction on top. */
	const int32_t *dw = m->dw + off;
	int64_t acc = 0;
	for (unsigned i = 0; i < m->E; i++)
		acc += (int64_t)dw[i] * e[i];
	const int64_t delta = p8_divpow2(acc, P8_MIX_SHIFT);
	const int64_t t = (int64_t)base + delta;
	return (int32_t)(t < -2047 ? -2047 : t > 2047 ? 2047 : t);
}

void p8_update(p8_mix *m, const int32_t e[P8_MAXE], const p1_sel_ctx *sc,
    int bit, uint32_t p_mix)
{
	int32_t *dw = m->dw + out_set(m, sc);
	const int64_t err = (bit ? 65536 : 0) - (int64_t)p_mix;
	for (unsigned i = 0; i < m->E; i++) {
		int64_t g = p8_divpow2(err * e[i] * (int64_t)m->rate[i],
		    m->ushift);
		dw[i] = clamp16((int64_t)dw[i] + g);
	}
}
