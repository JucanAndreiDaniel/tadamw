/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 3 model section: integer softmax, soft-gated mixer, inference.
 * Everything that is not the mixer (context models, squash, stretch, APM,
 * range coder) is shared with profile 1 and lives in p1_ctx.c / predictor.c. */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "half.h"
#include "p3.h"
#include "taw_internal.h"

/* ---- integer softmax table (normative, §2.3.2) ---------------------------- */

uint16_t p3_e_tab[512];

void p3_tables_init(void)
{
	p3_e_tab[0] = 256;
	for (unsigned d = 1; d < 512; d++)
		p3_e_tab[d] = (uint16_t)(((uint32_t)p3_e_tab[d - 1] * 235u) >> 8);
}

/* ---- model head ---------------------------------------------------------- */

void p3_head_write(uint8_t out[P3_HEAD], const p3_shape *s)
{
	out[0] = s->K;
	out[1] = s->hidden;
	out[2] = s->lines_log2;
	out[3] = s->hist_log2;
	out[4] = s->match_log2;
	out[5] = s->gate_flags;
	out[6] = 0;
	out[7] = 0;
}

taw_status p3_head_parse(const uint8_t *p, size_t n, p3_shape *s)
{
	if (n < P3_HEAD)
		return TAW_E_FORMAT;
	*s = (p3_shape){
		.K = p[0], .hidden = p[1], .lines_log2 = p[2],
		.hist_log2 = p[3], .match_log2 = p[4], .gate_flags = p[5],
	};
	if (p[6] || p[7])
		return TAW_E_FORMAT;
	if (s->K < P3_KMIN || s->K > P3_KMAX || s->hidden > P3_HMAX)
		return TAW_E_FORMAT;
	if (s->lines_log2 < P1_LINES_MIN || s->lines_log2 > P1_LINES_MAX ||
	    s->hist_log2 < P1_HIST_MIN || s->hist_log2 > P1_HIST_MAX ||
	    s->match_log2 < P1_MATCH_MIN || s->match_log2 > P1_MATCH_MAX)
		return TAW_E_FORMAT;
	if (s->gate_flags & ~(P3_FLAG_MSEL | P3_FLAG_CONF))
		return TAW_E_FORMAT;
	if ((s->gate_flags & P3_FLAG_CONF) && s->hidden != 0)
		return TAW_E_FORMAT; /* the fixed context expert needs H = 0 */
	return TAW_OK;
}

bool p3_quant_supported(uint8_t quant)
{
	return quant == TAW_QUANT_F16 || quant == TAW_QUANT_Q8;
}

uint64_t p3_model_size(const p3_shape *s, uint8_t quant, bool calibrated)
{
	return (calibrated ? P3_CAL_HEAD : P3_HEAD) + p3_weights_size(s, quant);
}

void p3_cal_head_write(uint8_t out[P3_CAL_HEAD], const p3_shape *s,
    const p3_cal *cal)
{
	p3_head_write(out, s);
	out[8] = cal->w0;
	out[9] = cal->w1;
	out[10] = cal->w2;
	for (unsigned i = 11; i < P3_CAL_HEAD; i++)
		out[i] = 0;
}

taw_status p3_cal_head_parse(const uint8_t *p, size_t n, p3_shape *s,
    p3_cal *cal)
{
	if (n < P3_CAL_HEAD || p3_head_parse(p, n, s) != TAW_OK)
		return TAW_E_FORMAT;
	*cal = (p3_cal){.w0 = p[8], .w1 = p[9], .w2 = p[10]};
	if (cal->w0 > P3_W_MAX || cal->w1 > P3_W_MAX || cal->w2 > P3_W_MAX ||
	    (cal->w0 == 0 && cal->w1 == 0 && cal->w2 == 0))
		return TAW_E_FORMAT;
	for (unsigned i = 11; i < P3_CAL_HEAD; i++)
		if (p[i])
			return TAW_E_FORMAT;
	return TAW_OK;
}

uint64_t p3_weights_size(const p3_shape *s, uint8_t quant)
{
	const unsigned H = s->hidden;
	const unsigned out_len = H ? H + 1 : P1_F;
	uint64_t bytes = 0;
	if (quant == TAW_QUANT_Q8) {
		if (H)
			bytes += (uint64_t)H * (2u + P1_F);
		bytes += (uint64_t)s->K * (2u + out_len);
	} else {
		if (H)
			bytes += (uint64_t)H * P1_F * 2u;
		bytes += (uint64_t)s->K * out_len * 2u;
	}
	bytes += (uint64_t)s->K * P3_GC0; /* int8 gate table */
	if (s->gate_flags & P3_FLAG_MSEL)
		bytes += (uint64_t)s->K * P3_GMSEL;
	if (s->gate_flags & P3_FLAG_CONF)
		bytes += (uint64_t)s->K * P3_GCONF;
	return bytes;
}

/* ---- loading -------------------------------------------------------------- */

static bool half_finite(uint16_t h)
{
	return (h & 0x7C00u) != 0x7C00u;
}

/* Dequantize one row into tmp[0..len-1] (profile 1 row encoding, §5.1). */
static taw_status read_row(const uint8_t **pp, uint8_t quant, unsigned len,
    float *tmp)
{
	const uint8_t *p = *pp;
	if (quant == TAW_QUANT_Q8) {
		uint16_t sh = taw_rd16(p);
		if (!half_finite(sh))
			return TAW_E_FORMAT;
		float scale = taw_h2f(sh);
		p += 2;
		for (unsigned i = 0; i < len; i++) {
			int8_t q = (int8_t)p[i];
			if (q == -128)
				return TAW_E_FORMAT;
			tmp[i] = (float)q * scale;
		}
		p += len;
	} else {
		for (unsigned i = 0; i < len; i++) {
			uint16_t h = taw_rd16(p + 2 * i);
			if (!half_finite(h))
				return TAW_E_FORMAT;
			tmp[i] = taw_h2f(h);
		}
		p += 2 * len;
	}
	*pp = p;
	return TAW_OK;
}

/* Shared by profiles 3 and 7; head_len is P3_HEAD or P3_CAL_HEAD. */
static taw_status net_load(p3_net *n, uint8_t quant, const uint8_t *model,
    size_t size, size_t head_len)
{
	p3_tables_init();
	memset(n, 0, sizeof *n);
	n->cal = P3_CAL_IDENTITY;
	if (!p3_quant_supported(quant))
		return TAW_E_UNSUPPORTED;
	if (size < head_len)
		return TAW_E_FORMAT;
	taw_status st = p3_head_parse(model, size, &n->s);
	if (st != TAW_OK)
		return st;
	if (size - head_len != p3_weights_size(&n->s, quant))
		return TAW_E_FORMAT;

	const unsigned H = n->s.hidden, K = n->s.K;
	n->hpad = (H + 7) & ~7u;
	n->vstride = H ? (size_t)n->hpad : P1_XPAD;
	const size_t w1n = H ? (size_t)H * P1_F : 16;
	const size_t vn = (size_t)K * n->vstride;
	n->w1 = aligned_alloc(64, ((w1n * sizeof(float)) + 63) & ~(size_t)63);
	n->v = aligned_alloc(64, ((vn * sizeof(float)) + 63) & ~(size_t)63);
	n->b = calloc(K ? K : 1, sizeof(float));
	n->gc0 = malloc((size_t)K * P3_GC0);
	n->gms = (n->s.gate_flags & P3_FLAG_MSEL)
	    ? malloc((size_t)K * P3_GMSEL) : nullptr;
	n->gcf = (n->s.gate_flags & P3_FLAG_CONF)
	    ? malloc((size_t)K * P3_GCONF) : nullptr;
	if (!n->w1 || !n->v || !n->b || !n->gc0 ||
	    ((n->s.gate_flags & P3_FLAG_MSEL) && !n->gms) ||
	    ((n->s.gate_flags & P3_FLAG_CONF) && !n->gcf)) {
		p3_net_free(n);
		return TAW_E_NOMEM;
	}
	memset(n->w1, 0, w1n * sizeof(float));
	memset(n->v, 0, vn * sizeof(float));

	const uint8_t *p = model + head_len;
	float row[P1_HMAX + P1_F];
	/* Shared hidden layer. */
	for (unsigned j = 0; j < H; j++) {
		st = read_row(&p, quant, P1_F, row);
		if (st != TAW_OK)
			goto fail;
		memcpy(n->w1 + (size_t)j * P1_F, row, P1_F * sizeof(float));
	}
	/* Expert output rows: H weights then the bias (H > 0), or 26 weights. */
	const unsigned out_len = H ? H + 1 : P1_F;
	for (unsigned k = 0; k < K; k++) {
		st = read_row(&p, quant, out_len, row);
		if (st != TAW_OK)
			goto fail;
		memcpy(n->v + (size_t)k * n->vstride, row,
		    (H ? H : P1_F) * sizeof(float));
		if (H)
			n->b[k] = row[H];
	}
	/* Gate tables (raw int8, no scale). */
	memcpy(n->gc0, p, (size_t)K * P3_GC0);
	p += (size_t)K * P3_GC0;
	if (n->gms) {
		memcpy(n->gms, p, (size_t)K * P3_GMSEL);
		p += (size_t)K * P3_GMSEL;
	}
	if (n->gcf) {
		memcpy(n->gcf, p, (size_t)K * P3_GCONF);
		p += (size_t)K * P3_GCONF;
	}
	if ((size_t)(p - model) != size) {
		st = TAW_E_FORMAT;
		goto fail;
	}
	return TAW_OK;
fail:
	p3_net_free(n);
	return st;
}

taw_status p3_net_load(p3_net *n, uint8_t quant, const uint8_t *model,
    size_t size)
{
	return net_load(n, quant, model, size, P3_HEAD);
}

taw_status p3_cal_net_load(p3_net *n, uint8_t quant, const uint8_t *model,
    size_t size)
{
	p3_shape s;
	p3_cal cal;
	taw_status st = p3_cal_head_parse(model, size, &s, &cal);
	if (st != TAW_OK)
		return st;
	st = net_load(n, quant, model, size, P3_CAL_HEAD);
	if (st == TAW_OK)
		n->cal = cal;
	return st;
}

void p3_net_free(p3_net *n)
{
	free(n->w1);
	free(n->v);
	free(n->b);
	free(n->gc0);
	free(n->gms);
	free(n->gcf);
	memset(n, 0, sizeof *n);
}

/* ---- inference ------------------------------------------------------------ */

/* The normative dot product (as in profile 1, §2.2.8). */
static inline float dot4(const float *w, size_t ws, const float *x,
    unsigned len)
{
	float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	for (unsigned i = 0; i < len; i++)
		s[i & 3] += w[i * ws] * x[i];
	return (s[0] + s[1]) + (s[2] + s[3]);
}

int32_t p3_net_eval(const p3_net *n, const float *x, const p1_sel_ctx *sc)
{
	const unsigned K = n->s.K, H = n->s.hidden;
	const uint32_t c0 = sc->c0, msel = sc->msel;

	/* Shared nonlinear hidden layer. */
	float h[P1_HMAX];
	for (unsigned j = 0; j < H; j++) {
		float a = dot4(n->w1 + (size_t)j * P1_F, 1, x, P1_F);
		h[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
	}

	/* Confidence = mean absolute deviation of the 25 stretched inputs,
	 * bucketed into P3_GCONF levels. Fixed order, no libm. */
	unsigned conf = 0;
	if (n->gcf) {
		float mean = 0.0f;
		for (unsigned i = 0; i < P1_NCTX; i++)
			mean += x[i];
		mean *= 1.0f / (float)P1_NCTX;
		float mad = 0.0f;
		for (unsigned i = 0; i < P1_NCTX; i++) {
			float d = x[i] - mean;
			mad += d < 0.0f ? -d : d;
		}
		mad *= 1.0f / (float)P1_NCTX;
		conf = (unsigned)(mad * 4.0f);
		if (conf >= P3_GCONF)
			conf = P3_GCONF - 1;
	}

	/* Deterministic integer gate over c0, match class and confidence. */
	int lg[P3_KMAX], mx = -1000000;
	for (unsigned k = 0; k < K; k++) {
		int l = n->gc0[(size_t)k * P3_GC0 + c0];
		if (n->gms)
			l += n->gms[(size_t)k * P3_GMSEL + (msel & 7)];
		if (n->gcf)
			l += n->gcf[(size_t)k * P3_GCONF + conf];
		lg[k] = l;
		if (l > mx)
			mx = l;
	}
	uint32_t e[P3_KMAX], sum = 0;
	for (unsigned k = 0; k < K; k++) {
		unsigned d = (unsigned)(mx - lg[k]);
		if (d > 511)
			d = 511;
		e[k] = p3_e_tab[d];
		sum += e[k];
	}
	/* Prefix-sum rounding: weights sum to exactly P3_WSUM. */
	uint32_t w[P3_KMAX], prefix = 0;
	for (unsigned k = 0; k < K; k++) {
		prefix += e[k];
		uint32_t pk = (uint32_t)(((uint64_t)P3_WSUM * prefix) / sum);
		w[k] = pk - (k ? ((uint32_t)(((uint64_t)P3_WSUM * (prefix - e[k])) / sum)) : 0u);
	}

	/* Weighted sum of the K experts in fixed order. */
	float z[P3_KMAX];
	for (unsigned k = 0; k < K; k++) {
		const float *v = n->v + (size_t)k * n->vstride;
		z[k] = H ? dot4(v, 1, h, H) + n->b[k] : dot4(v, 1, x, P1_F);
	}
	float acc = 0.0f;
	for (unsigned k = 0; k < K; k++)
		acc += (float)w[k] * z[k];
	float zb = acc * (1.0f / (float)P3_WSUM);
	const float lim = 2047.0f / 256.0f;
	if (zb > lim)
		zb = lim;
	else if (zb < -lim)
		zb = -lim;
	return (int32_t)(zb * 256.0f);
}
