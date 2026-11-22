/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Weight row formats: sizes, decoding (validated), quantization (encoder). */
#include <string.h>

#include "format.h"
#include "half.h"
#include "quant.h"
#include "taw_internal.h"

const int8_t taw_iq4_tab[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113};
const int8_t taw_iq2_tab[4] = {-127, -38, 38, 127};

float taw_e4m3_to_f32(uint8_t b)
{
	const uint32_t s = (uint32_t)(b & 0x80u) << 24;
	const uint32_t e = (b >> 3) & 15u, m = b & 7u;
	if (e == 0) { /* subnormal: m * 2^-9 */
		float f = (float)m * (1.0f / 512.0f);
		return taw_bits_f32(taw_f32_bits(f) | s);
	}
	return taw_bits_f32(s | (e + 120u) << 23 | m << 20);
}

float taw_e2m1_to_f32(uint8_t nib)
{
	static const float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f,
		6.0f};
	float f = v[nib & 7u];
	return nib & 8u ? -f : f;
}

static inline bool e4m3_nan(uint8_t b)
{
	return (b & 0x7Fu) == 0x7Fu;
}

static inline bool half_finite(uint16_t h)
{
	return (h & 0x7C00u) != 0x7C00u;
}

bool taw_quant_is_row_format(uint8_t q)
{
	switch (q) {
	case TAW_QUANT_F16: case TAW_QUANT_Q8: case TAW_QUANT_Q4:
	case TAW_QUANT_Q2: case TAW_QUANT_FP8: case TAW_QUANT_NVFP4:
	case TAW_QUANT_IQ8: case TAW_QUANT_IQ4: case TAW_QUANT_IQ2:
		return true;
	default:
		return false;
	}
}

uint64_t taw_quant_row_bytes(uint8_t q, uint64_t len)
{
	switch (q) {
	case TAW_QUANT_F16: return 2 * len;
	case TAW_QUANT_Q8: case TAW_QUANT_FP8: case TAW_QUANT_IQ8:
		return 2 + len;
	case TAW_QUANT_Q4: case TAW_QUANT_IQ4: return 2 + (len + 1) / 2;
	case TAW_QUANT_Q2: return 4 + (len + 3) / 4;
	case TAW_QUANT_IQ2: return 2 + (len + 3) / 4;
	case TAW_QUANT_NVFP4:
		return 2 + (len + TAW_NVFP4_BLOCK - 1) / TAW_NVFP4_BLOCK +
		    (len + 1) / 2;
	default: return 0;
	}
}

static inline unsigned nib_at(const uint8_t *q, unsigned i)
{
	return q[i >> 1] >> (4 * (i & 1)) & 15u;
}

static inline unsigned crumb_at(const uint8_t *q, unsigned i)
{
	return q[i >> 2] >> (2 * (i & 3)) & 3u;
}

/* Unused bits after the last packed code must be zero. */
static bool pad_ok(const uint8_t *q, unsigned len, unsigned bits)
{
	const unsigned per = 8 / bits, used = len % per;
	if (!used)
		return true;
	return (q[len / per] >> (used * bits)) == 0;
}

taw_status taw_quant_row_decode(uint8_t quant, const uint8_t *p,
    unsigned len, float *out)
{
	if (quant == TAW_QUANT_F16) {
		for (unsigned i = 0; i < len; i++) {
			uint16_t h = taw_rd16(p + 2 * i);
			if (!half_finite(h))
				return TAW_E_FORMAT;
			out[i] = taw_h2f(h);
		}
		return TAW_OK;
	}
	if (!taw_quant_is_row_format(quant))
		return TAW_E_UNSUPPORTED;
	const uint16_t sh = taw_rd16(p);
	if (!half_finite(sh))
		return TAW_E_FORMAT;
	const float d = taw_h2f(sh);
	const uint8_t *q = p + 2;
	switch (quant) {
	case TAW_QUANT_Q8:
		for (unsigned i = 0; i < len; i++) {
			int8_t k = (int8_t)q[i];
			if (k == -128)
				return TAW_E_FORMAT;
			out[i] = (float)k * d;
		}
		break;
	case TAW_QUANT_FP8:
		for (unsigned i = 0; i < len; i++) {
			if (e4m3_nan(q[i]))
				return TAW_E_FORMAT;
			out[i] = taw_e4m3_to_f32(q[i]) * d;
		}
		break;
	case TAW_QUANT_IQ8:
		for (unsigned i = 0; i < len; i++)
			out[i] = (float)taw_iq8_val((int8_t)q[i]) * d;
		break;
	case TAW_QUANT_Q4:
		if (!pad_ok(q, len, 4))
			return TAW_E_FORMAT;
		for (unsigned i = 0; i < len; i++)
			out[i] = (float)((int)nib_at(q, i) - 8) * d;
		break;
	case TAW_QUANT_IQ4:
		if (!pad_ok(q, len, 4))
			return TAW_E_FORMAT;
		for (unsigned i = 0; i < len; i++)
			out[i] = (float)taw_iq4_tab[nib_at(q, i)] * d;
		break;
	case TAW_QUANT_IQ2:
		if (!pad_ok(q, len, 2))
			return TAW_E_FORMAT;
		for (unsigned i = 0; i < len; i++)
			out[i] = (float)taw_iq2_tab[crumb_at(q, i)] * d;
		break;
	case TAW_QUANT_Q2: {
		const uint16_t mh = taw_rd16(q);
		if (!half_finite(mh))
			return TAW_E_FORMAT;
		const float m = taw_h2f(mh);
		q += 2;
		if (!pad_ok(q, len, 2))
			return TAW_E_FORMAT;
		for (unsigned i = 0; i < len; i++) {
			const float t = (float)crumb_at(q, i) * d;
			out[i] = t + m;
		}
		break;
	}
	case TAW_QUANT_NVFP4: {
		const unsigned nb = (len + TAW_NVFP4_BLOCK - 1) / TAW_NVFP4_BLOCK;
		const uint8_t *v = q + nb;
		if (!pad_ok(v, len, 4))
			return TAW_E_FORMAT;
		for (unsigned b = 0; b < nb; b++) {
			if (q[b] & 0x80u || e4m3_nan(q[b]))
				return TAW_E_FORMAT;
			const float bs = taw_e4m3_to_f32(q[b]) * d;
			const unsigned end = (b + 1) * TAW_NVFP4_BLOCK < len ?
			    (b + 1) * TAW_NVFP4_BLOCK : len;
			for (unsigned i = b * TAW_NVFP4_BLOCK; i < end; i++)
				out[i] = taw_e2m1_to_f32((uint8_t)nib_at(v, i)) * bs;
		}
		break;
	}
	default:
		return TAW_E_UNSUPPORTED;
	}
	return TAW_OK;
}

uint8_t taw_quant_from_weights(taw_weights w)
{
	switch (w) {
	case TAW_WEIGHTS_F16: return TAW_QUANT_F16;
	case TAW_WEIGHTS_Q8: return TAW_QUANT_Q8;
	case TAW_WEIGHTS_Q4: return TAW_QUANT_Q4;
	case TAW_WEIGHTS_Q2: return TAW_QUANT_Q2;
	case TAW_WEIGHTS_FP8: return TAW_QUANT_FP8;
	case TAW_WEIGHTS_NVFP4: return TAW_QUANT_NVFP4;
	case TAW_WEIGHTS_IQ8: return TAW_QUANT_IQ8;
	case TAW_WEIGHTS_IQ4: return TAW_QUANT_IQ4;
	case TAW_WEIGHTS_IQ2: return TAW_QUANT_IQ2;
	default: return TAW_QUANT_NONE;
	}
}

/* ---- encoder ------------------------------------------------------------------ */

#if TAW_HAVE_ENCODER

static float absf(float x)
{
	return x < 0 ? -x : x;
}

static float clamp_half(float w)
{
	return w > 65504.0f ? 65504.0f : w < -65504.0f ? -65504.0f : w;
}

/* A positive scale rounded to binary16 (0 if it underflows). */
static uint16_t half_scale(float d)
{
	if (!(d > 0.0f))
		return 0;
	return taw_f2h(clamp_half(d));
}

/* Sorted level set in units of the scale, with the code of each level. */
typedef struct {
	float lv[256];
	uint8_t code[256];
	unsigned n;
} levels;

static void levels_for(uint8_t quant, levels *L)
{
	memset(L, 0, sizeof *L);
	unsigned n = 0;
	switch (quant) {
	case TAW_QUANT_Q8:
		for (int k = -127; k <= 127; k++, n++) {
			L->lv[n] = (float)k;
			L->code[n] = (uint8_t)(int8_t)k;
		}
		break;
	case TAW_QUANT_IQ8:
		for (int k = -128; k <= 127; k++, n++) {
			L->lv[n] = (float)taw_iq8_val((int8_t)k);
			L->code[n] = (uint8_t)(int8_t)k;
		}
		break;
	case TAW_QUANT_Q4:
		for (int k = 0; k < 16; k++, n++) {
			L->lv[n] = (float)(k - 8);
			L->code[n] = (uint8_t)k;
		}
		break;
	case TAW_QUANT_IQ4:
		for (int k = 0; k < 16; k++, n++) {
			L->lv[n] = (float)taw_iq4_tab[k];
			L->code[n] = (uint8_t)k;
		}
		break;
	case TAW_QUANT_IQ2:
		for (int k = 0; k < 4; k++, n++) {
			L->lv[n] = (float)taw_iq2_tab[k];
			L->code[n] = (uint8_t)k;
		}
		break;
	case TAW_QUANT_FP8:
		/* negatives from the largest magnitude down, then 0.. +max */
		for (int b = 0xFE; b >= 0x81; b--, n++) {
			L->lv[n] = taw_e4m3_to_f32((uint8_t)b);
			L->code[n] = (uint8_t)b;
		}
		for (int b = 0x00; b <= 0x7E; b++, n++) {
			L->lv[n] = taw_e4m3_to_f32((uint8_t)b);
			L->code[n] = (uint8_t)b;
		}
		break;
	default:
		break;
	}
	L->n = n;
}

/* Nearest level to w under the decoder's product fl(lv * d), d >= 0.
 * Products are monotone in lv, so binary search then compare neighbours. */
static unsigned nearest(const levels *L, float d, float w, float *val)
{
	unsigned lo = 0, hi = L->n - 1;
	while (lo < hi) {
		unsigned mid = (lo + hi) / 2;
		if (L->lv[mid] * d < w)
			lo = mid + 1;
		else
			hi = mid;
	}
	unsigned best = lo;
	float bv = L->lv[lo] * d;
	if (lo > 0) {
		float v = L->lv[lo - 1] * d;
		if (absf(w - v) <= absf(w - bv)) {
			best = lo - 1;
			bv = v;
		}
	}
	*val = bv;
	return best;
}

static double row_sse(const levels *L, float d, const float *w, unsigned len)
{
	double e = 0.0;
	for (unsigned i = 0; i < len; i++) {
		float v;
		nearest(L, d, w[i], &v);
		double r = (double)w[i] - v;
		e += r * r;
	}
	return e;
}

/* Best binary16 scale for a symmetric level set: a grid around
 * amax / max|level|, then least-squares refinement of the assignment. */
static uint16_t search_scale(const levels *L, const float *w, unsigned len)
{
	float amax = 0.0f;
	for (unsigned i = 0; i < len; i++)
		if (absf(w[i]) > amax)
			amax = absf(w[i]);
	const float lmax = absf(L->lv[0]) > absf(L->lv[L->n - 1]) ?
	    absf(L->lv[0]) : absf(L->lv[L->n - 1]);
	if (amax == 0.0f)
		return 0;
	uint16_t best = half_scale(amax / lmax);
	double be = row_sse(L, taw_h2f(best), w, len);
	for (int k = -12; k <= 6; k++) {
		uint16_t h = half_scale(amax / lmax * (1.0f + 0.04f * (float)k));
		if (!h || h == best)
			continue;
		double e = row_sse(L, taw_h2f(h), w, len);
		if (e < be) {
			be = e;
			best = h;
		}
	}
	for (int it = 0; it < 3; it++) {
		const float d = taw_h2f(best);
		double num = 0.0, den = 0.0;
		for (unsigned i = 0; i < len; i++) {
			float v;
			unsigned j = nearest(L, d, w[i], &v);
			num += (double)w[i] * L->lv[j];
			den += (double)L->lv[j] * L->lv[j];
		}
		if (den <= 0.0)
			break;
		uint16_t h = half_scale((float)(num / den));
		if (!h || h == best)
			break;
		double e = row_sse(L, taw_h2f(h), w, len);
		if (!(e < be))
			break;
		be = e;
		best = h;
	}
	return best;
}

static void put16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

static void encode_levels(uint8_t quant, const float *w, unsigned len,
    uint8_t *out)
{
	levels L;
	levels_for(quant, &L);
	const uint16_t sh = search_scale(&L, w, len);
	const float d = taw_h2f(sh);
	put16(out, sh);
	uint8_t *q = out + 2;
	const unsigned bits = quant == TAW_QUANT_IQ2 ? 2 :
	    quant == TAW_QUANT_Q4 || quant == TAW_QUANT_IQ4 ? 4 : 8;
	memset(q, 0, (size_t)taw_quant_row_bytes(quant, len) - 2);
	for (unsigned i = 0; i < len; i++) {
		float v;
		uint8_t c = L.code[nearest(&L, d, w[i], &v)];
		if (bits == 8)
			q[i] = c;
		else
			q[i * bits / 8] |= (uint8_t)(c << (i * bits % 8));
	}
}

/* Q2: w = fl(fl(q * d) + m), q = 0..3. */
static double q2_assign(float d, float m, const float *w, unsigned len,
    uint8_t *qs)
{
	double e = 0.0;
	for (unsigned i = 0; i < len; i++) {
		unsigned bq = 0;
		float be = 0.0f;
		for (unsigned k = 0; k < 4; k++) {
			const float t = (float)k * d;
			float r = absf(w[i] - (t + m));
			if (k == 0 || r < be) {
				be = r;
				bq = k;
			}
		}
		if (qs)
			qs[i] = (uint8_t)bq;
		e += (double)be * be;
	}
	return e;
}

static void encode_q2(const float *w, unsigned len, uint8_t *out)
{
	float lo = w[0], hi = w[0];
	for (unsigned i = 1; i < len; i++) {
		lo = w[i] < lo ? w[i] : lo;
		hi = w[i] > hi ? w[i] : hi;
	}
	uint16_t bd = half_scale((hi - lo) / 3.0f), bm = taw_f2h(clamp_half(lo));
	double be = q2_assign(taw_h2f(bd), taw_h2f(bm), w, len, nullptr);
	/* Shrink the range from both ends (outliers), then refine by least
	 * squares on the assignment. */
	for (int a = 0; a <= 6; a++)
		for (int b = 0; b <= 6; b++) {
			const float span = hi - lo;
			const float l2 = lo + span * 0.05f * (float)a;
			const float h2 = hi - span * 0.05f * (float)b;
			if (!(h2 > l2))
				continue;
			uint16_t hd = half_scale((h2 - l2) / 3.0f);
			uint16_t hm = taw_f2h(clamp_half(l2));
			double e = q2_assign(taw_h2f(hd), taw_h2f(hm), w, len, nullptr);
			if (e < be) {
				be = e;
				bd = hd;
				bm = hm;
			}
		}
	uint8_t qs[64];
	for (int it = 0; it < 4; it++) {
		q2_assign(taw_h2f(bd), taw_h2f(bm), w, len, qs);
		double s1 = 0, sq = 0, sqq = 0, sw = 0, sqw = 0;
		for (unsigned i = 0; i < len; i++) {
			s1 += 1;
			sq += qs[i];
			sqq += (double)qs[i] * qs[i];
			sw += w[i];
			sqw += (double)qs[i] * w[i];
		}
		const double det = s1 * sqq - sq * sq;
		if (det <= 0.0)
			break;
		const double d = (s1 * sqw - sq * sw) / det;
		const double m = (sw - d * sq) / s1;
		uint16_t hd = half_scale((float)d), hm = taw_f2h(clamp_half((float)m));
		double e = q2_assign(taw_h2f(hd), taw_h2f(hm), w, len, nullptr);
		if (!(e < be))
			break;
		be = e;
		bd = hd;
		bm = hm;
	}
	put16(out, bd);
	put16(out + 2, bm);
	q2_assign(taw_h2f(bd), taw_h2f(bm), w, len, qs);
	uint8_t *q = out + 4;
	memset(q, 0, (len + 3) / 4);
	for (unsigned i = 0; i < len; i++)
		q[i / 4] |= (uint8_t)(qs[i] << (2 * (i % 4)));
}

/* E2M1 levels sorted, with their nibble codes (-0 left out). */
static const float e2m1_lv[15] = {-6, -4, -3, -2, -1.5f, -1, -0.5f, 0, 0.5f,
	1, 1.5f, 2, 3, 4, 6};
static const uint8_t e2m1_code[15] = {15, 14, 13, 12, 11, 10, 9, 0, 1, 2,
	3, 4, 5, 6, 7};

static double nv_block(float bs, const float *w, unsigned n, uint8_t *codes)
{
	levels L;
	L.n = 15;
	memcpy(L.lv, e2m1_lv, sizeof e2m1_lv);
	memcpy(L.code, e2m1_code, sizeof e2m1_code);
	double e = 0.0;
	for (unsigned i = 0; i < n; i++) {
		float v;
		unsigned j = nearest(&L, bs, w[i], &v);
		if (codes)
			codes[i] = L.code[j];
		double r = (double)w[i] - v;
		e += r * r;
	}
	return e;
}

/* Nearest non-negative E4M3 code to x >= 0 (0..0x7E, saturating). */
static uint8_t e4m3_round(float x)
{
	unsigned lo = 0, hi = 0x7E;
	while (lo < hi) {
		unsigned mid = (lo + hi) / 2;
		if (taw_e4m3_to_f32((uint8_t)mid) < x)
			lo = mid + 1;
		else
			hi = mid;
	}
	if (lo > 0 && x - taw_e4m3_to_f32((uint8_t)(lo - 1)) <=
	    taw_e4m3_to_f32((uint8_t)lo) - x)
		lo--;
	return (uint8_t)lo;
}

static double nv_encode(uint16_t dh, const float *w, unsigned len,
    const float *ideal, uint8_t *scales, uint8_t *codes)
{
	const float d = taw_h2f(dh);
	double e = 0.0;
	for (unsigned b = 0; b * TAW_NVFP4_BLOCK < len; b++) {
		const unsigned o = b * TAW_NVFP4_BLOCK;
		const unsigned n = len - o < TAW_NVFP4_BLOCK ? len - o :
		    TAW_NVFP4_BLOCK;
		uint8_t best = 0;
		double be = nv_block(0.0f, w + o, n, nullptr);
		if (d > 0.0f) {
			const uint8_t c0 = e4m3_round(ideal[b] / d);
			for (int k = -2; k <= 2; k++) {
				int c = (int)c0 + k;
				if (c < 0 || c > 0x7E)
					continue;
				double eb = nv_block(taw_e4m3_to_f32((uint8_t)c) * d,
				    w + o, n, nullptr);
				if (eb < be) {
					be = eb;
					best = (uint8_t)c;
				}
			}
		}
		if (scales) {
			scales[b] = best;
			nv_block(taw_e4m3_to_f32(best) * d, w + o, n, codes + o);
		}
		e += be;
	}
	return e;
}

static void encode_nvfp4(const float *w, unsigned len, uint8_t *out)
{
	const unsigned nb = (len + TAW_NVFP4_BLOCK - 1) / TAW_NVFP4_BLOCK;
	float ideal[8], imax = 0.0f;
	for (unsigned b = 0; b < nb; b++) {
		const unsigned o = b * TAW_NVFP4_BLOCK;
		const unsigned n = len - o < TAW_NVFP4_BLOCK ? len - o :
		    TAW_NVFP4_BLOCK;
		float am = 0.0f;
		for (unsigned i = 0; i < n; i++)
			am = absf(w[o + i]) > am ? absf(w[o + i]) : am;
		ideal[b] = am / 6.0f;
		imax = ideal[b] > imax ? ideal[b] : imax;
	}
	uint16_t bd = 0;
	double be = 0.0;
	static const float tops[] = {448, 384, 320, 256, 192, 128, 64, 32};
	for (unsigned t = 0; t < sizeof tops / sizeof *tops; t++) {
		uint16_t h = half_scale(imax / tops[t]);
		if (!h)
			continue;
		double e = nv_encode(h, w, len, ideal, nullptr, nullptr);
		if (!bd || e < be) {
			be = e;
			bd = h;
		}
	}
	put16(out, bd);
	uint8_t codes[64] = {0};
	nv_encode(bd, w, len, ideal, out + 2, codes);
	uint8_t *v = out + 2 + nb;
	memset(v, 0, (len + 1) / 2);
	for (unsigned i = 0; i < len; i++)
		v[i / 2] |= (uint8_t)(codes[i] << (4 * (i % 2)));
}

void taw_quant_row_encode(uint8_t quant, const float *w, unsigned len,
    uint8_t *out)
{
	switch (quant) {
	case TAW_QUANT_F16:
		for (unsigned i = 0; i < len; i++)
			put16(out + 2 * i, taw_f2h(clamp_half(w[i])));
		break;
	case TAW_QUANT_Q2:
		encode_q2(w, len, out);
		break;
	case TAW_QUANT_NVFP4:
		encode_nvfp4(w, len, out);
		break;
	default:
		encode_levels(quant, w, len, out);
		break;
	}
}

#endif
