/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Encoder: choose a profile, fit the profile 1 mixer with AdamW, quantize.
 *
 * Fitting may be sloppy (it is encoder-only and need not be reproducible);
 * the chosen weights are quantized and dequantized exactly as the decoder
 * does it, and the final estimate uses the real inference code. */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "fit_internal.h"
#include "format.h"
#include "half.h"
#include "p1.h"
#include "quant.h"
#include "taw_internal.h"

/* ---- small math without libm ---------------------------------------------- */

static float fsqrt(float x)
{
#if defined(__x86_64__) || defined(__i386__)
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
	if (x <= 0.0f)
		return 0.0f;
	float r = taw_bits_f32((taw_f32_bits(x) >> 1) + 0x1FC00000u);
	for (int i = 0; i < 4; i++)
		r = 0.5f * (r + x / r);
	return r;
#endif
}

/* log2(x) for x > 0, good to ~1e-9; used only for size estimates. */
static double dlog2(double x)
{
	int e = 0;
	while (x >= 2.0) {
		x *= 0.5;
		e++;
	}
	while (x < 1.0) {
		x *= 2.0;
		e--;
	}
	/* ln(x) = 2 atanh(y), y = (x-1)/(x+1) in [0, 1/3) */
	double y = (x - 1.0) / (x + 1.0), y2 = y * y, t = y, s = 0.0;
	for (int k = 1; k < 40; k += 2) {
		s += t / k;
		t *= y2;
	}
	return e + 2.0 * s * 1.4426950408889634;
}

/* cost_tab[p] = -log2(p / 65536) bits, p = 1..65535 */
static float *cost_tab(void)
{
	static _Atomic(float *) tab;
	float *cur = atomic_load(&tab);
	if (cur)
		return cur;
	float *t = malloc(65536 * sizeof *t);
	if (!t)
		return nullptr;
	t[0] = 16.0f;
	for (uint32_t p = 1; p < 65536; p++)
		t[p] = (float)(16.0 - dlog2((double)p));
	if (!atomic_compare_exchange_strong(&tab, &cur, t)) {
		free(t); /* another thread won */
		return cur;
	}
	return t;
}

/* ---- deterministic PRNG (rng_next: fit_internal.h) ----------------------------- */

static float rng_unif(uint64_t *s) /* [-1, 1) */
{
	return (float)(int32_t)(uint32_t)(rng_next(s) >> 32) * (1.0f / 2147483648.0f);
}

/* ---- samples (the type: fit_internal.h) ---------------------------------------- */

static inline p1_sel_ctx sample_sc(const sample *s)
{
	return (p1_sel_ctx){
		.bitpos = (uint8_t)(31 - __builtin_clz(s->c0)),
		.c0 = s->c0,
		.c1 = s->c1,
		.msel = s->msel,
	};
}

static inline void sample_x(const sample *s, float x[P1_XPAD])
{
	for (unsigned i = P1_F; i < P1_XPAD; i++)
		x[i] = 0.0f;
	for (unsigned i = 0; i < P1_NCTX; i++)
		x[i] = (float)s->st[i] * (1.0f / 256.0f);
	x[P1_IN_BIAS] = 1.0f;
}

static unsigned ceil_log2(uint64_t v)
{
	unsigned k = 0;
	while (k < 63 && (1ull << k) < v)
		k++;
	return k;
}

static unsigned clampu(unsigned v, unsigned lo, unsigned hi)
{
	return v < lo ? lo : v > hi ? hi : v;
}

/* ---- H-Net-style sample weighting (encoder only) -------------------------
 *
 * H-Net (arXiv:2507.07955) learns where the context changes and adds an
 * MoE-style ratio loss to hit a target compression ratio. Both ideas are
 * repurposed here as deterministic, encoder-only fitting weights: the
 * decoder, format and quantization are untouched, and every archive with
 * these knobs unset is byte-identical to the shipped fitter.
 *
 * A per-sample context-change score chg[i] compares neighbouring collected
 * samples (the sample array is in input-position order). The "weight" arm
 * multiplies the sample's cross-entropy gradient by
 *
 *     w[i] = clamp( r[i]^p / mean_j(r[j]^p), 1/C, C ),
 *     r[i] = chg[i] / mean_k(chg[k])                      (mean >= 1)
 *
 * so the overall gradient scale stays ~1 while boundary-like samples (large
 * chg) count more. The "ratio" arm is one projected step on the load-
 * balancing loss L = K * sum_k f_k^2, where f_k is the fraction of weighted
 * gradient mass on selector class k: it multiplies each sample's weight by
 *
 *     m[i] = clamp( (target / f_{k(i)})^alpha / Z, 1/C, C )
 *
 * with target = TAW_HNET_TARGET / K (1 = uniform) and Z chosen so the
 * base-weighted mean of m is 1. Both arms compose multiplicatively and are
 * clamped again to [1/C, C]. Only the training gradient is weighted: scoring
 * uses the decoder's real inference on unweighted samples, as before.
 *
 * Knobs (all opt-in; TAW_HNET_MODE unset/off reproduces baseline exactly):
 *   TAW_HNET_MODE   off | weight | boundary | ratio | both (default off)
 *   TAW_HNET_SCORE  st2 | all | byte | c0 | c0c1       (default st2)
 *   TAW_HNET_W      exponent p: -2,-1,-0.5,0,0.5,1,2 (default 1; <0 = inverse)
 *   TAW_HNET_LAG    compare sample i with i-lag        (default 1)
 *   TAW_HNET_CLAMP  clamp C >= 1                       (default 2)
 *   TAW_HNET_Q      boundary top-quantile q            (default 0.1)
 *   TAW_HNET_TARGET target class mass / uniform        (default 1)
 *   TAW_HNET_RATIO  ratio exponent alpha               (default 1)
 *   TAW_HNET_CTX    class context for the ratio arm:
 *                   sel | c0 | c1hi | bitpos           (default sel)
 * GPU fitting is disabled while these knobs are on (the GPU trainer has no
 * notion of sample weights), so an H-Net run always trains on the CPU.
 */

enum { HNET_ST2, HNET_ALL, HNET_BYTE, HNET_C0, HNET_C0C1 };
enum { HNET_CTX_SEL, HNET_CTX_C0, HNET_CTX_C1HI, HNET_CTX_BITPOS };

typedef struct {
	bool on, weight, ratio, hard;
	unsigned score, ctx, lag;
	float p, alpha, clamp, target, q;
} hnet_cfg;

static float hnet_env_float(const char *name, float dflt, float lo, float hi)
{
	const char *s = getenv(name);
	if (!s || !*s)
		return dflt;
	char *end;
	float v = strtof(s, &end);
	if (*end || v != v || v < lo || v > hi)
		return dflt;
	return v;
}

static hnet_cfg hnet_config(void)
{
	hnet_cfg c = {.p = 1.0f, .alpha = 1.0f, .clamp = 2.0f,
		.target = 1.0f, .q = 0.1f, .lag = 1, .score = HNET_ST2,
		.ctx = HNET_CTX_SEL};
	const char *mode = getenv("TAW_HNET_MODE");
	if (!mode || !*mode || !strcmp(mode, "off"))
		return c;
	if (!strcmp(mode, "weight") || !strcmp(mode, "w"))
		c.weight = true;
	else if (!strcmp(mode, "boundary") || !strcmp(mode, "bnd")) {
		c.weight = c.hard = true;
	} else if (!strcmp(mode, "ratio") || !strcmp(mode, "r"))
		c.ratio = true;
	else if (!strcmp(mode, "both") || !strcmp(mode, "b"))
		c.weight = c.ratio = true;
	else
		return c; /* unknown mode: silent no-op */
	c.on = true;
	const char *score = getenv("TAW_HNET_SCORE");
	if (score && *score) {
		if (!strcmp(score, "all")) c.score = HNET_ALL;
		else if (!strcmp(score, "byte")) c.score = HNET_BYTE;
		else if (!strcmp(score, "c0")) c.score = HNET_C0;
		else if (!strcmp(score, "c0c1")) c.score = HNET_C0C1;
	}
	const char *ctx = getenv("TAW_HNET_CTX");
	if (ctx && *ctx) {
		if (!strcmp(ctx, "c0")) c.ctx = HNET_CTX_C0;
		else if (!strcmp(ctx, "c1hi")) c.ctx = HNET_CTX_C1HI;
		else if (!strcmp(ctx, "bitpos")) c.ctx = HNET_CTX_BITPOS;
	}
	{
		const char *s = getenv("TAW_HNET_LAG");
		if (s && *s) {
			char *end;
			long v = strtol(s, &end, 10);
			if (!*end && v >= 1 && v <= 1024)
				c.lag = (unsigned)v;
		}
	}
	c.q = hnet_env_float("TAW_HNET_Q", 0.1f, 0.01f, 0.5f);
	c.p = hnet_env_float("TAW_HNET_W", 1.0f, -2.0f, 2.0f);
	c.alpha = hnet_env_float("TAW_HNET_RATIO", 1.0f, 0.0f, 4.0f);
	c.clamp = hnet_env_float("TAW_HNET_CLAMP", 2.0f, 1.0f, 64.0f);
	c.target = hnet_env_float("TAW_HNET_TARGET", 1.0f, 0.01f, 4.0f);
	/* Only these exponents are exact; anything between is snapped to 1. */
	if (c.p != -2.0f && c.p != -1.0f && c.p != -0.5f && c.p != 0.0f &&
	    c.p != 0.5f && c.p != 1.0f && c.p != 2.0f)
		c.p = 1.0f;
	if (!c.weight)
		c.p = 1.0f;
	if (!c.ratio) {
		c.alpha = 1.0f;
		c.target = 1.0f;
	}
	return c;
}

static float hnet_pow(float r, float p)
{
	if (p == 0.0f || r == 1.0f)
		return 1.0f;
	if (p < 0.0f) {
		float q = hnet_pow(r, -p);
		return q > 0.0f ? 1.0f / q : 1.0f;
	}
	if (p == 0.5f)
		return fsqrt(r);
	if (p == 2.0f)
		return r * r;
	return r; /* p == 1 */
}

static float hnet_clampw(float v, float clamp)
{
	float lo = 1.0f / clamp;
	return v < lo ? lo : v > clamp ? clamp : v;
}

/* chg[i] for i >= lag (0 before that, no predecessor); lag defaults to 1. */
static void hnet_chg(const sample *sm, size_t ns, unsigned score,
    unsigned lag, uint32_t *chg)
{
	if (!ns)
		return;
	for (size_t i = 0; i < ns && i < lag; i++)
		chg[i] = 0;
	for (size_t i = lag; i < ns; i++) {
		const sample *a = &sm[i], *b = &sm[i - lag];
		uint32_t v = 0;
		switch (score) {
		case HNET_ALL:
			for (unsigned q = 0; q < P1_NCTX; q++)
				v += (uint32_t)abs(a->st[q] - b->st[q]);
			break;
		case HNET_BYTE:
			v = (uint32_t)abs((int)a->c1 - (int)b->c1);
			break;
		case HNET_C0:
			v = (uint32_t)abs((int)a->c0 - (int)b->c0);
			break;
		case HNET_C0C1:
			v = (uint32_t)abs((int)a->c0 - (int)b->c0) +
			    (uint32_t)abs((int)a->c1 - (int)b->c1);
			break;
		default: /* HNET_ST2: direct o0 and o1 stretch features */
			v = (uint32_t)abs(a->st[0] - b->st[0]) +
			    (uint32_t)abs(a->st[1] - b->st[1]);
			break;
		}
		chg[i] = v;
	}
}

static int hnet_cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return (x > y) - (x < y);
}

/* The context-change arm of the weight. */
static void hnet_base(const hnet_cfg *c, const uint32_t *chg, size_t ns,
    float *w)
{
	if (!c->weight) {
		for (size_t i = 0; i < ns; i++)
			w[i] = 1.0f;
		return;
	}
	if (c->hard && ns) {
		/* H-Net's binary boundary analogue: the top q of chg values are
		 * boundaries, everything else is not. Normalize to mean 1. */
		uint32_t *sorted = malloc(ns * sizeof *sorted);
		if (sorted) {
			memcpy(sorted, chg, ns * sizeof *sorted);
			qsort(sorted, ns, sizeof *sorted, hnet_cmp_u32);
			uint32_t thr = sorted[(size_t)((double)(ns - 1) *
			    (1.0 - (double)c->q))];
			free(sorted);
			double sw = 0.0;
			for (size_t i = 0; i < ns; i++) {
				w[i] = chg[i] >= thr ? c->clamp : 1.0f / c->clamp;
				sw += w[i];
			}
			float scale = sw > 0.0 ? (float)((double)ns / sw) : 1.0f;
			for (size_t i = 0; i < ns; i++)
				w[i] = hnet_clampw(w[i] * scale, c->clamp);
			return;
		}
	}
	uint64_t sum = 0;
	for (size_t i = 0; i < ns; i++)
		sum += chg[i];
	double mu = ns ? (double)sum / (double)ns : 1.0;
	if (mu < 1.0)
		mu = 1.0;
	/* Raw r^p, normalized to mean 1 before clamping. */
	double mean = 0.0;
	for (size_t i = 0; i < ns; i++)
		mean += hnet_pow((float)((double)chg[i] / mu), c->p);
	mean = ns ? mean / (double)ns : 1.0;
	if (mean <= 0.0)
		mean = 1.0;
	for (size_t i = 0; i < ns; i++) {
		float v = hnet_pow((float)((double)chg[i] / mu), c->p) /
		    (float)mean;
		w[i] = hnet_clampw(v, c->clamp);
	}
}

/* One load-balancing step over a selector's classes: k is the class count,
 * cls the class of each sample, base the incoming weights. Writes a
 * base-weighted-mean-one multiplier per sample. */
static void hnet_balance(const float *base, const size_t *cls, size_t ns,
    unsigned k, float target, float alpha, float clamp, float *mult)
{
	float mass[4096] = {0}; /* largest selector (C0C1HI) has 4096 classes */
	double total = 0.0;
	for (size_t i = 0; i < ns; i++) {
		mass[cls[i]] += base[i];
		total += base[i];
	}
	if (total <= 0.0) {
		for (size_t i = 0; i < ns; i++)
			mult[i] = 1.0f;
		return;
	}
	float t = target / (float)k; /* target as a fraction of uniform mass */
	double z = 0.0;
	for (size_t i = 0; i < ns; i++) {
		float f = (float)((double)mass[cls[i]] / total);
		float v = hnet_pow(f > 0.0f ? t / f : clamp, alpha);
		mult[i] = hnet_clampw(v, clamp);
		z += (double)base[i] * mult[i];
	}
	/* Renormalize so the base-weighted mean multiplier is 1. */
	if (z > 0.0) {
		float scale = (float)(total / z);
		for (size_t i = 0; i < ns; i++) {
			float v = mult[i] * scale;
			mult[i] = hnet_clampw(v, clamp);
		}
	}
}

/* Final per-candidate weights: context-change arm times ratio arm(s). */
static float *hnet_weights(const hnet_cfg *c, const sample *sm, size_t ns,
    const float *base, unsigned sel1, unsigned sel2)
{
	float *w = malloc((ns ? ns : 1) * sizeof *w);
	if (!w)
		return nullptr;
	if (!c->ratio) {
		memcpy(w, base, ns * sizeof *w);
		return w;
	}
	/* The class context: the winning shape uses sel2 = BITPOS (8 classes),
	 * so `sel` is often already balanced; `c0`/`c1hi` expose a skewed one. */
	unsigned ctx = c->ctx == HNET_CTX_SEL ? sel2 :
	    c->ctx == HNET_CTX_C0 ? P1_SEL_C0 :
	    c->ctx == HNET_CTX_C1HI ? P1_SEL_C0C1HI : P1_SEL_BITPOS;
	unsigned k = p1_sel_size[ctx];
	size_t *cls = calloc(ns ? ns : 1, sizeof *cls);
	float *m = calloc(ns ? ns : 1, sizeof *m);
	if (!cls || !m) {
		free(cls);
		free(m);
		memcpy(w, base, ns * sizeof *w);
		return w;
	}
	for (size_t i = 0; i < ns; i++) {
		const p1_sel_ctx sc = sample_sc(&sm[i]);
		cls[i] = p1_sel_value(ctx, &sc);
	}
	hnet_balance(base, cls, ns, k, c->target, c->alpha, c->clamp, m);
	if (c->ctx == HNET_CTX_SEL && sel1 > P1_SEL_NONE &&
	    p1_sel_size[sel1] > 1) {
		float *m1 = malloc((ns ? ns : 1) * sizeof *m1);
		if (m1) {
			for (size_t i = 0; i < ns; i++) {
				const p1_sel_ctx sc = sample_sc(&sm[i]);
				cls[i] = p1_sel_value(sel1, &sc);
			}
			hnet_balance(base, cls, ns, p1_sel_size[sel1],
			    c->target, c->alpha, c->clamp, m1);
			for (size_t i = 0; i < ns; i++)
				m[i] *= m1[i];
			free(m1);
		}
	}
	for (size_t i = 0; i < ns; i++)
		w[i] = hnet_clampw(base[i] * m[i], c->clamp);
	free(cls);
	free(m);
	return w;
}

/* Table geometry from the coding-unit size. */
static void geometry(size_t unit, p1_shape *s)
{
	unsigned lu = ceil_log2(unit ? unit : 1);
	s->lines_log2 = (uint8_t)clampu(lu >= 2 ? lu - 2 : 0, P1_LINES_MIN, 18);
	s->hist_log2 = (uint8_t)clampu(lu, P1_HIST_MIN, 24);
	s->match_log2 = (uint8_t)clampu(lu >= 2 ? lu - 2 : 0, P1_MATCH_MIN, 22);
}

/* One pass of the context models over the input with the decoder's unit
 * resets, keeping an evenly spread set of about `want` bits as training
 * samples. Bit i (of 8n) is kept when floor((i+1) want / 8n) passes an
 * integer, and lands at index floor(i want / 8n): a rule of the position
 * alone, so units can run in parallel and the samples do not depend on the
 * thread count. */
typedef struct {
	const taw_source *in;
	const uint8_t *view; /* the input itself, for memory sources */
	uint64_t n, unit, total, want;
	const p1_shape *geo;
	sample *sm;
	p1_ctx *ctx;    /* [threads], set up on first use */
	uint8_t **buf;  /* [threads] unit buffers (non-memory sources) */
	taw_status *st; /* [units] */
} collect_job;

static inline uint64_t sample_index(uint64_t bit, uint64_t want,
    uint64_t total)
{
#if defined(__SIZEOF_INT128__)
	__extension__ typedef unsigned __int128 u128; /* -Wpedantic */
	return (uint64_t)(((u128)bit * want) / total);
#else
	/* want <= 2^22 and bit <= total: exact while total < 2^42 bits */
	return bit / total * want + bit % total * want / total;
#endif
}

static void collect_unit(void *arg, size_t u, unsigned w)
{
	collect_job *J = arg;
	const uint64_t start = (uint64_t)u * J->unit;
	const uint64_t len = J->n - start < J->unit ? J->n - start : J->unit;
	p1_ctx *c = &J->ctx[w];
	if (!c->arena) {
		if ((J->st[u] = p1_ctx_init(c, J->geo)) != TAW_OK)
			return;
	} else {
		p1_ctx_reset(c);
	}
	const uint8_t *src = J->view ? J->view + start : nullptr;
	if (!src) {
		if (!J->buf[w] && !(J->buf[w] = malloc((size_t)J->unit))) {
			J->st[u] = TAW_E_NOMEM;
			return;
		}
		if ((J->st[u] = J->in->read_at(J->in->ctx, start, J->buf[w],
			 (size_t)len)) != TAW_OK)
			return;
		src = J->buf[w];
	}
	/* Inside each byte the bits are visited in a rotated order for the
	 * selection (the rotation is a hash of the byte's position): a plain
	 * stride over bit positions keeps the same bit of every byte whenever
	 * 8n / want is a multiple of 8, e.g. for any power-of-two input size,
	 * and the mixer then never sees the other seven bit positions. */
	for (uint64_t i = 0; i < len; i++) {
		const uint64_t byte = start + i;
		uint64_t idx[9];
		for (unsigned j = 0; j <= 8; j++)
			idx[j] = sample_index(byte * 8 + j, J->want, J->total);
		const unsigned rot =
		    (unsigned)((byte * 0x9E3779B97F4A7C15ull) >> 61);
		const unsigned ch = src[i];
		for (unsigned bp = 0; bp < 8; bp++) {
			const int v = (ch >> (7 - bp)) & 1;
			const unsigned j = (bp + rot) & 7;
			float xv[32];
			p1_sel_ctx sc;
			p1_ctx_inputs(c, xv, &sc); /* the update needs them */
			if (idx[j + 1] != idx[j]) {
				sample *s = &J->sm[idx[j]];
				for (unsigned q = 0; q < P1_NCTX; q++) /* exact */
					s->st[q] = (int16_t)(xv[q] * 256.0f);
				s->c0 = sc.c0;
				s->c1 = sc.c1;
				s->msel = sc.msel;
				s->bit = (uint8_t)v;
			}
			p1_ctx_update(c, v);
		}
	}
}

static taw_status collect(const taw_source *in, uint64_t unit,
    const p1_shape *geo, uint64_t want, unsigned threads, sample **out,
    size_t *out_n)
{
	const uint64_t n = in->size, total = n * 8;
	if (want > total)
		want = total;
	const uint64_t units = n ? (n + unit - 1) / unit : 0;
	if (threads == 0)
		threads = 1;
	if (threads > units)
		threads = units ? (unsigned)units : 1;
	collect_job J = {.in = in, .view = taw_source_view(in), .n = n,
		.unit = unit, .total = total, .want = want, .geo = geo};
	J.sm = malloc((size_t)(want ? want : 1) * sizeof *J.sm);
	J.ctx = aligned_alloc(_Alignof(p1_ctx), (size_t)threads * sizeof *J.ctx);
	if (J.ctx)
		memset(J.ctx, 0, (size_t)threads * sizeof *J.ctx);
	J.buf = calloc(threads, sizeof *J.buf);
	J.st = calloc(units ? (size_t)units : 1, sizeof *J.st);
	taw_status st = TAW_E_NOMEM;
	if (J.sm && J.ctx && J.buf && J.st) {
		st = taw_parallel(threads, (size_t)units, 0, collect_unit, nullptr,
		    &J);
		for (uint64_t u = 0; u < units && st == TAW_OK; u++)
			st = J.st[u];
	}
	for (unsigned w = 0; J.ctx && w < threads; w++)
		if (J.ctx[w].arena)
			p1_ctx_free(&J.ctx[w]);
	for (unsigned w = 0; J.buf && w < threads; w++)
		free(J.buf[w]);
	free(J.ctx);
	free(J.buf);
	free(J.st);
	if (st != TAW_OK) {
		free(J.sm);
		return st;
	}
	*out = J.sm;
	*out_n = (size_t)want;
	return TAW_OK;
}

/* ---- the trainable network (tnet: fit_internal.h) ------------------------------- */

static void tnet_free(tnet *t)
{
	free(t->w);
	free(t->m);
	free(t->v);
	free(t->g);
	free(t->touched1);
	free(t->touched2);
	free(t->list1);
	free(t->list2);
	memset(t, 0, sizeof *t);
}

static taw_status tnet_init(tnet *t, const p1_shape *s, uint64_t seed)
{
	memset(t, 0, sizeof *t);
	t->s = *s;
	const unsigned H = s->hidden;
	t->s1 = H ? p1_sel_size[s->sel1] : 0;
	t->s2 = p1_sel_size[s->sel2];
	t->row1 = (size_t)H * P1_F;
	t->row2 = H ? H + 1u : P1_F;
	t->n1 = t->s1 * t->row1;
	t->n2 = t->s2 * t->row2;
	size_t np = t->n1 + t->n2;
	t->w = calloc(np, sizeof(float));
	t->m = calloc(np, sizeof(float));
	t->v = calloc(np, sizeof(float));
	t->g = calloc(np, sizeof(float));
	t->touched1 = calloc(t->s1 + 1, 1);
	t->touched2 = calloc(t->s2, 1);
	t->list1 = malloc((t->s1 + 1) * sizeof(uint32_t));
	t->list2 = malloc(t->s2 * sizeof(uint32_t));
	if (!t->w || !t->m || !t->v || !t->g || !t->touched1 || !t->touched2 ||
	    !t->list1 || !t->list2) {
		tnet_free(t);
		return TAW_E_NOMEM;
	}
	uint64_t rs = seed;
	float *w1 = t->w, *w2 = t->w + t->n1;
	if (H) {
		/* Every set starts from the same random init. */
		float base[P1_HMAX * P1_F];
		for (unsigned j = 0; j < H; j++)
			for (unsigned i = 0; i < P1_F; i++)
				base[j * P1_F + i] = (i == P1_IN_BIAS ? 0.0f : 0.15f) +
				    0.3f * rng_unif(&rs);
		float base2[P1_HMAX + 1];
		for (unsigned j = 0; j < H; j++)
			base2[j] = (0.5f + 0.5f * rng_unif(&rs)) / (float)H *
			    (j & 1 ? -1.0f : 1.0f);
		base2[H] = 0.0f;
		for (size_t set = 0; set < t->s1; set++)
			memcpy(w1 + set * t->row1, base, t->row1 * sizeof(float));
		for (size_t set = 0; set < t->s2; set++)
			memcpy(w2 + set * t->row2, base2, t->row2 * sizeof(float));
	} else {
		for (size_t set = 0; set < t->s2; set++)
			for (unsigned i = 0; i < P1_F; i++)
				w2[set * P1_F + i] = i == P1_IN_BIAS ? 0.0f : 0.3f;
	}
	return TAW_OK;
}

static inline float clamp8(float a)
{
	return a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
}

/* Forward + backward for one sample, accumulating into t->g. `wt` is the
 * H-Net sample weight; 1.0 (the default) adds no arithmetic at all. */
static void tnet_step(tnet *t, const sample *smp, float wt)
{
	const unsigned H = t->s.hidden;
	float x[P1_XPAD];
	sample_x(smp, x);
	const p1_sel_ctx sc = sample_sc(smp);
	const uint32_t s2 = t->tied ? 0 : p1_sel_value(t->s.sel2, &sc);
	float *w2 = t->w + t->n1 + s2 * t->row2;
	float *g2 = t->g + t->n1 + s2 * t->row2;
	float z = 0.0f, h[P1_HMAX], a[P1_HMAX];
	const float *w1 = nullptr;
	float *g1 = nullptr;
	if (H) {
		const uint32_t s1 = t->tied ? 0 : p1_sel_value(t->s.sel1, &sc);
		w1 = t->w + s1 * t->row1;
		g1 = t->g + s1 * t->row1;
		for (unsigned j = 0; j < H; j++) {
			float acc = 0.0f;
			const float *r = w1 + j * P1_F;
			for (unsigned i = 0; i < P1_F; i++)
				acc += r[i] * x[i];
			a[j] = acc;
			h[j] = clamp8(acc);
			z += w2[j] * h[j];
		}
		z += w2[H];
	} else {
		for (unsigned i = 0; i < P1_F; i++)
			z += w2[i] * x[i];
	}
	const float lim = 2047.0f / 256.0f;
	float zc = z > lim ? lim : z < -lim ? -lim : z;
	float p = (float)p1_squash((int32_t)(zc * 256.0f)) * (1.0f / 65536.0f);
	float gz = p - (float)smp->bit;
	if (wt != 1.0f)
		gz *= wt;
	if (H) {
		for (unsigned j = 0; j < H; j++) {
			g2[j] += gz * h[j];
			float ga = (a[j] > -8.0f && a[j] < 8.0f) ? gz * w2[j] : 0.0f;
			if (ga != 0.0f) {
				float *gr = g1 + j * P1_F;
				for (unsigned i = 0; i < P1_F; i++)
					gr[i] += ga * x[i];
			}
		}
		g2[H] += gz;
	} else {
		for (unsigned i = 0; i < P1_F; i++)
			g2[i] += gz * x[i];
	}
}

/* End of warm-up: copy set 0 (weights and moments) to every set. */
static void tnet_untie(tnet *t)
{
	float *arr[3] = {t->w, t->m, t->v};
	for (int a = 0; a < 3; a++) {
		float *b = arr[a];
		for (size_t set = 1; set < t->s1; set++)
			memcpy(b + set * t->row1, b, t->row1 * sizeof(float));
		for (size_t set = 1; set < t->s2; set++)
			memcpy(b + t->n1 + set * t->row2, b + t->n1,
			    t->row2 * sizeof(float));
	}
	t->tied = false;
}

typedef struct {
	float lr, b1, b2, eps, wd;
	float b1t, b2t; /* b1^t, b2^t */
} adamw;

static void adam_rows(tnet *t, adamw *o, float *w, float *m, float *v,
    float *g, size_t len, float scale)
{
	const float c1 = 1.0f / (1.0f - o->b1t), c2 = 1.0f / (1.0f - o->b2t);
	for (size_t k = 0; k < len; k++) {
		float gk = g[k] * scale;
		g[k] = 0.0f;
		m[k] = o->b1 * m[k] + (1.0f - o->b1) * gk;
		v[k] = o->b2 * v[k] + (1.0f - o->b2) * gk * gk;
		float mh = m[k] * c1, vh = v[k] * c2;
		w[k] -= o->lr * (mh / (fsqrt(vh) + o->eps) + o->wd * w[k]);
	}
	(void)t;
}

/* Benchmarking aid: shift both fitter RNG seeds. Default 0 = shipped. */
static uint64_t fit_seed_off(void)
{
	const char *s = getenv("TAW_SEED");
	if (!s || !*s)
		return 0;
	char *end;
	unsigned long long v = strtoull(s, &end, 10);
	return *end ? 0 : (uint64_t)v;
}

/* Optional AdamW tuning overrides (for the control arms; not H-Net).
 * Unset variables keep the shipped values, so the default fit is unchanged.
 * TAW_LR1 defaults to TAW_LR0 / 20, the shipped 0.02 -> 0.001 schedule. */
static float fit_env_float(const char *name, float fallback, float lo, float hi)
{
	const char *s = getenv(name);
	if (!s || !*s)
		return fallback;
	char *end = nullptr;
	float v = strtof(s, &end);
	if (end == s || *end || v != v || v < lo || v > hi)
		return fallback;
	return v;
}

/* Train on samples for the given epochs. Batches touch few weight sets, so
 * only those rows are updated ("lazy" AdamW). */
static void tnet_train(tnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed, const float *w)
{
	enum { BATCH = FIT_BATCH };
	const float lr0 = fit_env_float("TAW_LR0", 0.02f, 0.0f, 1.0f);
	float lr1 = fit_env_float("TAW_LR1", lr0 / 20.0f, 0.0f, 1.0f);
	adamw o = {.b1 = fit_env_float("TAW_B1", 0.9f, 0.0f, 0.999999f),
		.b2 = fit_env_float("TAW_B2", 0.999f, 0.0f, 0.999999f),
		.eps = 1e-6f,
		.wd = fit_env_float("TAW_WD", 1e-5f, 0.0f, 1.0f),
		.b1t = 1.0f, .b2t = 1.0f};
	const uint64_t steps_total = (uint64_t)epochs * ((ns + BATCH - 1) / BATCH);
	uint64_t step = 0;
	/* Stride through the samples (fit_stride). */
	uint64_t rs = seed;
	const unsigned H = t->s.hidden;
	/* Warm-up: the first epochs train one shared set, which then seeds
	 * every selected set. */
	const int warm = fit_warm_epochs(epochs);
	t->tied = t->s1 > 1 || t->s2 > 1;
	for (int ep = 0; ep < epochs; ep++) {
		if (t->tied && ep >= warm)
			tnet_untie(t);
		const uint64_t stride = fit_stride(ns);
		uint64_t pos = rng_next(&rs) % (ns ? ns : 1);
		for (size_t done = 0; done < ns;) {
			size_t nb = ns - done < BATCH ? ns - done : BATCH;
			size_t n1 = 0, n2 = 0;
			for (size_t k = 0; k < nb; k++) {
				const size_t si = pos;
				const sample *s = &sm[si];
				pos += stride;
				if (pos >= ns)
					pos %= ns;
				const p1_sel_ctx sc = sample_sc(s);
				uint32_t a2 = t->tied ? 0 : p1_sel_value(t->s.sel2, &sc);
				if (!t->touched2[a2]) {
					t->touched2[a2] = 1;
					t->list2[n2++] = a2;
				}
				if (H) {
					uint32_t a1 = t->tied ? 0 : p1_sel_value(t->s.sel1, &sc);
					if (!t->touched1[a1]) {
						t->touched1[a1] = 1;
						t->list1[n1++] = a1;
					}
				}
				tnet_step(t, s, w ? w[si] : 1.0f);
			}
			done += nb;
			step++;
			float frac = (float)step / (float)steps_total;
			o.lr = lr0 + (lr1 - lr0) * frac;
			o.b1t *= o.b1;
			o.b2t *= o.b2;
			const float scale = 1.0f / (float)nb;
			for (size_t k = 0; k < n1; k++) {
				size_t off = t->list1[k] * t->row1;
				adam_rows(t, &o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row1, scale);
				t->touched1[t->list1[k]] = 0;
			}
			for (size_t k = 0; k < n2; k++) {
				size_t off = t->n1 + t->list2[k] * t->row2;
				adam_rows(t, &o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row2, scale);
				t->touched2[t->list2[k]] = 0;
			}
		}
	}
}

/* ---- quantization / serialization -------------------------------------------- */

static void put_row(taw_buf *b, uint8_t quant, const float *w, size_t len)
{
	uint8_t row[2 * (P1_HMAX + P1_F)];
	const size_t n = (size_t)taw_quant_row_bytes(quant, len);
	taw_quant_row_encode(quant, w, (unsigned)len, row);
	taw_buf_put(b, row, n);
}

static taw_status serialize(const tnet *t, uint8_t quant, taw_buf *b)
{
	uint8_t head[P1_MODEL_HEAD];
	p1_head_write(head, &t->s);
	taw_buf_put(b, head, sizeof head);
	const unsigned H = t->s.hidden;
	for (size_t set = 0; set < t->s1; set++)
		for (unsigned j = 0; j < H; j++)
			put_row(b, quant, t->w + set * t->row1 + j * P1_F, P1_F);
	for (size_t set = 0; set < t->s2; set++)
		put_row(b, quant, t->w + t->n1 + set * t->row2, t->row2);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

/* Mean cost in bits per coded bit, with the decoder's exact inference. */
static taw_status evaluate(const uint8_t *model, size_t size, uint8_t quant,
    const sample *sm, size_t ns, double *bits)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	p1_net net;
	taw_status st = p1_net_load(&net, quant, model, size);
	if (st != TAW_OK)
		return st;
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float x[P1_XPAD];
		sample_x(&sm[k], x);
		const p1_sel_ctx sc = sample_sc(&sm[k]);
		uint32_t p = p1_squash(p1_net_eval(&net, x, &sc));
		sum += ct[sm[k].bit ? p : 65536 - p];
	}
	p1_net_free(&net);
	*bits = ns ? sum / (double)ns : 8.0;
	return TAW_OK;
}

/* ---- candidates and the size heuristic ----------------------------------------- */

typedef struct {
	p1_shape s;
	uint8_t quant;
	uint64_t size; /* model section bytes */
} cand;

static const uint8_t hidden_opts[] = {0, 4, 8, 16};
static const uint8_t sel1_opts[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0,
	P1_SEL_C0C1HI};
static const uint8_t sel2_opts_h[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0};
static const uint8_t sel2_opts_0[] = {P1_SEL_NONE, P1_SEL_BITPOS,
	P1_SEL_MATCH, P1_SEL_C0, P1_SEL_C1, P1_SEL_C0C1HI};

/* Prior preference (higher = fitted first), fitted to full candidate
 * sweeps at levels 1 and 5 over text, a word list, roff, C source and an
 * ELF binary (1 MiB each): the first candidate is on average 0.6% from
 * the best one (it was 3% with the earlier hand-set prior). Shapes near
 * n/500 bytes (at most 128 KiB) are preferred only mildly; no layer-1
 * selector and a bit-position output selector win most often; the 2-bit
 * formats and F16 rarely do. The 4- and 8-bit formats are close, so the
 * size term breaks their ties. */
static double prior(const cand *c, size_t n)
{
	/* Capped: past ~128 KiB the weight tables (a 4096-set selector is
	 * megabytes of floats) fall out of cache and decoding slows down; a
	 * 1.3 GB dump decoded about 3x slower with a 1.8 MB model. */
	double target = (double)n / 500.0 + 96.0;
	if (target > 131072.0)
		target = 131072.0;
	double r = dlog2((double)c->size / target);
	double pr = -0.25 * (r < 0 ? -r : r);
	if (c->s.sel1 == P1_SEL_NONE)
		pr += 0.5;
	if (c->s.sel2 == P1_SEL_BITPOS)
		pr += 0.6;
	switch (c->quant) {
	case TAW_QUANT_Q2: pr -= 0.5; break;
	case TAW_QUANT_IQ2: pr -= 1.0; break;
	case TAW_QUANT_F16: pr -= 0.3; break;
	default: break;
	}
	return pr;
}

static int cmp_prior(const void *a, const void *b)
{
	double x = ((const double *)a)[0], y = ((const double *)b)[0];
	return (x < y) - (x > y);
}

static size_t enumerate(const p1_shape *geo, uint64_t budget, int quant_pref,
    size_t n, cand *out, size_t cap)
{
	size_t k = 0;
	static const uint8_t quants[] = {TAW_QUANT_F16, TAW_QUANT_Q8,
		TAW_QUANT_Q4, TAW_QUANT_Q2, TAW_QUANT_FP8, TAW_QUANT_NVFP4,
		TAW_QUANT_IQ8, TAW_QUANT_IQ4, TAW_QUANT_IQ2};
	const uint8_t want = taw_quant_from_weights((taw_weights)quant_pref);
	for (size_t qi = 0; qi < sizeof quants; qi++) {
		uint8_t q = quants[qi];
		if (want != TAW_QUANT_NONE && q != want)
			continue;
		for (size_t hi = 0; hi < sizeof hidden_opts; hi++) {
			uint8_t H = hidden_opts[hi];
			size_t n1 = H ? sizeof sel1_opts : 1;
			size_t n2 = H ? sizeof sel2_opts_h : sizeof sel2_opts_0;
			for (size_t a = 0; a < n1; a++)
				for (size_t b = 0; b < n2; b++) {
					cand c = {.s = *geo, .quant = q};
					c.s.hidden = H;
					c.s.sel1 = H ? sel1_opts[a] : P1_SEL_NONE;
					c.s.sel2 = H ? sel2_opts_h[b] : sel2_opts_0[b];
					c.size = P1_MODEL_HEAD + p1_weights_size(&c.s, q);
					if (c.size <= budget && k < cap)
						out[k++] = c;
				}
		}
	}
	/* Sort by prior, best first. */
	struct {
		double pr;
		cand c;
	} *tmp = malloc(k * sizeof *tmp);
	if (!tmp)
		return k;
	for (size_t i = 0; i < k; i++) {
		tmp[i].pr = prior(&out[i], n);
		tmp[i].c = out[i];
	}
	qsort(tmp, k, sizeof *tmp, cmp_prior);
	for (size_t i = 0; i < k; i++)
		out[i] = tmp[i].c;
	free(tmp);
	return k;
}

/* Level -> effort. Epochs matter more than samples: an under-trained net
 * loses more than a smaller sample does. */
static const struct {
	uint32_t samples; /* training bits kept */
	uint8_t cands;    /* candidates fitted */
	uint8_t epochs;   /* the first max(1, epochs / 5) are tied warm-up */
} effort[10] = {
	{0, 0, 0},
	{1u << 18, 1, 3},
	{1u << 19, 1, 4},
	{1u << 19, 2, 6},
	{1u << 20, 2, 6},
	{1u << 20, 3, 6},
	{1u << 21, 3, 8},
	{1u << 21, 4, 8},
	{1u << 22, 6, 10},
	{1u << 22, 8, 12},
};

/* Fitting one candidate: train, quantize, score with the real inference. */
typedef struct {
	const cand *cs;
	const sample *sm;
	size_t ns;
	int epochs;
	uint64_t n;
	const hnet_cfg *hnet; /* opt-in H-Net weighting; nullptr = shipped */
	const float *hnet_base; /* [ns] context-change arm, shared by all candidates */
	taw_buf *model; /* [tries] */
	double *est, *bpb;
	taw_status *st;
	tnet *pre; /* [tries] or nullptr: nets already trained (on the GPU) */
} fit_job;

static void fit_one(void *arg, size_t ci, unsigned w)
{
	(void)w;
	fit_job *F = arg;
	tnet t;
	taw_status st = TAW_OK;
	if (F->pre && F->pre[ci].w) {
		t = F->pre[ci];
		F->pre[ci] = (tnet){0};
	} else if ((st = tnet_init(&t, &F->cs[ci].s,
			  0x7A57A11ull + ci + fit_seed_off())) == TAW_OK) {
		float *sw = nullptr; /* sample weights (not the worker index) */
		if (F->hnet && F->hnet->on)
			sw = hnet_weights(F->hnet, F->sm, F->ns, F->hnet_base,
			    F->cs[ci].s.sel1, F->cs[ci].s.sel2);
		if (F->hnet && F->hnet->on && !sw)
			st = TAW_E_NOMEM;
		else {
			tnet_train(&t, F->sm, F->ns, F->epochs,
			    0xC0FFEEull + ci + fit_seed_off(), sw);
			free(sw);
		}
	}
	if (st == TAW_OK) {
		st = serialize(&t, F->cs[ci].quant, &F->model[ci]);
		tnet_free(&t);
	}
	if (st == TAW_OK)
		st = evaluate(F->model[ci].p, F->model[ci].n, F->cs[ci].quant,
		    F->sm, F->ns, &F->bpb[ci]);
	F->est[ci] = F->bpb[ci] * 8.0 * (double)F->n + 8.0 * (double)F->model[ci].n;
	F->st[ci] = st;
}

/* ---- hybrid fitting: the GPU trains some candidates, CPU workers the rest ---- */

typedef struct {
	fit_job *F;
	const size_t *idx; /* the candidates of this list */
} fit_list;

static void fit_listed(void *arg, size_t i, unsigned w)
{
	const fit_list *L = arg;
	fit_one(L->F, L->idx[i], w);
}

typedef struct {
	taw_gpu *g;
	tnet *nets;
	size_t k;
	int epochs;
	const uint64_t *seeds;
	taw_status st;
} gpu_job;

static void *gpu_main(void *arg)
{
	gpu_job *J = arg;
	J->st = taw_gpu_train(J->g, J->nets, J->k, J->epochs, J->seeds);
	return nullptr;
}

static double now_s(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Fit candidates [0, tries). Which ones the GPU trains is a rule of the
 * candidates alone (every shape it supports), so the archive does not depend
 * on the thread count or on timing; CPU workers fit the others meanwhile. A
 * failing GPU falls back to the CPU (TAW_GPU_AUTO) or fails (TAW_GPU_ON). */
static taw_status fit_all(fit_job *F, size_t tries, unsigned threads,
    taw_gpu_mode mode, bool dbg)
{
	const double t0 = dbg ? now_s() : 0.0;
	/* The GPU trainer has no sample weights; H-Net runs always fit on CPU. */
	if (F->hnet && F->hnet->on)
		mode = TAW_GPU_OFF;
	/* AUTO leaves small fits to the CPU: opening a GPU takes about 0.2 s,
	 * more than levels 1-2 spend fitting. A rule of the effort alone. */
	if (mode == TAW_GPU_AUTO &&
	    (uint64_t)F->ns * (uint64_t)F->epochs * tries < (1u << 22)) {
		if (dbg)
			fprintf(stderr, "taw fit: gpu: not worth it for this "
					"little fitting\n");
		mode = TAW_GPU_OFF;
	}
	size_t *idx = malloc(2 * tries * sizeof *idx), ng = 0, nc = 0;
	if (!idx)
		return TAW_E_NOMEM;
	size_t *gi = idx, *ci = idx + tries;
	for (size_t c = 0; c < tries; c++) {
		if (mode != TAW_GPU_OFF && taw_gpu_fits(&F->cs[c].s))
			gi[ng++] = c;
		else
			ci[nc++] = c;
	}
	taw_gpu *g = nullptr;
	taw_status st = TAW_OK;
	if (ng && (st = taw_gpu_open(F->sm, F->ns, dbg, &g)) != TAW_OK) {
		if (mode == TAW_GPU_ON) {
			free(idx);
			return st;
		}
		nc = tries; /* auto: all on the CPU */
		for (size_t c = 0; c < tries; c++)
			ci[c] = c;
		ng = 0;
		st = TAW_OK;
	}
	tnet *nets = ng ? calloc(ng, sizeof *nets) : nullptr;
	uint64_t *seeds = ng ? malloc(ng * sizeof *seeds) : nullptr;
	F->pre = ng ? calloc(tries, sizeof *F->pre) : nullptr;
	if (ng && (!nets || !seeds || !F->pre))
		st = TAW_E_NOMEM;
	for (size_t j = 0; j < ng && st == TAW_OK; j++) {
		st = tnet_init(&nets[j], &F->cs[gi[j]].s,
		    0x7A57A11ull + gi[j] + fit_seed_off());
		seeds[j] = 0xC0FFEEull + gi[j] + fit_seed_off();
	}
	gpu_job J = {.g = g, .nets = nets, .k = ng, .epochs = F->epochs,
		.seeds = seeds, .st = TAW_OK};
	pthread_t th;
	bool spawned = false;
	if (st == TAW_OK && ng) {
		spawned = pthread_create(&th, nullptr, gpu_main, &J) == 0;
		if (!spawned)
			gpu_main(&J);
	}
	if (st == TAW_OK)
		st = taw_parallel(threads, nc, 0, fit_listed, nullptr,
		    &(fit_list){F, ci});
	if (spawned)
		pthread_join(th, nullptr);
	taw_gpu_close(g);
	const double t1 = dbg ? now_s() : 0.0;
	if (st == TAW_OK && ng) {
		if (J.st == TAW_OK) {
			for (size_t j = 0; j < ng; j++) {
				F->pre[gi[j]] = nets[j];
				nets[j] = (tnet){0};
			}
		} else if (mode == TAW_GPU_ON) {
			st = J.st;
		} else if (dbg) {
			fprintf(stderr, "taw fit: gpu: training failed, "
					"fitting on the CPU\n");
		}
		if (st == TAW_OK) /* quantize and score (or fit, after a failure) */
			st = taw_parallel(threads, ng, 0, fit_listed, nullptr,
			    &(fit_list){F, gi});
	}
	if (dbg)
		fprintf(stderr, "taw fit: %zu candidates (%zu on the GPU): "
				"training %.2f s, total %.2f s\n",
		    tries, J.st == TAW_OK ? ng : 0, t1 - t0, now_s() - t0);
	for (size_t j = 0; j < ng; j++)
		tnet_free(&nets[j]);
	for (size_t c = 0; F->pre && c < tries; c++)
		tnet_free(&F->pre[c]);
	free(F->pre);
	F->pre = nullptr;
	free(nets);
	free(seeds);
	free(idx);
	return st;
}

taw_status taw_model_fit(const uint8_t *src, size_t n, size_t unit,
    const taw_params *p, taw_model_choice *out)
{
	taw_source in;
	taw_source_memory(&in, src, n);
	return taw_model_fit_source(&in, unit, p, out);
}

taw_status taw_model_fit_source(const taw_source *in, uint64_t unit,
    const taw_params *p, taw_model_choice *out)
{
	*out = (taw_model_choice){.profile = TAW_PROFILE_INT_ORDER0,
		.quant = TAW_QUANT_NONE};
	if (p->predictor == TAW_PREDICTOR_ORDER0)
		return TAW_OK;
	const uint64_t n = in->size;
	const bool forced = p->predictor == TAW_PREDICTOR_MIX;
	const uint64_t budget = n / 32;
	if (unit == 0 || unit > n)
		unit = n;

	p1_shape geo = {0};
	geometry((size_t)unit, &geo);
	cand cs[1024];
	size_t ncand = enumerate(&geo, budget, p->weights, (size_t)n, cs, 1024);
	if (ncand == 0)
		return forced ? TAW_E_ARG : TAW_OK;

	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	sample *sm;
	size_t ns;
	taw_status st = collect(in, unit, &geo, effort[lv].samples, threads,
	    &sm, &ns);
	if (st != TAW_OK)
		return st;

	const char *dbg = getenv("TAW_FIT_DEBUG");
	size_t tries = effort[lv].cands;
	if (dbg && strchr(dbg, 'a'))
		tries = ncand;
	if (tries > ncand)
		tries = ncand;

	hnet_cfg hnet = hnet_config();
	uint32_t *chg = nullptr;
	float *base = nullptr;
	if (hnet.on) {
		chg = malloc((ns ? ns : 1) * sizeof *chg);
		base = malloc((ns ? ns : 1) * sizeof *base);
		if (!chg || !base)
			st = TAW_E_NOMEM;
		else {
			hnet_chg(sm, ns, hnet.score, hnet.lag, chg);
			hnet_base(&hnet, chg, ns, base);
			if (dbg)
				fprintf(stderr, "taw fit: hnet: mode=%s%s%s score=%u "
				    "p=%.2f C=%.2f target=%.2f alpha=%.2f ctx=%u "
				    "lag=%u q=%.2f\n",
				    hnet.weight ? (hnet.hard ? "b" : "w") : "",
				    hnet.ratio ? "r" : "",
				    hnet.weight || hnet.ratio ? "" : "?",
				    hnet.score, hnet.p, hnet.clamp, hnet.target,
				    hnet.alpha, hnet.ctx, hnet.lag, hnet.q);
		}
	}

	fit_job F = {.cs = cs, .sm = sm, .ns = ns, .epochs = effort[lv].epochs,
		.n = n, .hnet = hnet.on ? &hnet : nullptr,
		.hnet_base = hnet.on ? base : nullptr};
	if (st == TAW_OK) {
		F.model = calloc(tries, sizeof *F.model);
		F.est = calloc(tries, sizeof *F.est);
		F.bpb = calloc(tries, sizeof *F.bpb);
		F.st = calloc(tries, sizeof *F.st);
		if (!F.model || !F.est || !F.bpb || !F.st)
			st = TAW_E_NOMEM;
		else
			st = fit_all(&F, tries, threads, p->gpu, dbg != nullptr);
	}
	size_t best = SIZE_MAX;
	for (size_t ci = 0; st == TAW_OK && ci < tries; ci++) {
		if ((st = F.st[ci]) != TAW_OK)
			break;
		if (dbg)
			fprintf(stderr,
			    "taw fit: H=%u sel1=%u sel2=%u q=%u model=%zu "
			    "bpc=%.4f est=%.0f B\n",
			    cs[ci].s.hidden, cs[ci].s.sel1, cs[ci].s.sel2,
			    cs[ci].quant, F.model[ci].n, F.bpb[ci] * 8.0,
			    F.est[ci] / 8.0);
		if (best == SIZE_MAX || F.est[ci] < F.est[best])
			best = ci;
	}
	if (st == TAW_OK) {
		out->profile = TAW_PROFILE_F32_STRICT_MIX;
		out->quant = cs[best].quant;
		out->model = F.model[best].p;
		out->model_size = F.model[best].n;
		F.model[best] = (taw_buf){0};
	}
	for (size_t ci = 0; F.model && ci < tries; ci++)
		taw_buf_free(&F.model[ci]);
	free(F.model);
	free(F.est);
	free(F.bpb);
	free(F.st);
	free(chg);
	free(base);
	free(sm);
	return st;
}
