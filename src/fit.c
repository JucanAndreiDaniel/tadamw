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
#include "p3.h"
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

/* Forward + backward for one sample, accumulating into t->g. */
static void tnet_step(tnet *t, const sample *smp)
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

static void put_row(taw_buf *b, uint8_t quant, const float *w, size_t len);

/* ---- profile 3: soft-gated mixture of experts ------------------------------
 * Fitting is encoder-only and need not be reproducible. The gate is trained
 * in natural-log units and serialized as int8 logits scaled by 8/ln2, so the
 * integer softmax E[d] = 256*2^(-d/8) reproduces the trained probabilities.
 * The final score always uses the real decoder inference (p3_net_eval). */

#define P3_GATE_T (11.54156033f) /* 8 / ln(2) */

/* exp(x) for x <= 0, range-reduced by x/16 then a degree-6 Taylor series
 * and four squarings. Fitting-only; no determinism requirement. */
static float fexp_neg(float x)
{
	float r = x * (1.0f / 16.0f);
	float r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
	float e = 1.0f + r + r2 * 0.5f + r2 * r * (1.0f / 6.0f) +
	    r4 * (1.0f / 24.0f) + r4 * r * (1.0f / 120.0f) +
	    r6 * (1.0f / 720.0f);
	e *= e; e *= e; e *= e; e *= e;
	return e;
}

typedef struct {
	float *w, *m, *v, *g;
	size_t n;
} pgroup;

static void pg_free(pgroup *p)
{
	free(p->w); free(p->m); free(p->v); free(p->g);
	memset(p, 0, sizeof *p);
}

static bool pg_init(pgroup *p, size_t n)
{
	*p = (pgroup){0};
	if (n) {
		p->w = calloc(n, sizeof(float));
		p->m = calloc(n, sizeof(float));
		p->v = calloc(n, sizeof(float));
		p->g = calloc(n, sizeof(float));
		if (!p->w || !p->m || !p->v || !p->g) {
			pg_free(p);
			return false;
		}
	}
	p->n = n;
	return true;
}

static void pg_step(pgroup *p, adamw *o, float scale)
{
	if (p->n)
		adam_rows(nullptr, o, p->w, p->m, p->v, p->g, p->n, scale);
}

typedef struct {
	p3_shape s;
	unsigned K, H;
	pgroup w1, v, g0, gm, gcf;
} p3t;

static void p3t_free(p3t *t)
{
	pg_free(&t->w1); pg_free(&t->v); pg_free(&t->g0); pg_free(&t->gm);
	pg_free(&t->gcf);
	memset(t, 0, sizeof *t);
}

static taw_status p3t_init(p3t *t, const p3_shape *s, uint64_t seed)
{
	memset(t, 0, sizeof *t);
	t->s = *s;
	t->K = s->K;
	t->H = s->hidden;
	const size_t n1 = (size_t)t->H * P1_F;
	const size_t rowv = t->H ? t->H + 1u : P1_F;
	const size_t nv = (size_t)t->K * rowv;
	const size_t n0 = (size_t)t->K * P3_GC0;
	const size_t nm = (s->gate_flags & P3_FLAG_MSEL)
	    ? (size_t)t->K * P3_GMSEL : 0;
	const size_t nf = (s->gate_flags & P3_FLAG_CONF)
	    ? (size_t)t->K * P3_GCONF : 0;
	if (!pg_init(&t->w1, n1) || !pg_init(&t->v, nv) ||
	    !pg_init(&t->g0, n0) || (nm && !pg_init(&t->gm, nm)) ||
	    (nf && !pg_init(&t->gcf, nf))) {
		p3t_free(t);
		return TAW_E_NOMEM;
	}
	uint64_t rs = seed;
	for (unsigned j = 0; j < t->H; j++)
		for (unsigned i = 0; i < P1_F; i++)
			t->w1.w[j * P1_F + i] =
			    (i == P1_IN_BIAS ? 0.0f : 0.15f) + 0.3f * rng_unif(&rs);
	const bool fixed0 = (s->gate_flags & P3_FLAG_CONF) != 0;
	for (unsigned k = 0; k < t->K; k++)
		for (unsigned i = 0; i < rowv; i++) {
			if (fixed0 && k == 0)
				t->v.w[i] = i < P1_NCTX ? 1.0f / (float)P1_NCTX : 0.0f;
			else
				t->v.w[k * rowv + i] =
				    (t->H && i == t->H) ? 0.0f : 0.2f * rng_unif(&rs);
		}
	for (size_t i = 0; i < n0; i++)
		t->g0.w[i] = 0.5f * rng_unif(&rs);
	for (size_t i = 0; i < nm; i++)
		t->gm.w[i] = 0.5f * rng_unif(&rs);
	for (size_t i = 0; i < nf; i++)
		t->gcf.w[i] = 0.5f * rng_unif(&rs);
	return TAW_OK;
}

static void p3t_train(p3t *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed)
{
	enum { BATCH = 256 };
	adamw o = {.b1 = 0.9f, .b2 = 0.999f, .eps = 1e-6f, .wd = 1e-5f,
		.b1t = 1.0f, .b2t = 1.0f};
	const float lr0 = 0.02f, lr1 = 0.001f;
	const uint64_t steps_total = (uint64_t)epochs *
	    ((ns + BATCH - 1) / BATCH);
	uint64_t step = 0, rs = seed;
	const unsigned K = t->K, H = t->H;
	const size_t rowv = H ? H + 1u : P1_F;
	const unsigned flen = H ? H + 1u : P1_F;

	for (int ep = 0; ep < epochs; ep++) {
		uint64_t stride = (uint64_t)((double)ns * 0.6180339887) | 1;
		while (ns > 1) {
			uint64_t a = stride, b = ns;
			while (b) {
				uint64_t r = a % b;
				a = b; b = r;
			}
			if (a == 1)
				break;
			stride += 2;
		}
		uint64_t pos = rng_next(&rs) % (ns ? ns : 1);
		for (size_t done = 0; done < ns;) {
			size_t nb = ns - done < BATCH ? ns - done : BATCH;
			for (size_t q = 0; q < nb; q++) {
				const sample *s = &sm[pos];
				pos += stride;
				if (pos >= ns)
					pos %= ns;
				float x[P1_XPAD];
				sample_x(s, x);
				const p1_sel_ctx sc = sample_sc(s);

				/* shared hidden features (+ bias feature) */
				float feat[P1_F], ha[P1_HMAX];
				if (H) {
					for (unsigned j = 0; j < H; j++) {
						float acc = 0.0f;
						const float *r = t->w1.w + (size_t)j * P1_F;
						for (unsigned i = 0; i < P1_F; i++)
							acc += r[i] * x[i];
						ha[j] = acc;
						feat[j] = clamp8(acc);
					}
					feat[H] = 1.0f;
				} else {
					memcpy(feat, x, P1_F * sizeof(float));
				}

				/* confidence bucket (mean absolute deviation) */
				unsigned cbf = 0;
				if (t->gcf.n) {
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
					cbf = (unsigned)(mad * 4.0f);
					if (cbf >= P3_GCONF)
						cbf = P3_GCONF - 1;
				}

				/* soft gate in natural-log units */
				float lg[P3_KMAX], mx = -1.0e30f;
				for (unsigned k = 0; k < K; k++) {
					float l = t->g0.w[(size_t)k * P3_GC0 + sc.c0];
					if (t->gm.n)
						l += t->gm.w[(size_t)k * P3_GMSEL +
						    (sc.msel & 7)];
					if (t->gcf.n)
						l += t->gcf.w[(size_t)k * P3_GCONF +
						    cbf];
					lg[k] = l;
					if (l > mx)
						mx = l;
				}
				float pg[P3_KMAX], den = 0.0f;
				for (unsigned k = 0; k < K; k++) {
					pg[k] = fexp_neg(lg[k] - mx);
					den += pg[k];
				}
				for (unsigned k = 0; k < K; k++)
					pg[k] /= den;

				float z[P3_KMAX], zb = 0.0f;
				for (unsigned k = 0; k < K; k++) {
					const float *r = t->v.w + (size_t)k * rowv;
					float acc = 0.0f;
					for (unsigned i = 0; i < flen; i++)
						acc += r[i] * feat[i];
					z[k] = acc;
					zb += pg[k] * acc;
				}
				const float lim = 2047.0f / 256.0f;
				float zc = zb > lim ? lim : zb < -lim ? -lim : zb;
				float p = (float)p1_squash((int32_t)(zc * 256.0f)) *
				    (1.0f / 65536.0f);
				float gz = p - (float)s->bit;

				for (unsigned k = 0; k < K; k++) {
					if (!(t->gcf.n && k == 0)) {
						float *gv = t->v.g + (size_t)k * rowv;
						for (unsigned i = 0; i < flen; i++)
							gv[i] += gz * pg[k] * feat[i];
					}
					float gl = 0.25f * gz * pg[k] * (z[k] - zb);
					t->g0.g[(size_t)k * P3_GC0 + sc.c0] += gl;
					if (t->gm.n)
						t->gm.g[(size_t)k * P3_GMSEL +
						    (sc.msel & 7)] += gl;
					if (t->gcf.n)
						t->gcf.g[(size_t)k * P3_GCONF +
						    cbf] += gl;
				}
				if (H) {
					for (unsigned j = 0; j < H; j++) {
						float dh = 0.0f;
						for (unsigned k = 0; k < K; k++)
							dh += pg[k] * t->v.w[
							    (size_t)k * rowv + j];
						if (ha[j] > -8.0f && ha[j] < 8.0f) {
							float *gw = t->w1.g +
							    (size_t)j * P1_F;
							for (unsigned i = 0; i < P1_F; i++)
								gw[i] += gz * dh * x[i];
						}
					}
				}
			}
			done += nb;
			step++;
			o.lr = lr0 + (lr1 - lr0) *
			    ((float)step / (float)steps_total);
			o.b1t *= o.b1;
			o.b2t *= o.b2;
			const float scale = 1.0f / (float)nb;
			pg_step(&t->w1, &o, scale);
			pg_step(&t->v, &o, scale);
			pg_step(&t->g0, &o, scale);
			pg_step(&t->gm, &o, scale);
			pg_step(&t->gcf, &o, scale);
			/* Keep logits soft: int8 serialization and the E[d] table
			 * resolve differences up to ~64 natural-log units. */
			for (size_t i = 0; i < t->g0.n; i++) {
				if (t->g0.w[i] > 8.0f) t->g0.w[i] = 8.0f;
				else if (t->g0.w[i] < -8.0f) t->g0.w[i] = -8.0f;
			}
			for (size_t i = 0; i < t->gm.n; i++) {
				if (t->gm.w[i] > 8.0f) t->gm.w[i] = 8.0f;
				else if (t->gm.w[i] < -8.0f) t->gm.w[i] = -8.0f;
			}
			for (size_t i = 0; i < t->gcf.n; i++) {
				if (t->gcf.w[i] > 8.0f) t->gcf.w[i] = 8.0f;
				else if (t->gcf.w[i] < -8.0f) t->gcf.w[i] = -8.0f;
			}
		}
	}
}

static void put_i8row(taw_buf *b, const float *g, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		float v = g[i] * P3_GATE_T;
		if (v > 127.0f) v = 127.0f;
		if (v < -128.0f) v = -128.0f;
		int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
		taw_buf_put_u8(b, (uint8_t)(int8_t)q);
	}
}

static taw_status p3_serialize(const p3t *t, uint8_t quant, taw_buf *b)
{
	uint8_t head[P3_HEAD];
	p3_head_write(head, &t->s);
	taw_buf_put(b, head, sizeof head);
	const unsigned H = t->H;
	const size_t rowv = H ? H + 1u : P1_F;
	if (H)
		for (unsigned j = 0; j < H; j++)
			put_row(b, quant, t->w1.w + (size_t)j * P1_F, P1_F);
	for (unsigned k = 0; k < t->K; k++)
		put_row(b, quant, t->v.w + (size_t)k * rowv, rowv);
	for (unsigned k = 0; k < t->K; k++)
		put_i8row(b, t->g0.w + (size_t)k * P3_GC0, P3_GC0);
	if (t->gm.n)
		for (unsigned k = 0; k < t->K; k++)
			put_i8row(b, t->gm.w + (size_t)k * P3_GMSEL, P3_GMSEL);
	if (t->gcf.n)
		for (unsigned k = 0; k < t->K; k++)
			put_i8row(b, t->gcf.w + (size_t)k * P3_GCONF, P3_GCONF);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

static taw_status p3_evaluate(const uint8_t *model, size_t size,
    uint8_t quant, const sample *sm, size_t ns, double *bits)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	p3_net net;
	taw_status st = p3_net_load(&net, quant, model, size);
	if (st != TAW_OK)
		return st;
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float x[P1_XPAD];
		sample_x(&sm[k], x);
		const p1_sel_ctx sc = sample_sc(&sm[k]);
		uint32_t p = p1_squash(p3_net_eval(&net, x, &sc));
		sum += ct[sm[k].bit ? p : 65536 - p];
	}
	p3_net_free(&net);
	*bits = ns ? sum / (double)ns : 8.0;
	return TAW_OK;
}

/* Train on samples for the given epochs. Batches touch few weight sets, so
 * only those rows are updated ("lazy" AdamW). */
static void tnet_train(tnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed)
{
	enum { BATCH = FIT_BATCH };
	adamw o = {.b1 = 0.9f, .b2 = 0.999f, .eps = 1e-6f, .wd = 1e-5f,
		.b1t = 1.0f, .b2t = 1.0f};
	const float lr0 = 0.02f, lr1 = 0.001f;
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
				const sample *s = &sm[pos];
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
				tnet_step(t, s);
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

/* Profile 3 candidates. Ordered by expected strength: Q8, a 4-unit hidden
 * layer and the largest expert count first. */
typedef struct {
	p3_shape s;
	uint8_t quant;
	uint64_t size;
} p3cand;

static size_t p3_enumerate(const p1_shape *geo, uint64_t budget,
    int quant_pref, p3cand *out, size_t cap)
{
	static const uint8_t qs[] = {TAW_QUANT_Q8, TAW_QUANT_F16};
	static const uint8_t ks[] = {4, 3, 2};
	static const struct { uint8_t h, flags; } vs[] = {
		{0, P3_FLAG_MSEL | P3_FLAG_CONF}, /* fixed context expert */
		{4, P3_FLAG_MSEL},
		{0, P3_FLAG_MSEL},
	};
	size_t k = 0;
	for (size_t qi = 0; qi < 2; qi++) {
		uint8_t q = qs[qi];
		if (quant_pref == TAW_WEIGHTS_F16 && q != TAW_QUANT_F16)
			continue;
		if (quant_pref == TAW_WEIGHTS_Q8 && q != TAW_QUANT_Q8)
			continue;
		for (size_t ki = 0; ki < 3; ki++)
			for (size_t vi = 0; vi < 3; vi++) {
				p3cand c = {.quant = q};
				c.s = (p3_shape){
					.K = ks[ki], .hidden = vs[vi].h,
					.lines_log2 = geo->lines_log2,
					.hist_log2 = geo->hist_log2,
					.match_log2 = geo->match_log2,
					.gate_flags = vs[vi].flags,
				};
				c.size = P3_HEAD + p3_weights_size(&c.s, q);
				if (c.size <= budget && k < cap)
					out[k++] = c;
			}
	}
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
	} else if ((st = tnet_init(&t, &F->cs[ci].s, 0x7A57A11ull + ci)) == TAW_OK) {
		tnet_train(&t, F->sm, F->ns, F->epochs, 0xC0FFEEull + ci);
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
		st = tnet_init(&nets[j], &F->cs[gi[j]].s, 0x7A57A11ull + gi[j]);
		seeds[j] = 0xC0FFEEull + gi[j];
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

/* Fitting one profile 3 candidate. */
typedef struct {
	const p3cand *cs;
	const sample *sm;
	size_t ns;
	int epochs;
	uint64_t n;
	taw_buf *model;
	double *est, *bpb;
	taw_status *st;
} p3fit_job;

static void p3fit_one(void *arg, size_t ci, unsigned w)
{
	(void)w;
	p3fit_job *J = arg;
	p3t t;
	taw_status st = p3t_init(&t, &J->cs[ci].s, 0x5A17ull + ci);
	if (st == TAW_OK) {
		p3t_train(&t, J->sm, J->ns, J->epochs, 0xBEEFull + ci);
		st = p3_serialize(&t, J->cs[ci].quant, &J->model[ci]);
		p3t_free(&t);
	}
	if (st == TAW_OK)
		st = p3_evaluate(J->model[ci].p, J->model[ci].n,
		    J->cs[ci].quant, J->sm, J->ns, &J->bpb[ci]);
	J->est[ci] = J->bpb[ci] * 8.0 * (double)J->n +
	    8.0 * (double)J->model[ci].n;
	J->st[ci] = st;
}

/* Codelength of a middle slice with the real predictor (context models,
 * squash, APM and range coder), including the whole model section. The
 * sample-based estimate above ignores the APM stages, so it is not reliable
 * for comparing two different mixers. */
static uint64_t real_cost(const taw_model_choice *c, const taw_source *in,
    uint64_t n, uint64_t unit)
{
	const uint64_t slice = n < (1u << 20) ? n : (1u << 20);
	const uint64_t off = (n - slice) / 2;
	uint8_t *buf = malloc((size_t)slice);
	if (!buf)
		return UINT64_MAX;
	taw_status st = in->read_at(in->ctx, off, buf, (size_t)slice);
	uint64_t cost = UINT64_MAX;
	taw_pred pr;
	if (st == TAW_OK && taw_pred_init(&pr, c->profile, c->quant, c->model,
	    c->model_size) == TAW_OK) {
		taw_buf out = {0};
		taw_rc_enc e;
		taw_rc_enc_init(&e, &out);
		for (uint64_t pos = 0; pos < slice; pos += unit) {
			size_t len = (size_t)((slice - pos < unit)
			    ? slice - pos : unit);
			taw_pred_reset(&pr);
			taw_pred_encode(&pr, &e, buf + pos, len);
		}
		taw_rc_enc_flush(&e);
		if (!out.oom)
			cost = (uint64_t)out.n + c->model_size;
		taw_buf_free(&out);
		taw_pred_free(&pr);
	}
	free(buf);
	return cost;
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

	fit_job F = {.cs = cs, .sm = sm, .ns = ns, .epochs = effort[lv].epochs,
		.n = n};
	F.model = calloc(tries, sizeof *F.model);
	F.est = calloc(tries, sizeof *F.est);
	F.bpb = calloc(tries, sizeof *F.bpb);
	F.st = calloc(tries, sizeof *F.st);
	if (!F.model || !F.est || !F.bpb || !F.st)
		st = TAW_E_NOMEM;
	else
		st = fit_all(&F, tries, threads, p->gpu, dbg != nullptr);
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
	}

	/* Fit the soft-gated MoE candidates on the same samples and keep the
	 * profile (1 or 3) with the smaller estimated archive. */
	p3cand pcs[16];
	size_t npc = p3_enumerate(&geo, budget, p->weights, pcs, 16);
	size_t ptries = dbg && strchr(dbg, 'a') ? npc
	    : (effort[lv].cands > 4 ? effort[lv].cands : 4);
	if (ptries > npc)
		ptries = npc;
	p3fit_job G = {0};
	if (st == TAW_OK && npc && ptries) {
		G = (p3fit_job){.cs = pcs, .sm = sm, .ns = ns,
			.epochs = effort[lv].epochs, .n = n};
		G.model = calloc(ptries, sizeof *G.model);
		G.est = calloc(ptries, sizeof *G.est);
		G.bpb = calloc(ptries, sizeof *G.bpb);
		G.st = calloc(ptries, sizeof *G.st);
		if (!G.model || !G.est || !G.bpb || !G.st)
			st = TAW_E_NOMEM;
		else
			st = taw_parallel(threads, ptries, 0, p3fit_one,
			    nullptr, &G);
	}
	/* Rank every fitted candidate by its real coded cost over the whole
	 * input. The sample estimate omits the APM stages, so it can rank two
	 * mixers differently from the range coder that actually writes them. */
	uint64_t best_cost = UINT64_MAX;
	bool best_is_p3 = false;
	size_t best_i = SIZE_MAX;
	for (size_t ci = 0; st == TAW_OK && ci < tries; ci++) {
		if (F.st[ci] != TAW_OK)
			continue;
		taw_model_choice c = {.profile = TAW_PROFILE_F32_STRICT_MIX,
			.quant = cs[ci].quant, .model = F.model[ci].p,
			.model_size = F.model[ci].n};
		uint64_t cost = real_cost(&c, in, n, unit);
		if (dbg)
			fprintf(stderr,
			    "taw real p1[%zu] H=%u q=%u model=%zu cost=%llu\n",
			    ci, cs[ci].s.hidden, cs[ci].quant,
			    F.model[ci].n, (unsigned long long)cost);
		if (cost < best_cost) {
			best_cost = cost; best_is_p3 = false; best_i = ci;
		}
	}
	for (size_t ci = 0; st == TAW_OK && ci < ptries; ci++) {
		if (G.st[ci] != TAW_OK)
			continue;
		taw_model_choice c = {.profile = TAW_PROFILE_SOFT_GATED,
			.quant = pcs[ci].quant, .model = G.model[ci].p,
			.model_size = G.model[ci].n};
		uint64_t cost = real_cost(&c, in, n, unit);
		if (dbg)
			fprintf(stderr,
			    "taw real p3[%zu] K=%u H=%u q=%u model=%zu "
			    "cost=%llu\n",
			    ci, pcs[ci].s.K, pcs[ci].s.hidden, pcs[ci].quant,
			    G.model[ci].n, (unsigned long long)cost);
		if (cost < best_cost) {
			best_cost = cost; best_is_p3 = true; best_i = ci;
		}
	}
	if (st == TAW_OK && best_i != SIZE_MAX) {
		if (best_is_p3) {
			out->profile = TAW_PROFILE_SOFT_GATED;
			out->quant = pcs[best_i].quant;
			out->model = G.model[best_i].p;
			out->model_size = G.model[best_i].n;
			G.model[best_i] = (taw_buf){0};
		} else {
			out->profile = TAW_PROFILE_F32_STRICT_MIX;
			out->quant = cs[best_i].quant;
			out->model = F.model[best_i].p;
			out->model_size = F.model[best_i].n;
			F.model[best_i] = (taw_buf){0};
		}
	}
	for (size_t ci = 0; G.model && ci < ptries; ci++)
		taw_buf_free(&G.model[ci]);
	free(G.model); free(G.est); free(G.bpb); free(G.st);
	for (size_t ci = 0; F.model && ci < tries; ci++)
		taw_buf_free(&F.model[ci]);
	free(F.model);
	free(F.est);
	free(F.bpb);
	free(F.st);
	free(sm);
	return st;
}
