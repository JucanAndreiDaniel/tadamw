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
#include "p10.h"
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

static void adam_rows(adamw *o, float *w, float *m, float *v,
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
				adam_rows(&o, t->w + off, t->m + off, t->v + off,
				    t->g + off, t->row1, scale);
				t->touched1[t->list1[k]] = 0;
			}
			for (size_t k = 0; k < n2; k++) {
				size_t off = t->n1 + t->list2[k] * t->row2;
				adam_rows(&o, t->w + off, t->m + off, t->v + off,
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

/* ---- profile 10: chained predictors ----------------------------------------- */

typedef struct {
	p10_shape s;
	size_t nset[P10_UNITS], row[P10_UNITS], off[P10_UNITS];
	size_t np;
	float *w, *m, *v, *g; /* AdamW state, all np long */
	uint8_t *touched[P10_UNITS];
	uint32_t *list[P10_UNITS];
	bool tied; /* warm-up: every set is set 0 */
} cnet;

static void cnet_free(cnet *t)
{
	free(t->w);
	free(t->m);
	free(t->v);
	free(t->g);
	for (unsigned u = 0; u < P10_UNITS; u++) {
		free(t->touched[u]);
		free(t->list[u]);
	}
	memset(t, 0, sizeof *t);
}

static taw_status cnet_init(cnet *t, const p10_shape *s, uint64_t seed)
{
	memset(t, 0, sizeof *t);
	t->s = *s;
	for (unsigned u = 0; u < P10_UNITS; u++) {
		t->nset[u] = p1_sel_size[s->sel[u]];
		t->row[u] = p10_rowlen[u];
		t->off[u] = u ? t->off[u - 1] + t->nset[u - 1] * t->row[u - 1] : 0;
	}
	t->np = t->off[P10_UNITS - 1] +
	    t->nset[P10_UNITS - 1] * t->row[P10_UNITS - 1];
	t->w = calloc(t->np, sizeof(float));
	t->m = calloc(t->np, sizeof(float));
	t->v = calloc(t->np, sizeof(float));
	t->g = calloc(t->np, sizeof(float));
	for (unsigned u = 0; u < P10_UNITS; u++) {
		t->touched[u] = calloc(t->nset[u], 1);
		t->list[u] = malloc(t->nset[u] * sizeof(uint32_t));
	}
	if (!t->w || !t->m || !t->v || !t->g) {
		cnet_free(t);
		return TAW_E_NOMEM;
	}
	for (unsigned u = 0; u < P10_UNITS; u++)
		if (!t->touched[u] || !t->list[u]) {
			cnet_free(t);
			return TAW_E_NOMEM;
		}
	/* All sets start from the same small random weights; bias and
	 * inheritance weights start at zero so the chain begins as three
	 * nearly independent low-order predictors. */
	uint64_t rs = seed;
	for (unsigned u = 0; u < P10_UNITS; u++) {
		const unsigned m = p10_nin[u];
		for (unsigned j = 0; j < t->row[u]; j++) {
			float base = 0.0f;
			if (j != m - 1 && j != m) /* not bias, not inheritance */
				base = (j == 0 ? 0.5f : 0.3f) + 0.3f * rng_unif(&rs);
			for (size_t a = 0; a < t->nset[u]; a++)
				t->w[t->off[u] + a * t->row[u] + j] = base;
		}
	}
	return TAW_OK;
}

/* Forward + backward for one sample, accumulating into t->g. */
static void cnet_step(cnet *t, const sample *smp)
{
	float x[P1_XPAD];
	sample_x(smp, x);
	const p1_sel_ctx sc = sample_sc(smp);
	const float lim = 2047.0f / 256.0f;
	float v[P10_UNITS][P10_MAXROW], z[P10_UNITS], prev = 0.0f;
	bool act[P10_UNITS];
	for (unsigned u = 0; u < P10_UNITS; u++) {
		const uint32_t a = t->tied ? 0 : p1_sel_value(t->s.sel[u], &sc);
		const float *w = t->w + t->off[u] + (size_t)a * t->row[u];
		const unsigned m = p10_nin[u];
		for (unsigned i = 0; i < m; i++)
			v[u][i] = x[p10_subset[u][i]];
		if (u)
			v[u][m] = prev;
		float acc = 0.0f;
		for (unsigned j = 0; j < t->row[u]; j++)
			acc += w[j] * v[u][j];
		z[u] = acc;
		act[u] = acc > -lim && acc < lim;
		z[u] = acc;
		act[u] = acc > -lim && acc < lim;
		prev = acc < -lim ? -lim : acc > lim ? lim : acc;
	}
	const float p = (float)p1_squash((int32_t)(prev * 256.0f)) *
	    (1.0f / 65536.0f);
	float carry = p - (float)smp->bit; /* dL/dc[last] */
	for (unsigned k = P10_UNITS; k-- > 0;) {
		const unsigned u = k;
		const uint32_t a = t->tied ? 0 : p1_sel_value(t->s.sel[u], &sc);
		const size_t base = t->off[u] + (size_t)a * t->row[u];
		const float gu = carry * (act[u] ? 1.0f : 0.0f);
		float *g = t->g + base;
		for (unsigned j = 0; j < t->row[u]; j++)
			g[j] += gu * v[u][j];
		/* The clamped estimate of unit u-1 enters unit u through its
		 * last weight; propagate the gradient through it. */
		carry = u ? gu * t->w[base + p10_nin[u]] : 0.0f;
	}
}

static void cnet_untie(cnet *t)
{
	float *arr[3] = {t->w, t->m, t->v};
	for (int a = 0; a < 3; a++) {
		float *b = arr[a];
		for (unsigned u = 0; u < P10_UNITS; u++)
			for (size_t set = 1; set < t->nset[u]; set++)
				memcpy(b + t->off[u] + set * t->row[u],
				    b + t->off[u], t->row[u] * sizeof(float));
	}
	t->tied = false;
}

static void cnet_train(cnet *t, const sample *sm, size_t ns, int epochs,
    uint64_t seed)
{
	enum { BATCH = 256 };
	adamw o = {.b1 = 0.9f, .b2 = 0.999f, .eps = 1e-6f, .wd = 1e-5f,
		.b1t = 1.0f, .b2t = 1.0f};
	const float lr0 = 0.02f, lr1 = 0.001f;
	const uint64_t steps_total = (uint64_t)epochs *
	    ((ns + BATCH - 1) / BATCH);
	uint64_t step = 0, rs = seed;
	const int warm = epochs / 5 > 1 ? epochs / 5 : 1;
	t->tied = false;
	for (unsigned u = 0; u < P10_UNITS; u++)
		if (t->nset[u] > 1)
			t->tied = true;
	for (int ep = 0; ep < epochs; ep++) {
		if (t->tied && ep >= warm)
			cnet_untie(t);
		uint64_t stride = (uint64_t)((double)ns * 0.6180339887) | 1;
		while (ns > 1) {
			uint64_t a = stride, b = ns;
			while (b) {
				uint64_t r = a % b;
				a = b;
				b = r;
			}
			if (a == 1)
				break;
			stride += 2;
		}
		uint64_t pos = rng_next(&rs) % (ns ? ns : 1);
		for (size_t done = 0; done < ns;) {
			size_t nb = ns - done < BATCH ? ns - done : BATCH;
			size_t n[P10_UNITS] = {0, 0, 0};
			for (size_t k = 0; k < nb; k++) {
				const sample *s = &sm[pos];
				pos += stride;
				if (pos >= ns)
					pos %= ns;
				const p1_sel_ctx sc = sample_sc(s);
				for (unsigned u = 0; u < P10_UNITS; u++) {
					uint32_t a = t->tied ? 0 :
					    p1_sel_value(t->s.sel[u], &sc);
					if (!t->touched[u][a]) {
						t->touched[u][a] = 1;
						t->list[u][n[u]++] = a;
					}
				}
				cnet_step(t, s);
			}
			done += nb;
			step++;
			float frac = (float)step / (float)steps_total;
			o.lr = lr0 + (lr1 - lr0) * frac;
			o.b1t *= o.b1;
			o.b2t *= o.b2;
			const float scale = 1.0f / (float)nb;
			for (unsigned u = 0; u < P10_UNITS; u++)
				for (size_t k = 0; k < n[u]; k++) {
					size_t off = t->off[u] +
					    (size_t)t->list[u][k] * t->row[u];
					adam_rows(&o, t->w + off, t->m + off,
					    t->v + off, t->g + off, t->row[u],
					    scale);
					t->touched[u][t->list[u][k]] = 0;
				}
		}
	}
}

static taw_status cserialize(const cnet *t, uint8_t quant, taw_buf *b)
{
	uint8_t head[P10_MODEL_HEAD];
	p10_head_write(head, &t->s);
	taw_buf_put(b, head, sizeof head);
	for (unsigned u = 0; u < P10_UNITS; u++)
		for (size_t a = 0; a < t->nset[u]; a++)
			put_row(b, quant, t->w + t->off[u] + a * t->row[u],
			    t->row[u]);
	return b->oom ? TAW_E_NOMEM : TAW_OK;
}

static taw_status cevaluate(const uint8_t *model, size_t size, uint8_t quant,
    const sample *sm, size_t ns, double *bits)
{
	const float *ct = cost_tab();
	if (!ct)
		return TAW_E_NOMEM;
	p10_net net;
	taw_status st = p10_net_load(&net, quant, model, size);
	if (st != TAW_OK)
		return st;
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float x[P1_XPAD];
		sample_x(&sm[k], x);
		const p1_sel_ctx sc = sample_sc(&sm[k]);
		uint32_t p = p1_squash(p10_net_eval(&net, x, &sc));
		sum += ct[sm[k].bit ? p : 65536 - p];
	}
	p10_net_free(&net);
	*bits = ns ? sum / (double)ns : 8.0;
	return TAW_OK;
}

typedef struct {
	p10_shape s;
	uint8_t quant;
	uint64_t size;
} ccand;

static const uint8_t csel0[] = {P1_SEL_NONE, P1_SEL_BITPOS};
static const uint8_t csel1[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0};
static const uint8_t csel2[] = {P1_SEL_NONE, P1_SEL_BITPOS, P1_SEL_C0};

/* Retuned after upstream's bit-position sampler and stronger profile 1
 * prior. Measured on the bench corpus: a small chain with a bit-position
 * selector on its final unit is consistently better than a selector-free
 * one, while C0-selector tables cost more than they save. The prior
 * therefore favours small rows with sel2 = BITPOS. Final selection still
 * uses the measured cost including model bytes. */
static double cprior(const ccand *c, size_t n)
{
	double target = (double)n / 20000.0 + 16.0;
	if (target > 131072.0)
		target = 131072.0;
	double r = dlog2((double)c->size / target);
	double pr = -(r < 0 ? -r : r);
	if (c->s.sel[0] == P1_SEL_BITPOS)
		pr += 0.8;
	if (c->s.sel[2] == P1_SEL_BITPOS)
		pr += 2.0;
	if (c->quant == TAW_QUANT_Q8)
		pr += 0.3;
	return pr;
}

static int cmp_cprior(const void *a, const void *b)
{
	double x = ((const double *)a)[0], y = ((const double *)b)[0];
	return (x < y) - (x > y);
}

static size_t cenumerate(const p1_shape *geo, uint64_t budget, int quant_pref,
    size_t n, ccand *out, size_t cap)
{
	size_t k = 0;
	static const uint8_t quants[] = {TAW_QUANT_F16, TAW_QUANT_Q8};
	/* Experiment knob: TAW_CHAIN_FORCE="sel0,sel1,sel2,quant" fits one
	 * shape only. Not part of the archive format. */
	int f[4] = {-1, -1, -1, -1};
	const char *force = getenv("TAW_CHAIN_FORCE");
	if (force && sscanf(force, "%d,%d,%d,%d", &f[0], &f[1], &f[2], &f[3]) == 4)
		force = f[0] >= 0 && f[0] < P1_SEL_COUNT &&
			f[1] >= 0 && f[1] < P1_SEL_COUNT &&
			f[2] >= 0 && f[2] < P1_SEL_COUNT &&
			(f[3] == TAW_QUANT_F16 || f[3] == TAW_QUANT_Q8)
			? force : nullptr;
	else
		force = nullptr;
	for (size_t qi = 0; qi < 2; qi++) {
		uint8_t q = quants[qi];
		if (quant_pref == TAW_WEIGHTS_F16 && q != TAW_QUANT_F16)
			continue;
		if (quant_pref == TAW_WEIGHTS_Q8 && q != TAW_QUANT_Q8)
			continue;
		if (force && q != f[3])
			continue;
		for (size_t a = 0; a < sizeof csel0; a++) {
			if (force && csel0[a] != f[0])
				continue;
			for (size_t b = 0; b < sizeof csel1; b++) {
				if (force && csel1[b] != f[1])
					continue;
				for (size_t d = 0; d < sizeof csel2; d++) {
					if (force && csel2[d] != f[2])
						continue;
					ccand c = {.quant = q, .s = {
						.lines_log2 = geo->lines_log2,
						.hist_log2 = geo->hist_log2,
						.match_log2 = geo->match_log2,
						.sel = {csel0[a], csel1[b], csel2[d]},
					}};
					c.size = P10_MODEL_HEAD +
					    p10_weights_size(&c.s, q);
					/* Chain rows are tiny; large selector
					 * tables never repaid their bytes on the
					 * bench, and the biased sample estimator
					 * preferred them at high effort. */
					const uint64_t cap_bytes = 512;
					if (c.size <= budget && c.size <= cap_bytes &&
					    k < cap)
						out[k++] = c;
				}
			}
		}
	}
	struct {
		double pr;
		ccand c;
	} *tmp = malloc(k * sizeof *tmp);
	if (!tmp)
		return k;
	for (size_t i = 0; i < k; i++) {
		tmp[i].pr = cprior(&out[i], n);
		tmp[i].c = out[i];
	}
	qsort(tmp, k, sizeof *tmp, cmp_cprior);
	for (size_t i = 0; i < k; i++)
		out[i] = tmp[i].c;
	free(tmp);
	return k;
}

typedef struct {
	const ccand *cs;
	const sample *sm;
	size_t ns;
	int epochs;
	taw_buf *model;
	double *bpb;
	taw_status *st;
} cfit_job;

static void cfit_one(void *arg, size_t ci, unsigned w)
{
	(void)w;
	cfit_job *F = arg;
	cnet t;
	taw_status st = cnet_init(&t, &F->cs[ci].s, 0x7A57A11ull + ci);
	if (st == TAW_OK) {
		cnet_train(&t, F->sm, F->ns, F->epochs, 0xC0FFEEull + ci);
		st = cserialize(&t, F->cs[ci].quant, &F->model[ci]);
		cnet_free(&t);
	}
	if (st == TAW_OK)
		st = cevaluate(F->model[ci].p, F->model[ci].n,
		    F->cs[ci].quant, F->sm, F->ns, &F->bpb[ci]);
	F->st[ci] = st;
}

static taw_status chain_fit_source(const taw_source *in, uint64_t unit,
    const taw_params *p, taw_model_choice *out)
{
	const uint64_t n = in->size;
	if (!n)
		return TAW_E_ARG;
	const uint64_t budget = n / 32;
	if (unit == 0 || unit > n)
		unit = n;
	p1_shape geo1 = {0};
	geometry((size_t)unit, &geo1);
	ccand cs[64];
	size_t ncand = cenumerate(&geo1, budget, p->weights, (size_t)n, cs, 64);
	if (!ncand)
		return TAW_E_ARG;
	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	/* The tiny chain has few weights but needs more fitting steps than
	 * the level's default to converge; fitting is encoder-only. Extra
	 * knobs exist for experiments (TAW_CHAIN_EPOCHS / ..._SAMPLES). */
	int epochs = effort[lv].epochs > 12 ? effort[lv].epochs : 12;
	const char *ep = getenv("TAW_CHAIN_EPOCHS");
	if (ep) {
		int v = atoi(ep);
		if (v > 0 && v <= 100)
			epochs = v;
	}
	uint64_t want = effort[lv].samples;
	if (want < (1u << 19))
		want = 1u << 19;
	const char *sp = getenv("TAW_CHAIN_SAMPLES");
	if (sp) {
		unsigned v = (unsigned)atoi(sp);
		if (v > 0 && v <= 20)
			want = 1ull << v;
	}
	sample *sm;
	size_t ns;
	taw_status st = collect(in, unit, &geo1, want, threads, &sm, &ns);
	if (st != TAW_OK)
		return st;
	const char *dbg = getenv("TAW_FIT_DEBUG");
	size_t tries = effort[lv].cands;
	if (dbg && strchr(dbg, 'a'))
		tries = ncand;
	if (tries > ncand)
		tries = ncand;
	cfit_job F = {.cs = cs, .sm = sm, .ns = ns, .epochs = epochs};
	F.model = calloc(tries, sizeof *F.model);
	F.bpb = calloc(tries, sizeof *F.bpb);
	F.st = calloc(tries, sizeof *F.st);
	if (!F.model || !F.bpb || !F.st)
		st = TAW_E_NOMEM;
	else
		st = taw_parallel(threads, tries, 0, cfit_one, nullptr, &F);
	size_t best = SIZE_MAX;
	double best_est = 0.0;
	for (size_t ci = 0; st == TAW_OK && ci < tries; ci++) {
		if ((st = F.st[ci]) != TAW_OK)
			break;
		/* The shared sampler keeps an even spread of bit indices, so
		 * its bpc is only a proxy (at some sample counts it is almost
		 * all MSBs). Weight model bytes 4x: the chain's small rows
		 * fit reliably and larger selector tables did not repay
		 * their cost in measurements. */
		double est = F.bpb[ci] * 8.0 * (double)n +
		    32.0 * (double)F.model[ci].n;
		if (dbg)
			fprintf(stderr,
			    "chain fit: s=%u,%u,%u q=%u model=%zu "
			    "bpc=%.4f score=%.0f\n",
			    cs[ci].s.sel[0], cs[ci].s.sel[1], cs[ci].s.sel[2],
			    cs[ci].quant, F.model[ci].n, F.bpb[ci] * 8.0,
			    est / 8.0);
		if (best == SIZE_MAX || est < best_est) {
			best = ci;
			best_est = est;
		}
	}
	if (st == TAW_OK) {
		out->profile = TAW_PROFILE_F32_CHAINED_MIX;
		out->quant = cs[best].quant;
		out->model = F.model[best].p;
		out->model_size = F.model[best].n;
		F.model[best] = (taw_buf){0};
	}
	for (size_t ci = 0; F.model && ci < tries; ci++)
		taw_buf_free(&F.model[ci]);
	free(F.model);
	free(F.bpb);
	free(F.st);
	free(sm);
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
	if (p->predictor == TAW_PREDICTOR_CHAIN)
		return chain_fit_source(in, unit, p, out);
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
	free(sm);
	return st;
}
