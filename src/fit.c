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
#include "p12.h"
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
	unsigned phase; /* 0: bits at stride want; 1: rotate through bitpos */
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

/* ---- profile 12: MDL-pruned input-specific context tree ------------------
 * Adaptation of the tri-plane context-tree idea (Bai et al.,
 * arXiv:2608.13897) to the bit stream: all leaves share a frozen hidden
 * layer, each leaf has its own small output row, and a binary tree over
 * cheap decoder-visible features selects one row per bit. Splits are kept
 * only when the measured coding gain pays for the node and leaf bytes
 * (minimum description length). Decoding is integer except for the two
 * dot products, which follow profile 1's float conventions exactly. */

typedef struct {
	float w[P1_XPAD]; /* 0..H-1: hidden units; H: bias */
} lmod;

typedef struct {
	float *x;         /* [nfit][d] hidden activations + 1.0 */
	int16_t *st;      /* [nfit][P12_NF] stretched context inputs */
	p1_sel_ctx *sc;   /* [nfit] */
	uint8_t *bit;     /* [nfit] */
	uint8_t *fv;      /* [nfit][P12_NFEAT] split features */
	float *prob, *err; /* [nfit] scratch */
	uint32_t *idx, *tmp; /* [nfit] work list and partition scratch */
	size_t nfit;
	unsigned d;       /* H + 1 */
	uint8_t quant;
	p12_tree *t;
	float models[P12_MAX_LEAVES][P1_XPAD]; /* unquantized leaf rows */
	float init[8][P1_XPAD]; /* profile-1-style per-bitpos output rows */
	unsigned next_node, next_leaf;
	unsigned leaf_max, depth_max, epochs;
	float lr;
	double penalty;
} treejob;

/* Quantize a leaf row exactly as put_row() does, then dequantize it exactly
 * as the decoder does. Fitting therefore sees the bits the coder will see. */
static void row_deq(const float *w, uint8_t quant, unsigned len, float *deq)
{
	uint8_t row[256];
	const size_t n = (size_t)taw_quant_row_bytes(quant, len);
	(void)n;
	taw_quant_row_encode(quant, w, len, row);
	(void)taw_quant_row_decode(quant, row, len, deq);
}

/* The hidden layer, computed with the same operations as p12_hidden(). */
static void hidden_calc(const p1_net *n, unsigned sel1, const float *x,
    unsigned h, const p1_sel_ctx *sc, float *out)
{
	const float *w1 = n->w1 +
	    (size_t)p1_sel_value(sel1, sc) * P1_F * n->hpad;
	for (unsigned j = 0; j < h; j++) {
		float s[4] = {0, 0, 0, 0};
		for (unsigned i = 0; i < P1_F; i++)
			s[i & 3] += w1[j + i * n->hpad] * x[i];
		float a = (s[0] + s[1]) + (s[2] + s[3]);
		out[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
	}
}

static void lm_train(treejob *F, lmod *m, const uint32_t *idx, size_t n,
    int epochs, float lr)
{
	const unsigned d = F->d;
	float mm[P1_XPAD] = {0}, vv[P1_XPAD] = {0};
	float b1 = 0.9f, b2 = 0.999f, b1t = 1.0f, b2t = 1.0f;
	for (int ep = 0; ep < epochs; ep++) {
		float step = lr * (1.0f - 0.8f * (float)ep /
		    (float)(epochs > 1 ? epochs - 1 : 1));
		for (size_t k = 0; k < n; k++) {
			const uint32_t j = idx[k];
			const float *xx = F->x + (size_t)j * d;
			float t = m->w[d - 1];
			for (unsigned i = 0; i + 1 < d; i++)
				t += m->w[i] * xx[i];
			float p = (float)p1_squash((int32_t)t) * (1.0f / 65536.0f);
			float e = p - (float)F->bit[j];
			b1t *= b1;
			b2t *= b2;
			float c1 = 1.0f / (1.0f - b1t), c2 = 1.0f / (1.0f - b2t);
			for (unsigned q = 0; q < d; q++) {
				float g = e * xx[q];
				mm[q] = b1 * mm[q] + (1.0f - b1) * g;
				vv[q] = b2 * vv[q] + (1.0f - b2) * g * g;
				m->w[q] -= step * (mm[q] * c1) /
				    (fsqrt(vv[q] * c2) + 1e-8f);
			}
		}
	}
}

static double leaf_bits(const treejob *F, const lmod *m,
    const uint32_t *idx, size_t n)
{
	const float *ct = cost_tab();
	if (!ct)
		return 8.0 * (double)n;
	float deq[P1_XPAD];
	row_deq(m->w, F->quant, F->d, deq);
	double sum = 0.0;
	for (size_t k = 0; k < n; k++) {
		const uint32_t j = idx[k];
		uint32_t p = p1_squash(p12_leaf_t(deq,
		    F->x + (size_t)j * F->d, F->d - 1));
		sum += ct[F->bit[j] ? p : 65536u - p];
	}
	return sum;
}

static void prep(treejob *F, const uint32_t *idx, size_t n, const lmod *m)
{
	float deq[P1_XPAD];
	row_deq(m->w, F->quant, F->d, deq);
	for (size_t k = 0; k < n; k++) {
		const uint32_t j = idx[k];
		int32_t z = p12_leaf_t(deq, F->x + (size_t)j * F->d, F->d - 1);
		float p = (float)p1_squash(z) * (1.0f / 65536.0f);
		F->prob[j] = p;
		F->err[j] = p - (float)F->bit[j];
	}
}

/* Cross-entropy keeping the direction of a trained row but choosing its
 * overall magnitude on the samples. Long Adam runs saturate the logit; a
 * calibrated scale recovers the probability quality of a short fit. */
static void lm_calibrate(const treejob *F, lmod *m, const uint32_t *idx,
    size_t n)
{
	lmod best = *m;
	double bb = leaf_bits(F, m, idx, n);
	for (float sc = 0.5f; sc >= 0.002f; sc *= 0.5f) {
		lmod t = *m;
		for (unsigned q = 0; q < F->d; q++)
			t.w[q] *= sc;
		double b = leaf_bits(F, &t, idx, n);
		if (b < bb) {
			bb = b;
			best = t;
		}
	}
	*m = best;
}

typedef struct { unsigned f; uint8_t thr; double gain; } split_best;

/* One Newton step with a diagonal Hessian estimates the cross-entropy drop
 * of the best child models. This only ranks splits; the keep decision uses
 * the exact MDL cost after fitting and quantization. */
static split_best best_split(const treejob *F, const uint32_t *idx, size_t n)
{
	const unsigned d = F->d;
	split_best best = {.f = 0, .thr = 0, .gain = 0.0};
	const char *envf = getenv("TAW_P12_FEATURE");
	unsigned fmax = envf ? (unsigned)atoi(envf) + 1u : P12_NFEAT;
	unsigned fmin = envf ? (unsigned)atoi(envf) : 0u;
	for (unsigned f = fmin; f < fmax; f++) {
		uint8_t present[256] = {0};
		for (size_t k = 0; k < n; k++)
			present[F->fv[(size_t)idx[k] * P12_NFEAT + f]] = 1;
		uint8_t vals[256];
		unsigned nv = 0;
		for (unsigned v = 0; v < 256; v++)
			if (present[v])
				vals[nv++] = (uint8_t)v;
		if (nv < 2)
			continue;
		unsigned ntry = nv - 1 < 4 ? nv - 1 : 4;
		for (unsigned a = 0; a < ntry; a++) {
			unsigned kth = 1 + (unsigned)((uint64_t)a * (nv - 1) / ntry);
			uint8_t thr = vals[kth];
			float gl[P1_XPAD] = {0}, hl[P1_XPAD] = {0};
			float gr[P1_XPAD] = {0}, hr[P1_XPAD] = {0};
			for (size_t k = 0; k < n; k++) {
				const uint32_t j = idx[k];
				const float *xx = F->x + (size_t)j * d;
				float e = F->err[j], p = F->prob[j];
				float h = p * (1.0f - p) + 1e-7f;
				bool left = F->fv[(size_t)j * P12_NFEAT + f] < thr;
				float *g = left ? gl : gr;
				float *hh = left ? hl : hr;
				for (unsigned q = 0; q < d; q++) {
					float xd = xx[q];
					g[q] += e * xd;
					hh[q] += h * xd * xd;
				}
			}
			double gain = 0.0;
			for (unsigned q = 0; q < d; q++)
				gain += 0.5 * ((double)gl[q] * gl[q] / hl[q] +
				    (double)gr[q] * gr[q] / hr[q]);
			gain *= 1.4426950408889634; /* nats -> bits */
			if (gain > best.gain)
				best = (split_best){.f = f, .thr = thr, .gain = gain};
		}
	}
	return best;
}

/* The first three levels are a fixed balanced bit-position tree. They
 * reproduce profile 1's per-bitpos output sets, so the MDL tree starts from
 * a proven model; only deeper splits are data-adaptive and MDL-pruned. */
static unsigned tree_grow(treejob *F, uint32_t *idx, size_t n, unsigned depth,
    unsigned bitprefix, const lmod *m)
{
	unsigned ni = F->next_node++;
	bool forced = depth < 3;
	if (getenv("TAW_FIT_DEBUG"))
		fprintf(stderr, " tree_grow ni=%u depth=%u forced=%d n=%zu fv0=%u,%u,%u,%u sc0=%u,%u,%u\n",
		    ni, depth, (int)forced, n, F->fv[0], F->fv[P12_NFEAT],
		    F->fv[2 * P12_NFEAT], F->fv[3 * P12_NFEAT],
		    F->sc[0].bitpos, F->sc[1].bitpos, F->sc[2].bitpos);
	unsigned thr = forced ? (bitprefix | (1u << (2 - depth))) : 0;
	split_best best = {.f = P12_F_BITPOS, .thr = (uint8_t)thr,
		.gain = 1e30};
	if (!forced) {
		double cost = 8.0 * (P12_NODE_BYTES +
		    ((F->quant == TAW_QUANT_Q8 ? 2u + F->d : 2u * F->d))) *
		    F->penalty;
		if (n >= 256 && depth < F->depth_max &&
		    F->next_leaf < F->leaf_max) {
			prep(F, idx, n, m);
			best = best_split(F, idx, n);
			if (getenv("TAW_FIT_DEBUG"))
				fprintf(stderr,
				    " grow d=%u n=%zu f=%u thr=%u gain=%.1f cost=%.1f\n",
				    depth, n, best.f, best.thr, best.gain, cost);
			best.gain -= cost;
		} else
			best.gain = 0.0;
	}
	if (best.gain > 0.0) {
		size_t nl = 0;
		for (size_t k = 0; k < n; k++)
			nl += F->fv[(size_t)idx[k] * P12_NFEAT + best.f] < best.thr;
		size_t nr = n - nl, l = 0, r = nl;
		for (size_t k = 0; k < n; k++) {
			uint32_t j = idx[k];
			if (F->fv[(size_t)j * P12_NFEAT + best.f] < best.thr)
				F->tmp[l++] = j;
			else
				F->tmp[r++] = j;
		}
		if (getenv("TAW_FIT_DEBUG"))
			fprintf(stderr, "  partition f=%u thr=%u nl=%zu nr=%zu\n",
			    best.f, best.thr, nl, nr);
		if (nl >= 64 && nr >= 64) {
			memcpy(idx, F->tmp, n * sizeof *idx);
			lmod ml, mr;
			if (forced) {
				memcpy(&ml, F->init[bitprefix], sizeof ml);
				memcpy(&mr, F->init[bitprefix | (1u << (2 - depth))],
				    sizeof mr);
			} else {
				ml = *m;
				mr = *m;
				lm_train(F, &ml, idx, nl, (int)F->epochs, F->lr);
				lm_train(F, &mr, idx + nl, nr, (int)F->epochs,
				    F->lr);
				lm_calibrate(F, &ml, idx, nl);
				lm_calibrate(F, &mr, idx + nl, nr);
			}
			F->t->node[ni] = (p12_node){.feat = (uint8_t)best.f,
				.thr = best.thr, .left = 0, .right = 0};
			unsigned lc = tree_grow(F, idx, nl, depth + 1, bitprefix,
			    &ml);
			unsigned rc = tree_grow(F, idx + nl, nr, depth + 1,
			    bitprefix | (1u << (2 - depth)), &mr);
			F->t->node[ni].left = (uint8_t)lc;
			F->t->node[ni].right = (uint8_t)rc;
			return ni;
		}
	}
	unsigned li = F->next_leaf++;
	memcpy(F->models[li], m->w, F->d * sizeof *m->w);
	F->t->node[ni] = (p12_node){.feat = 0xFF, .thr = 0,
		.left = (uint8_t)li, .right = 0};
	return ni;
}

static double tree_full_bits(const p12_tree *t, const p1_net *pnet,
    const sample *sm, size_t ns)
{
	const float *ct = cost_tab();
	if (!ct)
		return 8.0 * (double)ns;
	double sum = 0.0;
	for (size_t k = 0; k < ns; k++) {
		alignas(32) float xv[P1_XPAD];
		sample_x(&sm[k], xv);
		const p1_sel_ctx sc = sample_sc(&sm[k]);
		float h[P1_HMAX] = {0};
		hidden_calc(pnet, t->sel1, xv, t->h, &sc, h);
		int16_t st[P12_NF];
		for (unsigned i = 0; i < P12_NF; i++)
			st[i] = sm[k].st[i];
		uint32_t p = p1_squash(p12_eval_h(t, h, &sc, st));
		sum += ct[sm[k].bit ? p : 65536u - p];
	}
	return sum;
}

static taw_status fit_tree(const p1_shape *geo, const sample *sm, size_t ns,
    unsigned leaf_max, unsigned depth_max, double penalty, uint8_t quant,
    unsigned hidden, unsigned sel1, int mix_epochs, taw_buf *out,
    double *bpb)
{
	if (leaf_max > P12_MAX_LEAVES || hidden < 1 || hidden > P12_HMAX)
		return TAW_E_ARG;
	size_t nfit = ns < (1u << 17) ? ns : (1u << 17);
	if (!nfit)
		return TAW_E_ARG;
	const unsigned d = hidden + 1;
	treejob F = {0};
	F.nfit = nfit;
	F.d = d;
	F.quant = quant;
	F.leaf_max = leaf_max;
	F.depth_max = depth_max;
	F.penalty = penalty;
	F.epochs = 30;
	F.lr = 0.01f;
	if (getenv("TAW_P12_EPOCHS"))
		F.epochs = (unsigned)atoi(getenv("TAW_P12_EPOCHS"));
	if (getenv("TAW_P12_LR"))
		F.lr = (float)atof(getenv("TAW_P12_LR"));

	F.x = malloc(nfit * P1_XPAD * sizeof *F.x);
	F.st = malloc(nfit * P12_NF * sizeof *F.st);
	F.sc = malloc(nfit * sizeof *F.sc);
	F.bit = malloc(nfit);
	F.fv = malloc(nfit * P12_NFEAT);
	F.prob = malloc(nfit * sizeof *F.prob);
	F.err = malloc(nfit * sizeof *F.err);
	F.idx = malloc(nfit * sizeof *F.idx);
	F.tmp = malloc(nfit * sizeof *F.tmp);
	F.t = calloc(1, sizeof *F.t);
	taw_buf w1b = {0}, leafb = {0}, pm = {0};
	p1_net pnet = {0};
	taw_status st = TAW_E_NOMEM;
	if (!F.x || !F.st || !F.sc || !F.bit || !F.fv || !F.prob || !F.err ||
	    !F.idx || !F.tmp || !F.t)
		goto done;

	for (size_t i = 0; i < nfit; i++) {
		size_t j = (size_t)((uint64_t)i * ns / nfit);
		const sample *s = &sm[j];
		F.idx[i] = (uint32_t)i;
		F.bit[i] = s->bit;
		F.sc[i] = sample_sc(s);
		if (getenv("TAW_FIT_DEBUG") && i < 4)
			fprintf(stderr, " sample i=%zu j=%zu c0=%u c1=%u bp=%u\n",
			    i, j, s->c0, s->c1, F.sc[i].bitpos);
		memcpy(F.st + i * P12_NF, s->st, P12_NF * sizeof *F.st);
		for (unsigned f = 0; f < P12_NFEAT; f++)
			F.fv[i * P12_NFEAT + f] =
			    (uint8_t)p12_feature(f, &F.sc[i], s->st);
	}

	/* Train the shared hidden layer with the existing mixer trainer. */
	unsigned base_sel = P1_SEL_BITPOS;
	if (getenv("TAW_P12_BASESEL"))
		base_sel = (unsigned)atoi(getenv("TAW_P12_BASESEL"));
	p1_shape sh = *geo;
	sh.sel1 = (uint8_t)sel1;
	sh.sel2 = (uint8_t)base_sel;
	sh.hidden = (uint8_t)hidden;
	tnet net;
	st = tnet_init(&net, &sh, 0x7A57A11ull + 12);
	if (st != TAW_OK)
		goto done;
	tnet_train(&net, sm, ns, mix_epochs, 0xC0FFEEull + 12);
	/* The eight bit-position output rows become the tree's base experts. */
	for (unsigned c = 0; c < 8; c++) {
		for (unsigned q = 0; q < d; q++)
			F.init[c][q] = 0.0f;
		if (base_sel == P1_SEL_BITPOS) {
			const float *row = net.w + net.n1 + (size_t)c * net.row2;
			for (unsigned q = 0; q < d; q++)
				F.init[c][q] = row[q];
		} else {
			for (size_t s = 0; s < net.s2; s++) {
				const float *row = net.w + net.n1 + s * net.row2;
				for (unsigned q = 0; q < d; q++)
					F.init[c][q] += row[q] / (float)net.s2;
			}
		}
	}
	if (getenv("TAW_FIT_DEBUG"))
		fprintf(stderr, " tnet w1[0..5]=%f %f %f %f %f %f row1=%zu\n",
		    net.w[0], net.w[1], net.w[2], net.w[3], net.w[4], net.w[5],
		    net.row1);
	for (size_t set = 0; set < net.s1; set++)
		for (unsigned j = 0; j < hidden; j++)
			put_row(&w1b, quant, net.w + set * net.row1 + j * P1_F,
			    P1_F);
	tnet_free(&net);
	if (w1b.oom)
		goto done;

	/* Load that first layer so hidden activations use exact decoder math. */
	uint8_t head[P1_MODEL_HEAD];
	p1_shape shtmp = sh;
	shtmp.sel2 = P1_SEL_NONE; /* only one dummy output row is appended */
	p1_head_write(head, &shtmp);
	taw_buf_put(&pm, head, sizeof head);
	taw_buf_put(&pm, w1b.p, w1b.n);
	float zero[P1_XPAD] = {0};
	put_row(&pm, quant, zero, d);
	st = p1_net_load(&pnet, quant, pm.p, pm.n);
	if (st != TAW_OK)
		goto done;

	for (size_t i = 0; i < nfit; i++) {
		alignas(32) float xv[P1_XPAD] = {0};
		for (unsigned q = 0; q < P12_NF; q++)
			xv[q] = (float)F.st[i * P12_NF + q] * (1.0f / 256.0f);
		xv[P1_IN_BIAS] = 1.0f;
		float h[P1_HMAX] = {0};
		hidden_calc(&pnet, sel1, xv, hidden, &F.sc[i], h);
		for (unsigned q = 0; q < hidden; q++)
			F.x[i * d + q] = h[q];
		F.x[i * d + hidden] = 1.0f;
	}

	if (getenv("TAW_FIT_DEBUG"))
		fprintf(stderr, " hidden[0..7]=%f %f %f %f %f %f %f %f bit=%u\n",
		    F.x[0], F.x[1], F.x[2], F.x[3], F.x[4], F.x[5], F.x[6],
		    F.x[7], F.bit[0]);
	F.t->geo = *geo;
	F.t->h = hidden;
	F.t->sel1 = sel1;
	lmod root;
	memcpy(&root, F.init[0], sizeof root);
	tree_grow(&F, F.idx, nfit, 0, 0, &root);
	F.t->nnodes = F.next_node;
	F.t->nleaves = F.next_leaf;
	if (getenv("TAW_FIT_DEBUG"))
		fprintf(stderr, " built tree nodes=%u leaves=%u\n", F.next_node,
		    F.next_leaf);
	for (unsigned i = 0; i < F.t->nleaves; i++)
		row_deq(F.models[i], quant, d, F.t->w2[i]);
	for (unsigned i = 0; i < F.t->nleaves; i++)
		put_row(&leafb, quant, F.models[i], d);
	if (leafb.oom)
		goto done;
	p12_serialize(F.t, w1b.p, w1b.n, leafb.p, leafb.n, out);
	*bpb = tree_full_bits(F.t, &pnet, sm, ns) / (double)ns;
	st = out->oom ? TAW_E_NOMEM : TAW_OK;
done:
	free(pm.p);
	free(w1b.p);
	free(leafb.p);
	p1_net_free(&pnet);
	free(F.t);
	free(F.x);
	free(F.st);
	free(F.sc);
	free(F.bit);
	free(F.fv);
	free(F.prob);
	free(F.err);
	free(F.idx);
	free(F.tmp);
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
	const bool force_mix = p->predictor == TAW_PREDICTOR_MIX;
	const bool force_tree = p->predictor == TAW_PREDICTOR_MDL;
	const uint64_t budget = n / 32;
	if (unit == 0 || unit > n)
		unit = n;

	p1_shape geo = {0};
	geometry((size_t)unit, &geo);
	cand cs[1024];
	size_t ncand = 0;
	if (!force_tree) {
		ncand = enumerate(&geo, budget, p->weights, (size_t)n, cs, 1024);
		if (ncand == 0)
			return force_mix ? TAW_E_ARG : TAW_OK;
	}

	const int lv = p->level;
	const unsigned threads = p->threads ? p->threads : 1;
	sample *sm;
	size_t ns;
	taw_status st = collect(in, unit, &geo, effort[lv].samples, threads,
	    &sm, &ns);
	if (st != TAW_OK)
		return st;

	const char *dbg = getenv("TAW_FIT_DEBUG");
	taw_model_choice best = {0};
	double best_est = 1e300;
	taw_status st_mix = TAW_OK;

	if (!force_tree) {
		size_t tries = effort[lv].cands;
		if (dbg && strchr(dbg, 'a'))
			tries = ncand;
		if (tries > ncand)
			tries = ncand;

		fit_job F = {.cs = cs, .sm = sm, .ns = ns,
			.epochs = effort[lv].epochs, .n = n};
		F.model = calloc(tries, sizeof *F.model);
		F.est = calloc(tries, sizeof *F.est);
		F.bpb = calloc(tries, sizeof *F.bpb);
		F.st = calloc(tries, sizeof *F.st);
		if (!F.model || !F.est || !F.bpb || !F.st)
			st_mix = TAW_E_NOMEM;
		else
			st_mix = fit_all(&F, tries, threads, p->gpu,
			    dbg != nullptr);
		size_t bi = SIZE_MAX;
		for (size_t ci = 0; st_mix == TAW_OK && ci < tries; ci++) {
			if ((st_mix = F.st[ci]) != TAW_OK)
				break;
			if (dbg)
				fprintf(stderr,
				    "taw fit: H=%u sel1=%u sel2=%u q=%u "
				    "model=%zu bpc=%.4f est=%.0f B\n",
				    cs[ci].s.hidden, cs[ci].s.sel1,
				    cs[ci].s.sel2, cs[ci].quant,
				    F.model[ci].n, F.bpb[ci] * 8.0,
				    F.est[ci] / 8.0);
			if (bi == SIZE_MAX || F.est[ci] < F.est[bi])
				bi = ci;
		}
		if (st_mix == TAW_OK) {
			best = (taw_model_choice){
				.profile = TAW_PROFILE_F32_STRICT_MIX,
				.quant = cs[bi].quant,
				.model = F.model[bi].p,
				.model_size = F.model[bi].n,
			};
			best_est = F.est[bi];
			F.model[bi] = (taw_buf){0};
		}
		for (size_t ci = 0; F.model && ci < tries; ci++)
			taw_buf_free(&F.model[ci]);
		free(F.model);
		free(F.est);
		free(F.bpb);
		free(F.st);
	}

	/* A forced mixer must not be displaced by the tree, and tiny inputs
	 * may not have room for a tree model at all. */
	if (force_mix) {
		free(sm);
		if (st_mix != TAW_OK)
			return st_mix;
		if (best.model)
			*out = best;
		return TAW_OK;
	}

	/* Fit the MDL tree and compare its MDL to the best mixer. */
	unsigned leaf_max = lv <= 3 ? 16 : lv <= 6 ? 24 : 32;
	unsigned depth_max = 6;
	double penalty = 1.0;
	if (getenv("TAW_P12_LEAVES"))
		leaf_max = (unsigned)atoi(getenv("TAW_P12_LEAVES"));
	if (getenv("TAW_P12_PENALTY"))
		penalty = atof(getenv("TAW_P12_PENALTY"));
	uint8_t tquant = p->weights == TAW_WEIGHTS_F16 ? TAW_QUANT_F16
	    : TAW_QUANT_Q8;
	unsigned hidden = 12, sel1 = P1_SEL_NONE;
	int mix_epochs = effort[lv].epochs;
	if (getenv("TAW_P12_H"))
		hidden = (unsigned)atoi(getenv("TAW_P12_H"));
	if (getenv("TAW_P12_SEL1"))
		sel1 = (unsigned)atoi(getenv("TAW_P12_SEL1"));
	if (getenv("TAW_P12_MIXEPOCHS"))
		mix_epochs = atoi(getenv("TAW_P12_MIXEPOCHS"));
	/* The format caps the model at raw/32 bytes; shrink the shared
	 * hidden layer until the smallest possible tree fits. */
	const unsigned rw1 = tquant == TAW_QUANT_Q8 ? 28u : 52u;
	while (hidden > 2 && P12_MODEL_HEAD +
	    (uint64_t)p1_sel_size[sel1] * hidden * rw1 + P12_NODE_BYTES +
	    (tquant == TAW_QUANT_Q8 ? 2u + hidden : 2u * (hidden + 1)) > budget)
		hidden >>= 1;
	if (P12_MODEL_HEAD + (uint64_t)p1_sel_size[sel1] * hidden * rw1 +
	    P12_NODE_BYTES + (tquant == TAW_QUANT_Q8 ? 2u + hidden
	    : 2u * (hidden + 1)) > budget && sel1 != P1_SEL_NONE)
		sel1 = P1_SEL_NONE;
	taw_buf tb = {0};
	double tree_bpb = 8.0;
	st = fit_tree(&geo, sm, ns, leaf_max, depth_max, penalty, tquant,
	    hidden, sel1, mix_epochs, &tb, &tree_bpb);
	if (st == TAW_OK) {
		double tree_est = tree_bpb * 8.0 * (double)n + 8.0 * (double)tb.n;
		if (dbg) {
			fprintf(stderr,
			    "taw fit: tree H=%u sel1=%u q=%u model=%zu "
			    "bpc=%.4f est=%.0f B\n",
			    hidden, sel1, tquant, tb.n, tree_bpb * 8.0,
			    tree_est / 8.0);
		}
		if (tb.n > budget)
			st = st_mix == TAW_OK ? TAW_OK : TAW_E_NOMEM;
		else if (!force_mix && (force_tree || tree_est < best_est)) {
			free(best.model);
			best = (taw_model_choice){
				.profile = TAW_PROFILE_MDL_TREE,
				.quant = tquant,
				.model = tb.p,
				.model_size = tb.n,
			};
			tb = (taw_buf){0};
		}
	}
	taw_buf_free(&tb);
	free(sm);
	if (st != TAW_OK)
		return st;
	if (force_tree && best.model == nullptr)
		return TAW_E_ARG;
	if (!force_mix && !force_tree && best.model == nullptr &&
	    st_mix != TAW_OK)
		return st_mix;
	if (best.model)
		*out = best;
	return TAW_OK;
}
