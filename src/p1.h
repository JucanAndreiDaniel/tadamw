/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 1, F32_STRICT_MIX (docs/format.md §2.2, §5.1).
 *
 * Adaptive integer context models produce stretched bit probabilities; a
 * small frozen MLP (weights fitted offline by the encoder) mixes them into
 * one 16-bit probability. Everything here is shared by the decoder, the
 * encoder's coding pass, and the encoder's feature extraction for fitting.
 */
#ifndef TAW_P1_H
#define TAW_P1_H

#include <stddef.h>
#include <stdint.h>

#include "taw/taw.h"

/* ---- shape constants (normative) ---------------------------------------- */

enum {
	P1_NCM = 12,     /* context models (each gives two inputs) */
	P1_NCTX = 25,    /* context-model inputs: 2 per model + match */
	P1_F = 26,       /* mixer inputs: P1_NCTX + bias */
	P1_NHASH = 10,   /* hashed context models (all but o0, o1) */
	P1_HMAX = 32,    /* max hidden units */
	P1_MODEL_HEAD = 8,
	P1_MINLEN = 5,   /* match model: minimum verified length */
	P1_LIMIT_SLOW = 30,   /* context slot count limit */
	P1_LIMIT_HMAP = 1023, /* history map count limit */
	P1_MAXVERIFY = 32,
};

/* Context models; model k gives inputs k (direct) and P1_NCM + k (history).
 * The S models are sparse: S23 = bytes 2 and 3 back, and so on. */
enum {
	P1_CM_O0, P1_CM_O1, P1_CM_O2, P1_CM_O3, P1_CM_O4, P1_CM_O6,
	P1_CM_S23, P1_CM_WORD, P1_CM_S34, P1_CM_S24, P1_CM_S14, P1_CM_S1278
};
enum { P1_IN_MATCH = 2 * P1_NCM, P1_IN_BIAS };

/* Weight-set selectors. */
enum {
	P1_SEL_NONE,   /* 1 set */
	P1_SEL_BITPOS, /* 8: bit index within the byte, 0 = MSB */
	P1_SEL_MATCH,  /* 8: match state, 0 = no active match */
	P1_SEL_C0,     /* 256: partial byte with leading 1 */
	P1_SEL_C1,     /* 256: previous byte */
	P1_SEL_C0C1HI, /* 4096: (c1 >> 4) << 8 | c0 */
	P1_SEL_COUNT
};
extern const uint16_t p1_sel_size[P1_SEL_COUNT];

/* Context slots and history-map entries are stored XOR these, so that
 * zeroed memory is the initial state (p1_ctx.c). */
#define P1_SLOT_INIT 0x80000100u
#define P1_HM_INIT 0x80000000u

/* Geometry limits a decoder accepts. */
enum {
	P1_LINES_MIN = 8, P1_LINES_MAX = 22,
	P1_HIST_MIN = 12, P1_HIST_MAX = 26,
	P1_MATCH_MIN = 10, P1_MATCH_MAX = 24,
};

typedef struct {
	uint8_t lines_log2; /* hashed-table lines per context, log2 */
	uint8_t hist_log2;  /* match history ring, log2 bytes */
	uint8_t match_log2; /* match hash table entries, log2 */
	uint8_t sel1, sel2; /* selectors for layer 1 / output layer */
	uint8_t hidden;     /* hidden units, 0..P1_HMAX (0 = single layer) */
} p1_shape;

/* ---- squash / stretch (integer, normative) ------------------------------ */

/* t in stretch units (1/256), clamped to [-2047, 2047] -> P(1) * 65536. */
uint32_t p1_squash(int32_t t);
/* 12-bit probability -> stretch units, [-2047, 2047]. */
extern int16_t p1_stretch_tab[4096];
/* The same as int32 (for gathers), and the counter rate table R[n]. */
extern int32_t p1_stretch32[4096];
extern uint32_t p1_rate32[1024];
void p1_tables_init(void);

/* ---- context models ----------------------------------------------------- */

typedef struct {
	uint8_t bitpos; /* 0..7 */
	uint8_t c0;     /* 1..255 */
	uint8_t c1;
	uint8_t msel;   /* 0..7 */
} p1_sel_ctx;

typedef struct {
	/* hot per-bit state, parallel arrays padded to 16 lanes */
	alignas(32) uint32_t val[16]; /* slot values read by inputs */
	alignas(32) uint32_t hmi[16]; /* history-map index per model */
	alignas(32) uint32_t hmv[16]; /* history-map entries read by inputs */
	alignas(32) uint32_t si[16];  /* arena index of each model's slot */
	alignas(32) uint32_t lb[16];  /* arena index of each hashed model's line */
	uint32_t c0, bitpos, nib, c4, c8, word, pos;
	uint32_t mptr, mlen, msi;
	int mexp;   /* expected byte, or -1 */
	int mbit;   /* expected bit for the current bit, or -1 */
	bool avx2;  /* use the AVX2 context kernels */
	/* geometry */
	unsigned lines_log2;
	uint32_t hist_mask, match_mask;
	/* tables; slots are stored XOR P1_SLOT_INIT so that zero = initial */
	uint32_t *arena; /* [O0 256][O1 65536][T_k: 2^L x 16] */
	size_t arena_n;
	uint32_t *mtab, sm[16];
	uint8_t *hist;
	alignas(32) uint32_t hmap[P1_NCM * 256]; /* history maps, [k][h] */
	alignas(32) uint32_t hash[16]; /* byte hashes g_k, padded */
} p1_ctx;

taw_status p1_ctx_init(p1_ctx *c, const p1_shape *s);
void p1_ctx_reset(p1_ctx *c);
void p1_ctx_free(p1_ctx *c);
/* Mixer inputs x[0..P1_NCTX-1] = stretch / 256 (exact) and the selector
 * context for the next bit. x[P1_IN_BIAS] is left to the caller. */
void p1_ctx_inputs(p1_ctx *c, float *x, p1_sel_ctx *sc);
void p1_ctx_update(p1_ctx *c, int bit);

/* AVX2 kernels for the 12 context models (p1_avx2.c); identical results
 * to the scalar code in p1_ctx.c. */
void p1_cm_inputs_avx2(p1_ctx *c, float *x);
void p1_cm_update_avx2(p1_ctx *c, int bit);
/* Byte hashes g[0..9] from the history; line hashes hh = H(g, q, 0).
 * g and hh are 16 aligned lanes. */
void p1_byte_hashes_avx2(uint32_t c4, uint32_t c8, uint32_t word,
    uint32_t *g);
void p1_line_hashes_avx2(const uint32_t *g, uint32_t q, uint32_t *hh);

static inline uint32_t p1_sel_value(unsigned sel, const p1_sel_ctx *sc)
{
	switch (sel) {
	case P1_SEL_BITPOS: return sc->bitpos;
	case P1_SEL_MATCH: return sc->msel;
	case P1_SEL_C0: return sc->c0;
	case P1_SEL_C1: return sc->c1;
	case P1_SEL_C0C1HI: return (uint32_t)(sc->c1 >> 4) << 8 | sc->c0;
	default: return 0;
	}
}

/* ---- the frozen network -------------------------------------------------- */

/* Mixer input vectors are P1_XPAD floats: x[0..P1_F-1] then zeros, so
 * that SIMD kernels may read whole vectors. */
enum { P1_XPAD = 32 };

typedef struct p1_net p1_net;
struct p1_net {
	p1_shape s;
	unsigned hpad; /* hidden rounded up to 8 */
	unsigned w2s;  /* output-layer row stride: hpad, or P1_XPAD if H = 0 */
	float *w1;     /* [S1][P1_F][hpad], transposed, zero padded */
	float *w2;     /* [S2][w2s] zero padded; no bias when H > 0 */
	float *b2;     /* [S2] output bias when H > 0 */
	/* z before the bias: dot4 over the output layer (format.md §2.2.8) */
	float (*mix)(const p1_net *n, const float *x, const float *w1,
	    const float *w2);
};

/* Bytes the weight payload needs for a shape and quant format. */
uint64_t p1_weights_size(const p1_shape *s, uint8_t quant);
bool p1_quant_supported(uint8_t quant);

/* Parse a model section (header + weights) and dequantize it. */
taw_status p1_net_load(p1_net *n, uint8_t quant, const uint8_t *model,
    size_t size);
void p1_net_free(p1_net *n);
/* Stretch-domain output, [-2047, 2047]. x has P1_XPAD entries,
 * x[P1_IN_BIAS] = 1 and x[P1_F..] = 0. */
int32_t p1_net_eval(const p1_net *n, const float *x, const p1_sel_ctx *sc);

/* The mixer kernels; all produce identical bits. */
float p1_mix_scalar(const p1_net *n, const float *x, const float *w1,
    const float *w2);
float p1_mix_avx2(const p1_net *n, const float *x, const float *w1,
    const float *w2);
float p1_mix_avx512(const p1_net *n, const float *x, const float *w1,
    const float *w2);
void p1_f16_to_f32_f16c(const uint16_t *src, float *dst, size_t n);

typedef enum { P1_ISA_SCALAR, P1_ISA_AVX2, P1_ISA_AVX512, P1_ISA_COUNT }
    p1_isa;
/* Best ISA this CPU and build support. */
p1_isa p1_isa_best(void);
bool p1_isa_available(p1_isa isa);
/* Testing: pin the kernels used by later p1_net_load calls (-1 = auto). */
void p1_isa_force(int isa);
/* The ISA that kernels picked now should use (forced or best). */
p1_isa p1_isa_current(void);

/* ---- serialization (encoder) -------------------------------------------- */

/* Write the 8-byte model head. */
void p1_head_write(uint8_t out[P1_MODEL_HEAD], const p1_shape *s);
taw_status p1_head_parse(const uint8_t *p, size_t n, p1_shape *s);

/* ---- encoder: fitting and model choice (fit.c) --------------------------- */

typedef struct {
	uint8_t profile, quant;
	uint8_t *model;
	size_t model_size;
} taw_model_choice;

/* Choose a profile and fit a model for src. unit = coding-unit size (n for
 * .aw, block_size for .awseek). Falls back to profile 0 when profile 1 is
 * not possible inside the budget. */
taw_status taw_model_fit(const uint8_t *src, size_t n, size_t unit,
    const taw_params *p, taw_model_choice *out);

#endif
