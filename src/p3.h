/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 3, SOFT_GATED (docs/format.md §2.3, §5.2).
 *
 * A soft-gated mixture of K linear experts over the shared hidden layer.
 * The context models, squash, APM stages and range coder are exactly those
 * of profile 1 (p1_ctx / p1_squash / p1_stretch_tab); only the mixer and its
 * model section are new.
 */
#ifndef TAW_P3_H
#define TAW_P3_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"
#include "tadamw/tadamw.h"

/* ---- shape constants (normative) ---------------------------------------- */

enum {
	P3_HEAD = 8,
	P3_KMIN = 2,
	P3_KMAX = 4,
	P3_HMAX = 8,
	P3_GC0 = 256, /* one gate logit per partial byte c0 */
	P3_GMSEL = 8, /* one gate logit per match class */
	P3_GCONF = 16, /* one gate logit per confidence bucket */
	P3_WSUM = 256, /* gate weights sum to this */
};

/* bit 0 of gate_flags enables the msel gate table, bit 1 the confidence
 * table. With bit 1 set, expert 0 is a fixed raw context mixture and is
 * only allowed when hidden = 0. */
enum { P3_FLAG_MSEL = 1, P3_FLAG_CONF = 2 };

/* Arithmetic profile 7 (SOFT_GATED_CAL): profile 3's model section with an
 * 8-byte learned APM-blend extension inserted after the 8-byte head.
 *
 *   off 8:      w0 (u8, 0..15)  weight of p_mix
 *   off 9:      w1 (u8, 0..15)  weight of APM1
 *   off 10:     w2 (u8, 0..15)  weight of APM2
 *   off 11..15: reserved, must be 0
 */
enum {
	P3_CAL_EXT = 8,
	P3_CAL_HEAD = P3_HEAD + P3_CAL_EXT,
	P3_W_MAX = 15,
};

typedef struct {
	uint8_t w0, w1, w2;
} p3_cal;

/* The identity blend: profile 3's fixed (2*p_mix + q1 + q2) >> 2. */
#define P3_CAL_IDENTITY ((p3_cal){ .w0 = 2, .w1 = 1, .w2 = 1 })

typedef struct {
	uint8_t K;          /* experts, 2..4 */
	uint8_t hidden;     /* shared hidden units, 0..P3_HMAX */
	uint8_t lines_log2; /* geometry, same limits as profile 1 */
	uint8_t hist_log2;
	uint8_t match_log2;
	uint8_t gate_flags;
} p3_shape;

/* ---- the integer softmax ------------------------------------------------- */

/* E[d] = 256 * 2^(-d/8) approximated by E[0]=256,
 * E[d] = (E[d-1] * 235) >> 8. Indexed with d = min(511, maxlogit - logit). */
extern uint16_t p3_e_tab[512];
void p3_tables_init(void);

/* ---- the frozen network -------------------------------------------------- */

typedef struct p3_net p3_net;
struct p3_net {
	p3_shape s;
	unsigned hpad;       /* hidden rounded up to 8 */
	float *w1;           /* [P3_HMAX][P1_F], zero padded, scalar dot4 */
	float *v;            /* [K][vstride]: H+1 output weights+bias, or 26 */
	float *b;            /* [K] output bias (H > 0 only) */
	size_t vstride;
	int8_t *gc0;         /* [K][P3_GC0] */
	int8_t *gms;         /* [K][P3_GMSEL], or nullptr */
	int8_t *gcf;         /* [K][P3_GCONF], or nullptr */
	p3_cal cal;          /* profile 7 blend; identity for profile 3 */
};

bool p3_quant_supported(uint8_t quant);
uint64_t p3_weights_size(const p3_shape *s, uint8_t quant);
void p3_head_write(uint8_t out[P3_HEAD], const p3_shape *s);
taw_status p3_head_parse(const uint8_t *p, size_t n, p3_shape *s);
taw_status p3_net_load(p3_net *n, uint8_t quant, const uint8_t *model,
    size_t size);
void p3_net_free(p3_net *n);
/* Stretch-domain output, [-2047, 2047]. x has P1_XPAD entries with
 * x[P1_IN_BIAS] = 1 and x[P1_F..] = 0. */
int32_t p3_net_eval(const p3_net *n, const float *x, const p1_sel_ctx *sc);

/* ---- profile 7: learned APM blend (definitions need p3_shape / p3_net) ---- */

/* Size of a model section, with or without the blend extension. */
uint64_t p3_model_size(const p3_shape *s, uint8_t quant, bool calibrated);
/* Write / parse the P3_CAL_HEAD head and load a profile 7 model. */
void p3_cal_head_write(uint8_t out[P3_CAL_HEAD], const p3_shape *s,
    const p3_cal *cal);
taw_status p3_cal_head_parse(const uint8_t *p, size_t n, p3_shape *s,
    p3_cal *cal);
taw_status p3_cal_net_load(p3_net *n, uint8_t quant, const uint8_t *model,
    size_t size);
/* Integer blend of p_mix, q1, q2; all 4-bit weights, numerator < 2^22. */
static inline uint32_t p3_cal_prob(const p3_cal *cal, uint32_t p_mix,
    uint32_t q1, uint32_t q2)
{
	unsigned den = (unsigned)cal->w0 + cal->w1 + cal->w2;
	return (cal->w0 * p_mix + cal->w1 * q1 + cal->w2 * q2) / den;
}

#endif
