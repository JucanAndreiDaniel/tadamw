/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 10, F32_CHAINED_MIX (docs/format.md §2.3, §5.2).
 *
 * A chain of three frozen single-layer predictors. Each unit sees a fixed
 * subset of profile 1's 25 stretched context inputs plus the bias; units 1
 * and 2 additionally receive the previous unit's clamped estimate
 * ("information inheritance"). The integer context models, squash/stretch,
 * APM stages and range coder are shared with profile 1.
 */
#ifndef TAW_P10_H
#define TAW_P10_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"
#include "tadamw/tadamw.h"

/* ---- shape constants (normative) ---------------------------------------- */

enum {
	P10_UNITS = 3,
	P10_MODEL_HEAD = 8,
	/* own inputs per unit (nested: each higher unit sees the lower orders) */
	P10_M0 = 3, P10_M1 = 11, P10_M2 = 26,
	/* row lengths: own inputs + inheritance input for units 1 and 2 */
	P10_ROW0 = P10_M0, P10_ROW1 = P10_M1 + 1, P10_ROW2 = P10_M2 + 1,
	P10_MAXROW = P10_ROW2,
};

extern const uint8_t p10_nin[P10_UNITS];    /* own input count */
extern const uint8_t p10_rowlen[P10_UNITS]; /* weights per row */
/* p10_subset[u][0..p10_nin[u]-1] are indices into x[0..25]. */
extern const uint8_t p10_subset[P10_UNITS][P10_MAXROW];

typedef struct {
	uint8_t lines_log2;
	uint8_t hist_log2;
	uint8_t match_log2;
	uint8_t sel[P10_UNITS]; /* weight-set selector for each unit */
} p10_shape;

typedef struct p10_net {
	p10_shape s;
	uint32_t nset[P10_UNITS];
	float *w[P10_UNITS]; /* [nset[u]][p10_rowlen[u]] */
} p10_net;

/* Geometry for the shared context models. */
static inline p1_shape p10_ctx_shape(const p10_shape *s)
{
	return (p1_shape){
		.lines_log2 = s->lines_log2,
		.hist_log2 = s->hist_log2,
		.match_log2 = s->match_log2,
		.sel1 = s->sel[1],
		.sel2 = s->sel[2],
		.hidden = 0,
	};
}

void p10_head_write(uint8_t out[P10_MODEL_HEAD], const p10_shape *s);
taw_status p10_head_parse(const uint8_t *p, size_t n, p10_shape *s);
uint64_t p10_weights_size(const p10_shape *s, uint8_t quant);
bool p10_quant_supported(uint8_t quant);

/* Parse and dequantize a model section (head + weights). */
taw_status p10_net_load(p10_net *n, uint8_t quant, const uint8_t *model,
    size_t size);
void p10_net_free(p10_net *n);

/* Stretch-domain output in [-2047, 2047]. x has P1_XPAD entries,
 * x[P1_IN_BIAS] = 1 and x[P1_F..] = 0. */
int32_t p10_net_eval(const p10_net *n, const float *x, const p1_sel_ctx *sc);

#endif
