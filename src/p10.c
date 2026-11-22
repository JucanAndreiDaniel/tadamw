/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 10 model section: parsing, dequantization, scalar inference. */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "p10.h"
#include "quant.h"
#include "taw_internal.h"

const uint8_t p10_nin[P10_UNITS] = {P10_M0, P10_M1, P10_M2};
const uint8_t p10_rowlen[P10_UNITS] = {P10_ROW0, P10_ROW1, P10_ROW2};

/* Normative input subsets (docs/format.md §2.3.1). Padded slots are never
 * read: only the first p10_nin[u] entries of each row are used. The subsets
 * are nested: a higher-order unit also sees every lower-order input, plus
 * the previous unit's estimate. */
const uint8_t p10_subset[P10_UNITS][P10_MAXROW] = {
	{0, 12, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{0, 12, 1, 13, 2, 14, 3, 15, 6, 18, 25, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
	 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0},
};

/* ---- model head ---------------------------------------------------------- */

void p10_head_write(uint8_t out[P10_MODEL_HEAD], const p10_shape *s)
{
	out[0] = s->lines_log2;
	out[1] = s->hist_log2;
	out[2] = s->match_log2;
	for (unsigned u = 0; u < P10_UNITS; u++)
		out[3 + u] = s->sel[u];
	out[6] = 0;
	out[7] = 0;
}

taw_status p10_head_parse(const uint8_t *p, size_t n, p10_shape *s)
{
	if (n < P10_MODEL_HEAD)
		return TAW_E_FORMAT;
	*s = (p10_shape){
		.lines_log2 = p[0],
		.hist_log2 = p[1],
		.match_log2 = p[2],
		.sel = {p[3], p[4], p[5]},
	};
	if (p[6] || p[7])
		return TAW_E_FORMAT;
	if (s->lines_log2 < P1_LINES_MIN || s->lines_log2 > P1_LINES_MAX ||
	    s->hist_log2 < P1_HIST_MIN || s->hist_log2 > P1_HIST_MAX ||
	    s->match_log2 < P1_MATCH_MIN || s->match_log2 > P1_MATCH_MAX)
		return TAW_E_FORMAT;
	for (unsigned u = 0; u < P10_UNITS; u++)
		if (s->sel[u] >= P1_SEL_COUNT)
			return TAW_E_FORMAT;
	return TAW_OK;
}

bool p10_quant_supported(uint8_t quant)
{
	return quant == TAW_QUANT_F16 || quant == TAW_QUANT_Q8;
}

uint64_t p10_weights_size(const p10_shape *s, uint8_t quant)
{
	uint64_t bytes = 0;
	for (unsigned u = 0; u < P10_UNITS; u++)
		bytes += (uint64_t)p1_sel_size[s->sel[u]] *
		    taw_quant_row_bytes(quant, p10_rowlen[u]);
	return bytes;
}

/* ---- loading ------------------------------------------------------------- */

/* Rows are dequantized through profile 1's shared quant.c code, so the
 * weight bits are identical to the same row in profile 1. */

taw_status p10_net_load(p10_net *n, uint8_t quant, const uint8_t *model,
    size_t size)
{
	memset(n, 0, sizeof *n);
	if (!p10_quant_supported(quant))
		return TAW_E_UNSUPPORTED;
	taw_status st = p10_head_parse(model, size, &n->s);
	if (st != TAW_OK)
		return st;
	if (size - P10_MODEL_HEAD != p10_weights_size(&n->s, quant))
		return TAW_E_FORMAT;

	for (unsigned u = 0; u < P10_UNITS; u++) {
		n->nset[u] = p1_sel_size[n->s.sel[u]];
		const size_t count = (size_t)n->nset[u] * p10_rowlen[u];
		n->w[u] = aligned_alloc(64,
		    (count * sizeof(float) + 63) & ~(size_t)63);
		if (!n->w[u]) {
			p10_net_free(n);
			return TAW_E_NOMEM;
		}
		memset(n->w[u], 0, count * sizeof(float));
	}
	const uint8_t *p = model + P10_MODEL_HEAD;
	for (unsigned u = 0; u < P10_UNITS; u++)
		for (uint32_t a = 0; a < n->nset[u]; a++) {
			st = taw_quant_row_decode(quant, p, p10_rowlen[u],
			    n->w[u] + (size_t)a * p10_rowlen[u]);
			if (st != TAW_OK) {
				p10_net_free(n);
				return st;
			}
			p += taw_quant_row_bytes(quant, p10_rowlen[u]);
		}
	return TAW_OK;
}

void p10_net_free(p10_net *n)
{
	for (unsigned u = 0; u < P10_UNITS; u++)
		free(n->w[u]);
	memset(n, 0, sizeof *n);
}

/* ---- inference ----------------------------------------------------------- */

/* The normative dot product: four partial sums over i mod 4, combined as
 * (s0 + s1) + (s2 + s3). */
static inline float dot4(const float *w, const float *v, unsigned n)
{
	float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	for (unsigned i = 0; i < n; i++)
		s[i & 3] += w[i] * v[i];
	return (s[0] + s[1]) + (s[2] + s[3]);
}

int32_t p10_net_eval(const p10_net *n, const float *x, const p1_sel_ctx *sc)
{
	const float lim = 2047.0f / 256.0f;
	float prev = 0.0f, c = 0.0f;
	for (unsigned u = 0; u < P10_UNITS; u++) {
		const uint32_t a = p1_sel_value(n->s.sel[u], sc);
		const float *w = n->w[u] + (size_t)a * p10_rowlen[u];
		float v[P10_MAXROW];
		const unsigned m = p10_nin[u];
		for (unsigned i = 0; i < m; i++)
			v[i] = x[p10_subset[u][i]];
		if (u)
			v[m] = prev;
		float z = dot4(w, v, p10_rowlen[u]);
		c = z < -lim ? -lim : z > lim ? lim : z;
		prev = c;
	}
	return (int32_t)(c * 256.0f);
}
