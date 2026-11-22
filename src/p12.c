/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Profile 12 model section and scalar inference (docs/format.md §2.12). */
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "half.h"
#include "p12.h"
#include "quant.h"
#include "taw_internal.h"

unsigned p12_feature(unsigned f, const p1_sel_ctx *sc, const int16_t *st)
{
	switch (f) {
	case P12_F_BITPOS:  return sc->bitpos;
	case P12_F_MATCH:   return sc->msel;
	case P12_F_C0HI:    return sc->c0 >> 4;
	case P12_F_C1HI:    return sc->c1 >> 4;
	case P12_F_MATCHON: return sc->msel != 0;
	case P12_F_S0SIGN:  return st[0] >= 0;
	case P12_F_S12SIGN: return st[P1_NCM] >= 0;
	case P12_F_C0LO:    return sc->c0 & 15;
	case P12_F_C1LO:    return sc->c1 & 15;
	case P12_F_S2SIGN:  return st[2] >= 0;
	default: return 0;
	}
}

/* The normative dot4: four partial sums, combined (s0+s1)+(s2+s3). */
static inline float dot4(const float *w, size_t ws, const float *x,
    unsigned n)
{
	float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	for (unsigned i = 0; i < n; i++)
		s[i & 3] += w[i * ws] * x[i];
	return (s[0] + s[1]) + (s[2] + s[3]);
}

void p12_hidden(const p12_tree *t, const float *x, const p1_sel_ctx *sc,
    float *h)
{
	const unsigned sel = p1_sel_value(t->sel1, sc);
	const float *w1 = t->net.w1 + (size_t)sel * P1_F * t->net.hpad;
	for (unsigned j = 0; j < t->h; j++) {
		float a = dot4(w1 + j, t->net.hpad, x, P1_F);
		h[j] = a < -8.0f ? -8.0f : a > 8.0f ? 8.0f : a;
	}
}

int32_t p12_leaf_t(const float *w, const float *h, unsigned nh)
{
	float z = dot4(w, 1, h, nh) + w[nh];
	const float lim = 2047.0f / 256.0f;
	if (z > lim)
		z = lim;
	else if (z < -lim)
		z = -lim;
	return (int32_t)(z * 256.0f);
}

int32_t p12_eval_h(const p12_tree *t, const float *h, const p1_sel_ctx *sc,
    const int16_t *st)
{
	unsigned ni = 0;
	for (unsigned depth = 0; depth < P12_MAX_NODES; depth++) {
		const p12_node *nd = &t->node[ni];
		if (nd->feat == 0xFF)
			break;
		unsigned v = p12_feature(nd->feat, sc, st);
		ni = v < nd->thr ? nd->left : nd->right;
	}
	return p12_leaf_t(t->w2[t->node[ni].left], h, t->h);
}

int32_t p12_eval(const p12_tree *t, const float *x, const p1_sel_ctx *sc)
{
	int16_t st[P12_NF];
	for (unsigned i = 0; i < P12_NF; i++)
		st[i] = (int16_t)(x[i] * 256.0f); /* exact: x[i] = st[i]/256 */
	float h[P1_HMAX] = {0};
	p12_hidden(t, x, sc, h);
	return p12_eval_h(t, h, sc, st);
}

static uint64_t row_size(uint8_t quant, unsigned len)
{
	return taw_quant_row_bytes(quant, len);
}

uint64_t p12_model_size(unsigned s1, unsigned h, unsigned nnodes,
    unsigned nleaves, uint8_t quant)
{
	return P12_MODEL_HEAD +
	    (uint64_t)s1 * h * row_size(quant, P1_F) +
	    (uint64_t)nnodes * P12_NODE_BYTES +
	    (uint64_t)nleaves * row_size(quant, h + 1);
}

void p12_serialize(const p12_tree *t, const uint8_t *w1, size_t w1_size,
    const uint8_t *leaf, size_t leaf_size, taw_buf *b)
{
	uint8_t head[P12_MODEL_HEAD] = {0};
	head[0] = t->geo.lines_log2;
	head[1] = t->geo.hist_log2;
	head[2] = t->geo.match_log2;
	head[3] = (uint8_t)t->h;
	head[4] = (uint8_t)t->sel1;
	head[5] = (uint8_t)t->nnodes;
	head[6] = (uint8_t)t->nleaves;
	taw_buf_put(b, head, sizeof head);
	taw_buf_put(b, w1, w1_size);
	for (unsigned i = 0; i < t->nnodes; i++) {
		const p12_node *n = &t->node[i];
		taw_buf_put_u8(b, n->feat);
		taw_buf_put_u8(b, n->thr);
		taw_buf_put_u8(b, n->left);
		taw_buf_put_u8(b, n->right);
	}
	taw_buf_put(b, leaf, leaf_size);
}

/* Dequantize one leaf row with the shared, validated row decoder. */
static taw_status read_leaf(const uint8_t **pp, uint8_t quant, unsigned len,
    float *dst)
{
	const uint64_t n = taw_quant_row_bytes(quant, len);
	taw_status st = taw_quant_row_decode(quant, *pp, len, dst);
	*pp += n;
	return st;
}

taw_status p12_load(p12_tree *t, uint8_t quant, const uint8_t *model,
    size_t size)
{
	memset(t, 0, sizeof *t);
	if (!p1_quant_supported(quant) || size < P12_MODEL_HEAD)
		return TAW_E_UNSUPPORTED;
	const uint8_t *p = model;
	t->geo = (p1_shape){.lines_log2 = p[0], .hist_log2 = p[1],
		.match_log2 = p[2]};
	t->h = p[3];
	t->sel1 = p[4];
	t->nnodes = p[5];
	t->nleaves = p[6];
	if (p[7])
		return TAW_E_FORMAT;
	if (t->h < 1 || t->h > P12_HMAX || t->sel1 >= P1_SEL_COUNT ||
	    t->nleaves < 1 || t->nleaves > P12_MAX_LEAVES ||
	    t->nnodes != 2 * t->nleaves - 1)
		return TAW_E_FORMAT;
	if (t->geo.lines_log2 < P1_LINES_MIN || t->geo.lines_log2 > P1_LINES_MAX ||
	    t->geo.hist_log2 < P1_HIST_MIN || t->geo.hist_log2 > P1_HIST_MAX ||
	    t->geo.match_log2 < P1_MATCH_MIN || t->geo.match_log2 > P1_MATCH_MAX)
		return TAW_E_FORMAT;
	const unsigned s1 = p1_sel_size[t->sel1];
	const size_t w1_size = (size_t)s1 * t->h * row_size(quant, P1_F);
	const size_t leaf_size = (size_t)t->nleaves * row_size(quant, t->h + 1);
	if (size != P12_MODEL_HEAD + w1_size +
	    (size_t)P12_NODE_BYTES * t->nnodes + leaf_size)
		return TAW_E_FORMAT;

	/* Reconstruct a minimal profile-1 model holding just the shared first
	 * layer, so its dequantization and shape validation are reused. */
	const size_t dummy = row_size(quant, t->h + 1);
	uint8_t *tmp = malloc(P1_MODEL_HEAD + w1_size + dummy);
	if (!tmp)
		return TAW_E_NOMEM;
	p1_shape s1shape = t->geo;
	s1shape.sel1 = (uint8_t)t->sel1;
	s1shape.sel2 = P1_SEL_NONE;
	s1shape.hidden = (uint8_t)t->h;
	p1_head_write(tmp, &s1shape);
	memcpy(tmp + P1_MODEL_HEAD, model + P12_MODEL_HEAD, w1_size);
	memset(tmp + P1_MODEL_HEAD + w1_size, 0, dummy);
	taw_status st = p1_net_load(&t->net, quant, tmp,
	    P1_MODEL_HEAD + w1_size + dummy);
	free(tmp);
	if (st != TAW_OK)
		return st;

	p += P12_MODEL_HEAD + w1_size;
	for (unsigned i = 0; i < t->nnodes; i++) {
		t->node[i] = (p12_node){.feat = p[0], .thr = p[1],
			.left = p[2], .right = p[3]};
		p += 4;
	}
	for (unsigned i = 0; i < t->nleaves; i++) {
		st = read_leaf(&p, quant, t->h + 1, t->w2[i]);
		if (st != TAW_OK)
			goto fail;
	}
	uint8_t seen[P12_MAX_NODES] = {0};
	unsigned stack[P12_MAX_NODES], sp = 0, count = 0;
	stack[sp++] = 0;
	while (sp) {
		unsigned ni = stack[--sp];
		if (seen[ni]) {
			st = TAW_E_FORMAT;
			goto fail;
		}
		seen[ni] = 1;
		count++;
		const p12_node *n = &t->node[ni];
		if (n->feat == 0xFF) {
			if (n->left >= t->nleaves || n->right != 0) {
				st = TAW_E_FORMAT;
				goto fail;
			}
			continue;
		}
		if (n->feat >= P12_NFEAT || n->thr == 0 ||
		    n->left >= t->nnodes || n->right >= t->nnodes) {
			st = TAW_E_FORMAT;
			goto fail;
		}
		stack[sp++] = n->left;
		stack[sp++] = n->right;
	}
	if (count != t->nnodes)
		st = TAW_E_FORMAT;
	return st;
fail:
	p1_net_free(&t->net);
	return st;
}

void p12_free(p12_tree *t)
{
	p1_net_free(&t->net);
}
