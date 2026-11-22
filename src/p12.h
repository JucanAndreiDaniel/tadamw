/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Arithmetic profile 12, MDL_TREE (docs/format.md §2.12, §5.2).
 *
 * An input-specific, MDL-pruned binary context tree over cheap integer
 * features. All leaves share a frozen hidden layer (the first layer of a
 * profile-1-style mixer); each leaf owns one tiny output weight row. One
 * path is taken per bit, so decoding is deterministic and integer apart
 * from the two dot products, which follow profile 1's float conventions.
 * APM and the range coder are unchanged.
 */
#ifndef TAW_P12_H
#define TAW_P12_H

#include <stddef.h>
#include <stdint.h>

#include "p1.h"
#include "taw_internal.h"
#include "tadamw/tadamw.h"

enum {
	P12_NF = 25,          /* context-model inputs (shared hidden layer) */
	P12_NFEAT = 10,       /* cheap tree-split features */
	P12_MAX_LEAVES = 32,
	P12_MAX_NODES = 2 * P12_MAX_LEAVES - 1,
	P12_NODE_BYTES = 4,
	P12_MODEL_HEAD = 8,
	P12_HMAX = 16,        /* hidden units; 1..P12_HMAX */
};

/* Split features (all decoder-visible, small non-negative integers). */
enum {
	P12_F_BITPOS, P12_F_MATCH, P12_F_C0HI, P12_F_C1HI, P12_F_MATCHON,
	P12_F_S0SIGN, P12_F_S12SIGN, P12_F_C0LO, P12_F_C1LO, P12_F_S2SIGN,
};

typedef struct {
	uint8_t feat;  /* 0..P12_NFEAT-1 internal; 0xFF = leaf */
	uint8_t thr;   /* internal: go left when value < thr */
	uint8_t left;  /* child node index; for a leaf, the leaf index */
	uint8_t right; /* child node index; 0 for a leaf */
} p12_node;

typedef struct p12_tree {
	p1_shape geo;
	unsigned h, sel1, nnodes, nleaves;
	p12_node node[P12_MAX_NODES];
	p1_net net;                 /* holds the shared first-layer weights */
	float w2[P12_MAX_LEAVES][P1_XPAD]; /* dequantized leaf output rows */
} p12_tree;

unsigned p12_feature(unsigned f, const p1_sel_ctx *sc, const int16_t *st);
/* Hidden activations, bit-identical to profile 1's first layer. */
void p12_hidden(const p12_tree *t, const float *x, const p1_sel_ctx *sc,
    float *h);
/* Stretch-domain output t in [-2047, 2047], ready for p1_squash(). */
int32_t p12_eval(const p12_tree *t, const float *x, const p1_sel_ctx *sc);
/* Same, from precomputed hidden activations and integer inputs (fit.c). */
int32_t p12_eval_h(const p12_tree *t, const float *h, const p1_sel_ctx *sc,
    const int16_t *st);
int32_t p12_leaf_t(const float *w, const float *h, unsigned nh);

uint64_t p12_model_size(unsigned s1, unsigned h, unsigned nnodes,
    unsigned nleaves, uint8_t quant);
void p12_serialize(const p12_tree *t, const uint8_t *w1, size_t w1_size,
    const uint8_t *leaf, size_t leaf_size, taw_buf *b);
taw_status p12_load(p12_tree *t, uint8_t quant, const uint8_t *model,
    size_t size);
void p12_free(p12_tree *t);

#endif
