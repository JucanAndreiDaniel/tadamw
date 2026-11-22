/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Weight row formats (docs/format.md §3, §5.1): sizes, decoding with
 * validation, and (encoder) quantization. A row is `len` weights with its
 * own scales; profile 1 rows are 26 or H + 1 weights. */
#ifndef TAW_QUANT_H
#define TAW_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "tadamw/tadamw.h"

enum { TAW_NVFP4_BLOCK = 16 };

/* Value tables (normative): IQ4 is ggml's IQ4_NL table, IQ2 a 4-level
 * Lloyd-Max table for a Gaussian, IQ8 is k * (|k| + 128) for k = -128..127. */
extern const int8_t taw_iq4_tab[16];
extern const int8_t taw_iq2_tab[4];
static inline int32_t taw_iq8_val(int8_t k)
{
	int32_t v = k;
	return v * ((v < 0 ? -v : v) + 128);
}

/* E4M3 (bias 7, no infinities, 0x7F/0xFF = NaN) and E2M1 to binary32;
 * both exact. */
float taw_e4m3_to_f32(uint8_t b);
float taw_e2m1_to_f32(uint8_t nib);

/* Is this a weight row format profile 1 knows (F16, Q8 and the formats
 * added with them)? */
bool taw_quant_is_row_format(uint8_t quant);
/* Bytes one row of len weights takes. */
uint64_t taw_quant_row_bytes(uint8_t quant, uint64_t len);
/* Decode and validate one row (len <= 64). F16 rows are decoded here too
 * (scalar); callers may use F16C instead, the results are identical. */
taw_status taw_quant_row_decode(uint8_t quant, const uint8_t *p,
    unsigned len, float *out);

/* Encoder: quantize a row, writing taw_quant_row_bytes(quant, len) bytes.
 * The choice of codes minimizes the squared error of the dequantized row
 * (as the decoder computes it) over a small search of scales. */
void taw_quant_row_encode(uint8_t quant, const float *w, unsigned len,
    uint8_t *out);

/* Public weight choice <-> quant id (TAW_QUANT_NONE for AUTO). */
uint8_t taw_quant_from_weights(taw_weights w);

/* Encoder-only helpers for quantization-aware training (fit.c). The format
 * and the decoder are unchanged: they reuse the same level tables and the
 * same F16 scale representation as taw_quant_row_encode/decode.
 *  - row_scale: cheap amax / max-level F16 scale (the search's start point).
 *  - simulate: dequantized row under `scale` (>0), or row_scale when scale
 *    is 0; *scale_out receives the scale used. Returns false for Q2/NVFP4.
 *  - encode_with_scale: encode a row using a prescribed scale instead of
 *    searching; returns false for Q2/NVFP4 (fall back to the normal search). */
float taw_quant_row_scale(uint8_t quant, const float *w, unsigned len);
bool taw_quant_simulate(uint8_t quant, const float *w, unsigned len,
    float scale, float *out, float *scale_out);
bool taw_quant_row_encode_with_scale(uint8_t quant, const float *w,
    unsigned len, float scale, uint8_t *out);

#endif
