/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Weight row formats (docs/format.md §3, §5.1): decoded values pinned by
 * hand, validation, the encoder's quantizer, and archives in every format. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "half.h"
#include "quant.h"
#include "testdata.h"

static const uint8_t all_q[] = {TAW_QUANT_F16, TAW_QUANT_Q8, TAW_QUANT_Q4,
	TAW_QUANT_Q2, TAW_QUANT_FP8, TAW_QUANT_NVFP4, TAW_QUANT_IQ8,
	TAW_QUANT_IQ4, TAW_QUANT_IQ2};

static void minifloats(void)
{
	CHECK(taw_e4m3_to_f32(0x38) == 1.0f);
	CHECK(taw_e4m3_to_f32(0xB8) == -1.0f);
	CHECK(taw_e4m3_to_f32(0x7E) == 448.0f);
	CHECK(taw_e4m3_to_f32(0x08) == 1.0f / 64.0f);   /* min normal */
	CHECK(taw_e4m3_to_f32(0x01) == 1.0f / 512.0f);  /* min subnormal */
	CHECK(taw_e4m3_to_f32(0x07) == 7.0f / 512.0f);
	CHECK(taw_e4m3_to_f32(0x3B) == 1.375f);
	CHECK(taw_e4m3_to_f32(0x00) == 0.0f);
	/* monotone over the non-negative codes */
	for (unsigned b = 1; b < 0x7F; b++)
		CHECK(taw_e4m3_to_f32((uint8_t)b) > taw_e4m3_to_f32((uint8_t)(b - 1)));
	static const float e2[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
	for (unsigned k = 0; k < 8; k++) {
		CHECK(taw_e2m1_to_f32((uint8_t)k) == e2[k]);
		CHECK(taw_e2m1_to_f32((uint8_t)(k | 8)) == -e2[k]);
	}
	CHECK(taw_iq8_val(0) == 0);
	CHECK(taw_iq8_val(1) == 129);
	CHECK(taw_iq8_val(-1) == -129);
	CHECK(taw_iq8_val(127) == 32385);
	CHECK(taw_iq8_val(-128) == -32768);
	for (int k = -127; k <= 127; k++)
		CHECK(taw_iq8_val((int8_t)k) > taw_iq8_val((int8_t)(k - 1)));
}

static void put16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

/* Hand-decoded rows: scale 0.5 (0x3800). */
static void decode_values(void)
{
	float o[8];
	uint8_t r[16] = {0};
	put16(r, 0x3800);

	r[2] = 0x0F; r[3] = 0x83; /* Q4: codes 15, 0, 3, 8 */
	CHECK(taw_quant_row_decode(TAW_QUANT_Q4, r, 4, o) == TAW_OK);
	CHECK(o[0] == 3.5f && o[1] == -4.0f && o[2] == -2.5f && o[3] == 0.0f);

	r[2] = 0x0F; r[3] = 0x08; /* IQ4: 15, 0, 8 */
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ4, r, 3, o) == TAW_OK);
	CHECK(o[0] == 56.5f && o[1] == -63.5f && o[2] == 0.5f);

	r[2] = 0xE4; /* IQ2 / Q2 crumbs: 0, 1, 2, 3 */
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ2, r, 4, o) == TAW_OK);
	CHECK(o[0] == -63.5f && o[1] == -19.0f && o[2] == 19.0f &&
	    o[3] == 63.5f);
	put16(r + 2, 0xBC00); /* Q2 offset -1 */
	r[4] = 0xE4;
	CHECK(taw_quant_row_decode(TAW_QUANT_Q2, r, 4, o) == TAW_OK);
	CHECK(o[0] == -1.0f && o[1] == -0.5f && o[2] == 0.0f && o[3] == 0.5f);

	r[2] = 0x38; r[3] = 0xFE; r[4] = 0x01; /* FP8 */
	CHECK(taw_quant_row_decode(TAW_QUANT_FP8, r, 3, o) == TAW_OK);
	CHECK(o[0] == 0.5f && o[1] == -224.0f && o[2] == 1.0f / 1024.0f);

	r[2] = 0x02; r[3] = 0xFF; r[4] = 0x80; /* IQ8: 2, -1, -128 */
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ8, r, 3, o) == TAW_OK);
	CHECK(o[0] == 130.0f && o[1] == -64.5f && o[2] == -16384.0f);

	/* NVFP4, 18 weights = 2 blocks: scales 2.0 (0x40) and 1/64 (0x08);
	 * weight 0 = +6, weight 1 = -0.5, weight 16 = +1. */
	memset(r, 0, sizeof r);
	put16(r, 0x3800);
	r[2] = 0x40;
	r[3] = 0x08;
	r[4] = 0x97;      /* w0 code 7 (+6), w1 code 9 (-0.5) */
	r[4 + 8] = 0x02;  /* w16 code 2 (+1) */
	float o2[18];
	CHECK(taw_quant_row_decode(TAW_QUANT_NVFP4, r, 18, o2) == TAW_OK);
	CHECK(o2[0] == 6.0f && o2[1] == -0.5f && o2[2] == 0.0f);
	CHECK(o2[16] == 1.0f / 128.0f && o2[17] == 0.0f);
	CHECK(taw_quant_row_bytes(TAW_QUANT_NVFP4, 18) == 2 + 2 + 9);
}

static void validation(void)
{
	float o[8];
	uint8_t r[16] = {0};
	put16(r, 0x3C00);
	for (size_t k = 0; k < sizeof all_q; k++)
		CHECK(taw_quant_row_decode(all_q[k], r, 3, o) == TAW_OK);
	/* non-finite scale */
	put16(r, 0x7C00);
	for (size_t k = 0; k < sizeof all_q; k++)
		if (all_q[k] != TAW_QUANT_F16)
			CHECK(taw_quant_row_decode(all_q[k], r, 3, o) == TAW_E_FORMAT);
	put16(r, 0x3C00);
	/* padding bits */
	r[3] = 0x10; /* Q4/IQ4 len 3: high nibble of byte 1 is padding */
	CHECK(taw_quant_row_decode(TAW_QUANT_Q4, r, 3, o) == TAW_E_FORMAT);
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ4, r, 3, o) == TAW_E_FORMAT);
	CHECK(taw_quant_row_decode(TAW_QUANT_Q4, r, 4, o) == TAW_OK);
	r[3] = 0;
	r[2] = 0x40; /* IQ2 len 3: bits 6-7 are padding */
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ2, r, 3, o) == TAW_E_FORMAT);
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ2, r, 4, o) == TAW_OK);
	r[2] = 0;
	/* Q2: non-finite offset, padding */
	put16(r + 2, 0x7E00);
	CHECK(taw_quant_row_decode(TAW_QUANT_Q2, r, 3, o) == TAW_E_FORMAT);
	put16(r + 2, 0);
	r[4] = 0xC0;
	CHECK(taw_quant_row_decode(TAW_QUANT_Q2, r, 3, o) == TAW_E_FORMAT);
	r[4] = 0;
	/* FP8 NaN codes */
	r[3] = 0x7F;
	CHECK(taw_quant_row_decode(TAW_QUANT_FP8, r, 3, o) == TAW_E_FORMAT);
	r[3] = 0xFF;
	CHECK(taw_quant_row_decode(TAW_QUANT_FP8, r, 3, o) == TAW_E_FORMAT);
	r[3] = 0;
	/* NVFP4 block scale: negative or NaN */
	r[2] = 0xB8;
	CHECK(taw_quant_row_decode(TAW_QUANT_NVFP4, r, 3, o) == TAW_E_FORMAT);
	r[2] = 0x7F;
	CHECK(taw_quant_row_decode(TAW_QUANT_NVFP4, r, 3, o) == TAW_E_FORMAT);
	r[2] = 0x38;
	r[4] = 0x30; /* len 3: high nibble of the second code byte */
	CHECK(taw_quant_row_decode(TAW_QUANT_NVFP4, r, 3, o) == TAW_E_FORMAT);
	/* IQ8: every code is valid; Q8: -128 is not */
	memset(r + 2, 0x80, 3);
	CHECK(taw_quant_row_decode(TAW_QUANT_IQ8, r, 3, o) == TAW_OK);
	CHECK(taw_quant_row_decode(TAW_QUANT_Q8, r, 3, o) == TAW_E_FORMAT);
	/* reserved / unknown */
	CHECK(taw_quant_row_decode(TAW_QUANT_BF16, r, 3, o) == TAW_E_UNSUPPORTED);
	CHECK(taw_quant_row_decode(TAW_QUANT_COUNT, r, 3, o) == TAW_E_UNSUPPORTED);
}

/* The quantizer: what it writes decodes, and its error stays within what
 * the format's resolution allows (relative RMS error per format). */
static void quantizer(void)
{
	static const struct {
		uint8_t q;
		double max_rel; /* RMS error / RMS weight, Gaussian-ish rows */
	} lim[] = {
		{TAW_QUANT_F16, 0.001}, {TAW_QUANT_Q8, 0.02},
		{TAW_QUANT_IQ8, 0.03}, {TAW_QUANT_FP8, 0.05},
		{TAW_QUANT_Q4, 0.2}, {TAW_QUANT_IQ4, 0.2},
		{TAW_QUANT_NVFP4, 0.2}, {TAW_QUANT_Q2, 0.6},
		{TAW_QUANT_IQ2, 0.6},
	};
	static const unsigned lens[] = {1, 3, 9, 17, 26, 33};
	uint64_t r = 77;
	for (size_t k = 0; k < sizeof lim / sizeof *lim; k++) {
		double err = 0.0, tot = 0.0;
		for (int t = 0; t < 400; t++) {
			const unsigned len = lens[t % 6];
			float w[64], o[64];
			uint8_t row[160];
			const float mag = t % 7 == 0 ? 3e-5f : t % 5 == 0 ? 40.0f : 0.7f;
			for (unsigned i = 0; i < len; i++) {
				/* sum of uniforms: roughly Gaussian */
				float s = 0.0f;
				for (int u = 0; u < 4; u++)
					s += (float)(int32_t)(uint32_t)(td_next(&r) >> 32) *
					    (1.0f / 2147483648.0f);
				w[i] = s * mag;
			}
			if (t % 11 == 0)
				memset(w, 0, len * sizeof *w);
			memset(row, 0xAA, sizeof row);
			taw_quant_row_encode(lim[k].q, w, len, row);
			const size_t n = (size_t)taw_quant_row_bytes(lim[k].q, len);
			CHECK(row[n] == 0xAA); /* wrote exactly n bytes */
			CHECK(taw_quant_row_decode(lim[k].q, row, len, o) == TAW_OK);
			double e = 0.0, s = 0.0;
			for (unsigned i = 0; i < len; i++) {
				e += ((double)w[i] - o[i]) * ((double)w[i] - o[i]);
				s += (double)w[i] * w[i];
			}
			if (s == 0.0)
				CHECK(e == 0.0);
			else if (len >= 9 && mag > 1e-3f) {
				err += e;
				tot += s;
			}
		}
		double rel = tot > 0 ? err / tot : 0.0;
		/* compare squared ratios to avoid sqrt */
		if (!(rel <= lim[k].max_rel * lim[k].max_rel)) {
			fprintf(stderr, "quant %u: relative MSE %g\n", lim[k].q, rel);
			check_failures++;
		}
	}
}

/* Whole archives in every weight format. */
static void archives(void)
{
	enum { N = 96000 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 5);
	for (int w = TAW_WEIGHTS_F16; w <= TAW_WEIGHTS_IQ2; w++) {
		const taw_params p = {.level = 1, .predictor = TAW_PREDICTOR_MIX,
			.weights = (taw_weights)w};
		void *arc, *dec = nullptr;
		size_t n, dn;
		REQUIRE(taw_compress(src, N, &p, &arc, &n) == TAW_OK);
		taw_info info;
		REQUIRE(taw_read_info(arc, n, &info) == TAW_OK);
		CHECK(info.quant == taw_quant_from_weights((taw_weights)w));
		CHECK(taw_decompress(arc, n, &dec, &dn) == TAW_OK);
		CHECK(dn == N && dec && memcmp(dec, src, N) == 0);
		free(arc);
		free(dec);
	}
	free(src);
}

int main(void)
{
	minifloats();
	decode_values();
	validation();
	quantizer();
	archives();
	return CHECK_DONE();
}
