/* Range coder + profile 0 predictor: roundtrip and sanity of compression. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "taw_internal.h"
#include "testdata.h"

/* Raw coder with extreme and arbitrary probabilities. */
static void test_raw_bits(void)
{
	enum { N = 200000 };
	static uint8_t bits[N];
	static uint16_t probs[N];
	uint64_t s = 42;
	for (size_t i = 0; i < N; i++) {
		uint64_t r = td_next(&s);
		switch (r & 3) {
		case 0: probs[i] = 1; break;
		case 1: probs[i] = 65535; break;
		default: probs[i] = (uint16_t)((r >> 8) % 65535 + 1); break;
		}
		bits[i] = (uint8_t)((r >> 40) & 1);
	}
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	for (size_t i = 0; i < N; i++)
		taw_rc_enc_bit(&e, bits[i], probs[i]);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);

	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	size_t bad = 0;
	for (size_t i = 0; i < N; i++)
		bad += taw_rc_dec_bit(&d, probs[i]) != bits[i];
	CHECK(bad == 0);
	taw_buf_free(&out);
}

static size_t roundtrip(const uint8_t *src, size_t n)
{
	taw_pred pr;
	REQUIRE(taw_pred_init(&pr, TAW_PROFILE_INT_ORDER0, TAW_QUANT_NONE,
	    nullptr, 0) == TAW_OK);
	taw_buf out = {0};
	taw_rc_enc e;
	taw_rc_enc_init(&e, &out);
	taw_pred_encode(&pr, &e, src, n);
	taw_rc_enc_flush(&e);
	REQUIRE(!out.oom);

	uint8_t *dst = malloc(n ? n : 1);
	REQUIRE(dst);
	taw_pred_reset(&pr);
	taw_rc_dec d;
	taw_rc_dec_init(&d, out.p, out.n);
	taw_pred_decode(&pr, &d, dst, n);
	CHECK(memcmp(src, dst, n) == 0);
	size_t coded = out.n;
	free(dst);
	taw_buf_free(&out);
	return coded;
}

int main(void)
{
	test_raw_bits();

	enum { N = 1 << 18 };
	uint8_t *src = malloc(N);
	REQUIRE(src);
	for (int kind = 0; kind < TD_KINDS; kind++) {
		td_fill(src, N, kind, 7);
		size_t c = roundtrip(src, N);
		if (kind == TD_ZERO)
			CHECK(c < N / 100);
		if (kind == TD_RANDOM)
			CHECK(c < N + N * 3 / 100); /* rate-1/16 noise: ~2.4% */
		if (kind == TD_SKEWED)
			CHECK(c < N / 2);
	}
	for (size_t n = 0; n < 64; n++) {
		td_fill(src, n, TD_RANDOM, n);
		roundtrip(src, n);
	}
	CHECK(taw_pred_init(&(taw_pred){0}, TAW_PROFILE_F32_STRICT_MIX,
	    TAW_QUANT_F16, nullptr, 0) == TAW_E_UNSUPPORTED);
	free(src);
	return CHECK_DONE();
}
