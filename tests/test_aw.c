/* .aw stream: roundtrip, header fields, and refusal of corrupt input. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "taw_internal.h"
#include "testdata.h"

static void roundtrip(const uint8_t *src, size_t n)
{
	void *arc, *dec;
	size_t arc_n, dec_n;
	REQUIRE(taw_compress(src, n, nullptr, &arc, &arc_n) == TAW_OK);
	taw_info info;
	CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.kind == TAW_KIND_STREAM);
	CHECK(info.level == TAW_LEVEL_DEFAULT);
	CHECK(info.block_size == 0);
	CHECK(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == n);
	CHECK(dec_n == n && memcmp(src, dec, n) == 0);
	free(arc);
	free(dec);
}

/* Patch the header and fix its CRC, to test semantic refusals. */
static taw_status decode_patched(const uint8_t *arc, size_t n, size_t off,
    uint8_t v)
{
	uint8_t *c = malloc(n);
	REQUIRE(c);
	memcpy(c, arc, n);
	c[off] = v;
	taw_wr32(c + 28, taw_crc32c(0, c, 28));
	void *dec = nullptr;
	size_t dec_n;
	taw_status st = taw_decompress(c, n, &dec, &dec_n);
	free(dec);
	free(c);
	return st;
}

static void corruption(void)
{
	enum { N = 3 << 20 }; /* > 1 chunk */
	uint8_t *src = malloc(N);
	REQUIRE(src);
	td_fill(src, N, TD_TEXT, 3);
	void *arc_v;
	size_t n;
	REQUIRE(taw_compress(src, N, nullptr, &arc_v, &n) == TAW_OK);
	uint8_t *arc = arc_v;
	void *dec = nullptr;
	size_t dec_n;

	/* Header byte flip without CRC fix -> checksum. */
	arc[8] ^= 1;
	CHECK(taw_decompress(arc, n, &dec, &dec_n) == TAW_E_CHECKSUM);
	arc[8] ^= 1;

	/* Unknown profile / quant / version / flags -> refused. */
	CHECK(decode_patched(arc, n, 6, TAW_PROFILE_F32_STRICT_MIX) ==
	    TAW_E_UNSUPPORTED);
	CHECK(decode_patched(arc, n, 7, TAW_QUANT_F16) == TAW_E_UNSUPPORTED);
	CHECK(decode_patched(arc, n, 4, 2) == TAW_E_UNSUPPORTED);
	CHECK(decode_patched(arc, n, 9, 1) == TAW_E_UNSUPPORTED);
	CHECK(decode_patched(arc, n, 5, 7) == TAW_E_UNSUPPORTED);

	/* Payload flip -> chunk CRC catches it. */
	arc[n / 2] ^= 0x40;
	CHECK(taw_decompress(arc, n, &dec, &dec_n) == TAW_E_CHECKSUM);
	arc[n / 2] ^= 0x40;

	/* Truncation and trailing garbage -> format. */
	CHECK(taw_decompress(arc, n - 1, &dec, &dec_n) == TAW_E_FORMAT);
	CHECK(taw_decompress(arc, 40, &dec, &dec_n) == TAW_E_FORMAT);
	CHECK(taw_decompress(arc, 10, &dec, &dec_n) == TAW_E_FORMAT);
	uint8_t *longer = malloc(n + 1);
	REQUIRE(longer);
	memcpy(longer, arc, n);
	longer[n] = 0;
	CHECK(taw_decompress(longer, n + 1, &dec, &dec_n) == TAW_E_FORMAT);

	/* Sanity: untouched archive still decodes. */
	CHECK(taw_decompress(arc, n, &dec, &dec_n) == TAW_OK);
	free(dec);
	free(longer);
	free(arc);
	free(src);
}

int main(void)
{
	roundtrip(nullptr, 0);
	uint8_t one = 0xA5;
	roundtrip(&one, 1);

	static const size_t sizes[] = {2, 255, 4096, (1u << 20) - 1, 1u << 20,
		(1u << 20) + 1, (5u << 20) / 2};
	for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
		uint8_t *src = malloc(sizes[i]);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, sizes[i], kind, i * 31 + (size_t)kind);
			roundtrip(src, sizes[i]);
		}
		free(src);
	}

	taw_params bad = {.level = 10};
	void *o;
	size_t on;
	CHECK(taw_compress(&one, 1, &bad, &o, &on) == TAW_E_ARG);

	corruption();
	return CHECK_DONE();
}
