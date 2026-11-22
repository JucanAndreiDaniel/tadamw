/* SPDX-License-Identifier: GPL-3.0-or-later */
/* CRC-32C: known values, and every implementation (slicing-by-8 table,
 * SSE4.2) against a bitwise reference over many lengths and alignments. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "taw_internal.h"
#include "testdata.h"

static uint32_t crc_bitwise(uint32_t crc, const uint8_t *p, size_t n)
{
	uint32_t c = ~crc;
	while (n--) {
		c ^= *p++;
		for (int k = 0; k < 8; k++)
			c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1)));
	}
	return ~c;
}

int main(void)
{
	/* Standard check value for CRC-32C. */
	CHECK(taw_crc32c(0, "123456789", 9) == 0xE3069283u);
	CHECK(taw_crc32c(0, "", 0) == 0);

	/* Incremental == one-shot. */
	uint32_t c = taw_crc32c(0, "1234", 4);
	c = taw_crc32c(c, "56789", 5);
	CHECK(c == 0xE3069283u);

	/* RFC 3720 B.4 */
	uint8_t v[32];
	memset(v, 0, sizeof v);
	CHECK(taw_crc32c(0, v, 32) == 0x8A9136AAu);
	memset(v, 0xFF, sizeof v);
	CHECK(taw_crc32c(0, v, 32) == 0x62A8AB43u);
	for (int i = 0; i < 32; i++)
		v[i] = (uint8_t)i;
	CHECK(taw_crc32c(0, v, 32) == 0x46DD794Eu);
	for (int i = 0; i < 32; i++)
		v[i] = (uint8_t)(31 - i);
	CHECK(taw_crc32c(0, v, 32) == 0x113FDB5Cu);

	/* every implementation, every alignment, lengths 0..300 and a long one */
	enum { BUF = 1 << 16 };
	uint8_t *b = malloc(BUF + 16);
	REQUIRE(b);
	td_fill(b, BUF + 16, TD_RANDOM, 5);
	bool sse42 = false;
#if defined(__x86_64__)
	sse42 = __builtin_cpu_supports("sse4.2");
#endif
	int bad = 0;
	for (size_t align = 0; align < 8; align++)
		for (size_t n = 0; n <= 300; n++) {
			uint32_t ref = crc_bitwise(0x12345678u, b + align, n);
			bad += taw_crc32c(0x12345678u, b + align, n) != ref;
			bad += ~taw_crc32c_table(~0x12345678u, b + align, n) != ref;
#if defined(__x86_64__)
			if (sse42)
				bad += ~taw_crc32c_sse42(~0x12345678u, b + align, n) !=
				    ref;
#endif
		}
	uint32_t ref = crc_bitwise(0, b + 3, BUF);
	bad += taw_crc32c(0, b + 3, BUF) != ref;
	bad += ~taw_crc32c_table(~0u, b + 3, BUF) != ref;
	CHECK(bad == 0);
	free(b);
	return CHECK_DONE();
}
