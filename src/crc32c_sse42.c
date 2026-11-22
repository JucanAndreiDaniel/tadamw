/* SPDX-License-Identifier: GPL-3.0-or-later */
/* CRC-32C with the SSE4.2 crc32 instruction. Built with -msse4.2; called
 * only after a CPUID check (crc32c.c). Works on the raw (inverted) state. */
#include <nmmintrin.h>
#include <string.h>

#include "taw_internal.h"

uint32_t taw_crc32c_sse42(uint32_t c, const uint8_t *p, size_t n)
{
	uint64_t c64 = c;
	while (n && ((uintptr_t)p & 7)) {
		c64 = _mm_crc32_u8((uint32_t)c64, *p++);
		n--;
	}
	/* three independent streams hide the instruction's latency would need
	 * a combine step; one stream already runs at several GB/s */
	while (n >= 32) {
		uint64_t w[4];
		memcpy(w, p, 32);
		c64 = _mm_crc32_u64(c64, w[0]);
		c64 = _mm_crc32_u64(c64, w[1]);
		c64 = _mm_crc32_u64(c64, w[2]);
		c64 = _mm_crc32_u64(c64, w[3]);
		p += 32;
		n -= 32;
	}
	while (n >= 8) {
		uint64_t w;
		memcpy(&w, p, 8);
		c64 = _mm_crc32_u64(c64, w);
		p += 8;
		n -= 8;
	}
	uint32_t c32 = (uint32_t)c64;
	while (n--)
		c32 = _mm_crc32_u8(c32, *p++);
	return c32;
}
