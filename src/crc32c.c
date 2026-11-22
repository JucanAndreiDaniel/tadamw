/* SPDX-License-Identifier: GPL-3.0-or-later */
/* CRC-32C (Castagnoli, reflected 0x82F63B78). The SSE4.2 crc32 instruction
 * when the CPU has it (crc32c_sse42.c), else slicing-by-8 tables: both give
 * the same values, which tests/test_crc32c.c checks. */
#include <stdatomic.h>
#include <string.h>

#include "taw_internal.h"

static uint32_t tab[8][256];
static atomic_int tab_state; /* 0 none, 1 building, 2 ready */

static void tab_build(void)
{
	for (uint32_t i = 0; i < 256; i++) {
		uint32_t c = i;
		for (int k = 0; k < 8; k++)
			c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1)));
		tab[0][i] = c;
	}
	for (uint32_t i = 0; i < 256; i++)
		for (int s = 1; s < 8; s++)
			tab[s][i] = (tab[s - 1][i] >> 8) ^ tab[0][tab[s - 1][i] & 0xFF];
}

static void tab_init(void)
{
	if (atomic_load(&tab_state) == 2)
		return;
	int expect = 0;
	if (atomic_compare_exchange_strong(&tab_state, &expect, 1)) {
		tab_build();
		atomic_store(&tab_state, 2);
		return;
	}
	while (atomic_load(&tab_state) != 2)
		;
}

/* Slicing-by-8 over the raw (inverted) state. */
uint32_t taw_crc32c_table(uint32_t c, const uint8_t *p, size_t n)
{
	tab_init();
	while (n && ((uintptr_t)p & 7)) {
		c = (c >> 8) ^ tab[0][(c ^ *p++) & 0xFF];
		n--;
	}
	while (n >= 8) {
		uint64_t w;
		memcpy(&w, p, 8);
		/* little-endian load; on big-endian hosts swap */
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
		w = __builtin_bswap64(w);
#endif
		uint32_t lo = (uint32_t)w ^ c, hi = (uint32_t)(w >> 32);
		c = tab[7][lo & 0xFF] ^ tab[6][(lo >> 8) & 0xFF] ^
		    tab[5][(lo >> 16) & 0xFF] ^ tab[4][lo >> 24] ^
		    tab[3][hi & 0xFF] ^ tab[2][(hi >> 8) & 0xFF] ^
		    tab[1][(hi >> 16) & 0xFF] ^ tab[0][hi >> 24];
		p += 8;
		n -= 8;
	}
	while (n--)
		c = (c >> 8) ^ tab[0][(c ^ *p++) & 0xFF];
	return c;
}

#if TAW_X86_SIMD
#include <cpuid.h>

static int hw = -1; /* benign race: every thread computes the same */

static bool have_sse42(void)
{
	if (hw < 0) {
		unsigned a, b, c, d;
		hw = __get_cpuid(1, &a, &b, &c, &d) && (c >> 20 & 1);
	}
	return hw;
}
#endif

uint32_t taw_crc32c(uint32_t crc, const void *data, size_t n)
{
	uint32_t c = ~crc;
#if TAW_X86_SIMD
	if (have_sse42())
		return ~taw_crc32c_sse42(c, data, n);
#endif
	return ~taw_crc32c_table(c, data, n);
}
