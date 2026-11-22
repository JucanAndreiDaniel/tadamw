/* CRC-32C (Castagnoli), slicing-by-1 table. */
#include "taw_internal.h"

static uint32_t crc_table[256];
static bool crc_ready;

static void crc_init(void)
{
	for (uint32_t i = 0; i < 256; i++) {
		uint32_t c = i;
		for (int k = 0; k < 8; k++)
			c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
		crc_table[i] = c;
	}
	crc_ready = true;
}

uint32_t taw_crc32c(uint32_t crc, const void *data, size_t n)
{
	/* Table init is idempotent, so a benign race only repeats work. */
	if (!crc_ready)
		crc_init();
	const uint8_t *p = data;
	crc = ~crc;
	while (n--)
		crc = crc_table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
	return ~crc;
}
