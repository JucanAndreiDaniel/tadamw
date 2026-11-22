/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "check.h"
#include "taw_internal.h"

int main(void)
{
	/* Standard check value for CRC-32C. */
	CHECK(taw_crc32c(0, "123456789", 9) == 0xE3069283u);
	CHECK(taw_crc32c(0, "", 0) == 0);

	/* Incremental == one-shot. */
	uint32_t c = taw_crc32c(0, "1234", 4);
	c = taw_crc32c(c, "56789", 5);
	CHECK(c == 0xE3069283u);

	/* RFC 3720 B.4: 32 bytes of zeros. */
	uint8_t z[32] = {0};
	CHECK(taw_crc32c(0, z, sizeof z) == 0x8A9136AAu);
	return CHECK_DONE();
}
