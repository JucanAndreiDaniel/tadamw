/* SPDX-License-Identifier: GPL-3.0-or-later */
/* tadamw: hactool's nso.c asks whether an NSO segment is ScarletNX's ZBIC (common/utilities/zbic.h). The
   transcoder never decompresses NSO segments, so this build answers no instead of carrying zstd. */
#include <stddef.h>

#include "common/utilities/zbic.h"

int snx_zbic_is(const void *src, size_t size)
{
	(void)src;
	(void)size;
	return 0;
}

long snx_zbic_decompress(void *dst, size_t dst_size, const void *src,
    size_t src_size)
{
	(void)dst;
	(void)dst_size;
	(void)src;
	(void)src_size;
	return -1;
}
