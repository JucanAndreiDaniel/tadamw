/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdlib.h>
#include <string.h>

#include "taw_internal.h"

static bool buf_reserve(taw_buf *b, size_t extra)
{
	if (b->oom)
		return false;
	if (extra <= b->cap - b->n)
		return true;
	size_t cap = b->cap ? b->cap : 4096;
	while (cap - b->n < extra) {
		if (cap > SIZE_MAX / 2) {
			b->oom = true;
			return false;
		}
		cap *= 2;
	}
	uint8_t *p = realloc(b->p, cap);
	if (!p) {
		b->oom = true;
		return false;
	}
	b->p = p;
	b->cap = cap;
	return true;
}

void taw_buf_put(taw_buf *b, const void *src, size_t n)
{
	if (n && buf_reserve(b, n)) {
		memcpy(b->p + b->n, src, n);
		b->n += n;
	}
}

uint8_t *taw_buf_extend(taw_buf *b, size_t n)
{
	if (!buf_reserve(b, n))
		return nullptr;
	uint8_t *p = b->p + b->n;
	b->n += n;
	return p;
}

void taw_buf_put_u8(taw_buf *b, uint8_t v)
{
	if (b->n < b->cap) {
		b->p[b->n++] = v;
		return;
	}
	taw_buf_put(b, &v, 1);
}

void taw_buf_put_u32(taw_buf *b, uint32_t v)
{
	uint8_t t[4];
	taw_wr32(t, v);
	taw_buf_put(b, t, 4);
}

void taw_buf_put_u64(taw_buf *b, uint64_t v)
{
	uint8_t t[8];
	taw_wr64(t, v);
	taw_buf_put(b, t, 8);
}

void taw_buf_free(taw_buf *b)
{
	free(b->p);
	*b = (taw_buf){0};
}
