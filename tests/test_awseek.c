/* SPDX-License-Identifier: GPL-3.0-or-later */
/* .awseek: roundtrip, random access, I/O locality, corruption isolation. */
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "format.h"
#include "taw_internal.h"
#include "testdata.h"

/* Memory source that counts the bytes it serves. */
typedef struct {
	const uint8_t *p;
	uint64_t bytes;
} counting;

static taw_status count_read(void *ctx, uint64_t off, void *dst, size_t n)
{
	counting *c = ctx;
	memcpy(dst, c->p + off, n);
	c->bytes += n;
	return TAW_OK;
}

static void *compress_seek_p(const uint8_t *src, size_t n, uint32_t bs,
    taw_predictor pred, size_t *arc_n)
{
	taw_params p = {.seekable = true, .block_size = bs, .predictor = pred,
		.level = pred == TAW_PREDICTOR_MIX ? 1 : 0};
	void *arc;
	REQUIRE(taw_compress(src, n, &p, &arc, arc_n) == TAW_OK);
	return arc;
}

/* Container tests use the fast order-0 predictor. */
static void *compress_seek(const uint8_t *src, size_t n, uint32_t bs,
    size_t *arc_n)
{
	return compress_seek_p(src, n, bs, TAW_PREDICTOR_ORDER0, arc_n);
}

static void roundtrip_p(const uint8_t *src, size_t n, uint32_t bs,
    taw_predictor pred)
{
	size_t arc_n;
	void *arc = compress_seek_p(src, n, bs, pred, &arc_n);
	taw_info info;
	CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.kind == TAW_KIND_SEEKABLE);
	CHECK(info.block_size == (bs ? bs : TAW_BLOCK_DEFAULT));
	void *dec;
	size_t dec_n;
	CHECK(taw_decompress(arc, arc_n, &dec, &dec_n) == TAW_OK);
	CHECK(dec_n == n && memcmp(src, dec, n) == 0);
	free(dec);
	free(arc);
}

static void roundtrip(const uint8_t *src, size_t n, uint32_t bs)
{
	roundtrip_p(src, n, bs, TAW_PREDICTOR_ORDER0);
}

/* Profile 1: one shared model, history reset per block. */
static void random_access_mix(void)
{
	enum { N = 300000, BS = 1 << 15 };
	uint8_t *src = malloc(N), *buf = malloc(2 * BS);
	REQUIRE(src && buf);
	td_fill(src, N, TD_TEXT, 12);
	size_t arc_n;
	void *arc = compress_seek_p(src, N, BS, TAW_PREDICTOR_MIX, &arc_n);
	taw_info info;
	CHECK(taw_read_info(arc, arc_n, &info) == TAW_OK);
	CHECK(info.arith_profile == TAW_PROFILE_F32_STRICT_MIX);
	taw_source mem;
	taw_source_memory(&mem, arc, arc_n);
	taw_seek *s;
	REQUIRE(taw_seek_open(&mem, &s) == TAW_OK);
	uint64_t r = 5;
	for (int t = 0; t < 40; t++) {
		uint64_t off = td_next(&r) % N;
		size_t len = (size_t)(td_next(&r) % (2 * BS));
		if (len > N - off)
			len = (size_t)(N - off);
		REQUIRE(taw_seek_read(s, off, buf, len) == TAW_OK);
		CHECK(memcmp(buf, src + off, len) == 0);
	}
	taw_seek_close(s);
	free(arc);
	free(buf);
	free(src);
}

static void random_access(void)
{
	enum { N = (1 << 20) + 12345, BS = 4096 };
	uint8_t *src = malloc(N), *buf = malloc(3 * BS);
	REQUIRE(src && buf);
	td_fill(src, N, TD_TEXT, 11);
	size_t arc_n;
	void *arc = compress_seek(src, N, BS, &arc_n);

	counting cnt = {.p = arc};
	taw_source in = {.read_at = count_read, .ctx = &cnt, .size = arc_n};
	taw_seek *s;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	CHECK(taw_seek_size(s) == N);
	uint64_t open_bytes = cnt.bytes;

	/* Random ranges, including block-straddling ones. */
	uint64_t r = 99;
	for (int t = 0; t < 2000; t++) {
		uint64_t off = td_next(&r) % N;
		size_t len = (size_t)(td_next(&r) % (3 * BS));
		if (len > N - off)
			len = (size_t)(N - off);
		REQUIRE(taw_seek_read(s, off, buf, len) == TAW_OK);
		CHECK(memcmp(buf, src + off, len) == 0);
	}

	/* One byte from the middle reads at most one coded block. */
	taw_seek_close(s);
	cnt.bytes = 0;
	REQUIRE(taw_seek_open(&in, &s) == TAW_OK);
	open_bytes = cnt.bytes;
	CHECK(taw_seek_read(s, N / 2, buf, 1) == TAW_OK);
	CHECK(buf[0] == src[N / 2]);
	CHECK(cnt.bytes - open_bytes <= BS);
	CHECK(cnt.bytes < arc_n / 4);

	/* Bounds. */
	CHECK(taw_seek_read(s, N, buf, 0) == TAW_OK);
	CHECK(taw_seek_read(s, N, buf, 1) == TAW_E_RANGE);
	CHECK(taw_seek_read(s, N - 1, buf, 2) == TAW_E_RANGE);
	CHECK(taw_seek_read(s, UINT64_MAX, buf, 1) == TAW_E_RANGE);
	taw_seek_close(s);

	/* Corrupting block 3 only breaks reads that touch block 3. */
	uint8_t *a = arc;
	uint64_t ix = taw_rd64(a + arc_n - 16);
	uint64_t b3 = taw_rd64(a + ix + 16 + 3 * 16);
	a[b3 + 2] ^= 0x10;
	taw_source mem;
	taw_source_memory(&mem, arc, arc_n);
	REQUIRE(taw_seek_open(&mem, &s) == TAW_OK);
	CHECK(taw_seek_read(s, 3 * BS + 100, buf, 10) == TAW_E_CHECKSUM);
	CHECK(taw_seek_read(s, 5 * BS, buf, BS) == TAW_OK);
	CHECK(memcmp(buf, src + 5 * BS, BS) == 0);
	CHECK(taw_seek_read(s, 2 * BS, buf, 2 * BS) == TAW_E_CHECKSUM);
	taw_seek_close(s);
	a[b3 + 2] ^= 0x10;

	/* Index, footer and truncation damage is caught at open. */
	a[ix + 20] ^= 1;
	CHECK(taw_seek_open(&mem, &s) == TAW_E_CHECKSUM);
	a[ix + 20] ^= 1;
	a[arc_n - 1] ^= 1;
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);
	a[arc_n - 1] ^= 1;
	taw_source_memory(&mem, arc, arc_n - 1);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);
	taw_source_memory(&mem, arc, 20);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_FORMAT);

	/* A stream archive is not seekable. */
	void *aw;
	size_t aw_n;
	REQUIRE(taw_compress(src, 1000, &(taw_params){.predictor =
	    TAW_PREDICTOR_ORDER0}, &aw, &aw_n) == TAW_OK);
	taw_source_memory(&mem, aw, aw_n);
	CHECK(taw_seek_open(&mem, &s) == TAW_E_ARG);

	free(aw);
	free(arc);
	free(buf);
	free(src);
}

int main(void)
{
	roundtrip(nullptr, 0, 4096);
	uint8_t one = 7;
	roundtrip(&one, 1, 4096);

	static const size_t sizes[] = {4095, 4096, 4097, 3 * 4096, 100000};
	for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
		uint8_t *src = malloc(sizes[i]);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, sizes[i], kind, i + 17 * (size_t)kind);
			roundtrip(src, sizes[i], 4096);
			roundtrip(src, sizes[i], 1u << 16);
		}
		free(src);
	}
	uint8_t big[5000];
	td_fill(big, sizeof big, TD_SKEWED, 5);
	roundtrip(big, sizeof big, 0); /* default block size */

	taw_params bad = {.seekable = true, .block_size = 5000};
	void *o;
	size_t on;
	CHECK(taw_compress(big, 10, &bad, &o, &on) == TAW_E_ARG);
	bad.block_size = 2048;
	CHECK(taw_compress(big, 10, &bad, &o, &on) == TAW_E_ARG);

	/* Profile 1 roundtrips, including a partial last block. */
	{
		enum { N = 100000 };
		uint8_t *src = malloc(N);
		REQUIRE(src);
		for (int kind = 0; kind < TD_KINDS; kind++) {
			td_fill(src, N, kind, 40 + (size_t)kind);
			roundtrip_p(src, N, 1u << 14, TAW_PREDICTOR_MIX);
		}
		td_fill(src, N, TD_TEXT, 3);
		roundtrip_p(src, N, 4096, TAW_PREDICTOR_AUTO);
		free(src);
	}

	random_access();
	random_access_mix();
	return CHECK_DONE();
}
