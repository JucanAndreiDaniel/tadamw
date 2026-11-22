/* SPDX-License-Identifier: GPL-3.0-or-later */
/* taw-nx's crypto without real dumps or keys: random keys, a synthetic NSP
 * holding one NCA3 with an AES-CTR, an AES-XTS and an unencrypted section.
 *
 * The key property is checked against hactool's own decryptor: the NCA that
 * snx_hac_encrypt makes from a plain one must decrypt (snx_hac_plain) back to
 * exactly that plain NCA. The files it leaves in the work directory (an
 * encrypted NSP and its keys) then go through taw-nx in cli_nx.cmake.
 *
 * usage: test_nx <work dir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "nca.h"
#include "snx_hac.h"
#include "testdata.h"

enum {
	SEC_LEN = 0x20000,          /* bytes per section */
	NCA_LEN = 0xC00 + 3 * SEC_LEN,
	PFS_HEAD = 0x10 + 0x18 + 0x10, /* one entry, string table 0x10 */
};

static uint64_t rs = 2026;

static void fill_random(uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (uint8_t)td_next(&rs);
}

static void hex(char *out, const uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		sprintf(out + 2 * i, "%02x", p[i]);
}

/* A plain NCA3: PFS0-type sections at 0xC00 (CTR), 0x20C00 (XTS) and
 * 0x40C00 (none), each filled with compressible text. */
static void build_plain_nca(uint8_t *nca)
{
	memset(nca, 0, NCA_LEN);
	nca_header_t *h = (nca_header_t *)nca;
	h->magic = MAGIC_NCA3;
	h->distribution = 0;
	h->content_type = 0;
	h->crypto_type = 0; /* master key 0 */
	h->crypto_type2 = 0;
	h->kaek_ind = 0;    /* application */
	h->nca_size = NCA_LEN;
	h->title_id = 0x0100000000ABC000ull;
	fill_random(&h->encrypted_keys[0][0], sizeof h->encrypted_keys);
	static const uint8_t crypt[3] = {CRYPT_CTR, CRYPT_XTS, CRYPT_NONE};
	for (unsigned i = 0; i < 3; i++) {
		uint64_t start = 0xC00 + (uint64_t)i * SEC_LEN;
		h->section_entries[i].media_start_offset = (uint32_t)(start / 0x200);
		h->section_entries[i].media_end_offset =
		    (uint32_t)((start + SEC_LEN) / 0x200);
		nca_fs_header_t *fs = &h->fs_headers[i];
		fs->_0x0 = 2; /* FS header version */
		fs->partition_type = PARTITION_PFS0;
		fs->fs_type = FS_TYPE_PFS0;
		fs->crypt_type = crypt[i];
		fill_random(fs->section_ctr, sizeof fs->section_ctr);
		td_fill(nca + start, SEC_LEN, TD_TEXT, 10 + i);
	}
	/* the fields above are in host order; the file is little-endian (a
	 * no-op on little-endian hosts) */
	nca_header_swap(h);
}

typedef struct {
	uint8_t *p;
	size_t n, cap;
} membuf;

static int mem_write(const void *bytes, size_t n, void *data)
{
	membuf *m = data;
	if (m->n + n > m->cap)
		return 1;
	memcpy(m->p + m->n, bytes, n);
	m->n += n;
	return 0;
}

static FILE *file_of(const char *path, const uint8_t *p, size_t n)
{
	FILE *f = fopen(path, "w+b");
	REQUIRE(f && fwrite(p, 1, n, f) == n && fflush(f) == 0);
	return f;
}

int main(int argc, char **argv)
{
	REQUIRE(argc == 2);
	char path[4096];

	/* keys: random, in a prod.keys for taw-nx too */
	uint8_t header_key[32], kak[16];
	fill_random(header_key, sizeof header_key);
	fill_random(kak, sizeof kak);
	char hk[65], ka[33];
	hex(hk, header_key, 32);
	hex(ka, kak, 16);
	SnxHac *hac = snx_hac_new();
	REQUIRE(hac);
	CHECK(snx_hac_set_key(hac, "header_key", hk) == 1);
	CHECK(snx_hac_set_key(hac, "key_area_key_application_00", ka) == 1);
	/* XCI gamecard info: decrypt is the inverse of hactool's encrypt */
	{
		uint8_t xk[16], iv[16], info[0x70], enc[0x70], back[0x70];
		char xh[33];
		fill_random(xk, sizeof xk);
		fill_random(iv, sizeof iv);
		fill_random(info, sizeof info);
		CHECK(snx_hac_xci_info(hac, iv, info, enc) == -1); /* no key yet */
		hex(xh, xk, 16);
		CHECK(snx_hac_set_key(hac, "xci_header_key", xh) == 1);
		CHECK(snx_hac_xci_info(hac, iv, info, enc) == 0);
		CHECK(memcmp(enc, info, sizeof info) != 0);
		CHECK(snx_hac_xci_info_decrypt(hac, iv, enc, back) == 0);
		CHECK(memcmp(back, info, sizeof info) == 0);
	}
	snx_hac_derive_keys(hac);
	snprintf(path, sizeof path, "%s/prod.keys", argv[1]);
	FILE *kf = fopen(path, "w");
	REQUIRE(kf);
	fprintf(kf, "; test keys\nheader_key = %s\nkey_area_key_application_00 = %s\n",
	    hk, ka);
	fclose(kf);

	/* plain NCA -> encrypted (ours) -> plain (hactool's) */
	uint8_t *plain = malloc(NCA_LEN), *enc = malloc(NCA_LEN);
	REQUIRE(plain && enc);
	build_plain_nca(plain);
	snprintf(path, sizeof path, "%s/plain.nca", argv[1]);
	FILE *pf = file_of(path, plain, NCA_LEN);
	membuf m = {.p = enc, .cap = NCA_LEN};
	CHECK(snx_hac_encrypt(hac, pf, 0, NCA_LEN, mem_write, &m) == 0);
	CHECK(m.n == NCA_LEN);
	fclose(pf);
	/* the encrypted sections must differ from the plain ones, except the
	 * unencrypted third */
	CHECK(memcmp(enc, plain, 0xC00) != 0);
	CHECK(memcmp(enc + 0xC00, plain + 0xC00, SEC_LEN) != 0);
	CHECK(memcmp(enc + 0xC00 + SEC_LEN, plain + 0xC00 + SEC_LEN, SEC_LEN) != 0);
	CHECK(memcmp(enc + 0xC00 + 2 * SEC_LEN, plain + 0xC00 + 2 * SEC_LEN,
	    SEC_LEN) == 0);

	snprintf(path, sizeof path, "%s/enc.nca", argv[1]);
	FILE *ef = file_of(path, enc, NCA_LEN);
	uint8_t *back = malloc(NCA_LEN);
	REQUIRE(back);
	membuf b = {.p = back, .cap = NCA_LEN};
	int was_plain = -1;
	CHECK(snx_hac_plain(hac, ef, 0, NCA_LEN, nullptr, mem_write, &b,
	    &was_plain) == 0);
	CHECK(was_plain == 0);
	CHECK(b.n == NCA_LEN && memcmp(back, plain, NCA_LEN) == 0);
	fclose(ef);

	/* the encrypted NCA in an NSP, for taw-nx (cli_nx.cmake) */
	uint8_t head[PFS_HEAD] = {0};
	memcpy(head, "PFS0", 4);
	head[4] = 1;    /* files */
	head[8] = 0x10; /* string table size */
	/* entry: offset 0, size NCA_LEN, name at 0 */
	for (int i = 0; i < 8; i++)
		head[0x10 + 8 + i] = (uint8_t)((uint64_t)NCA_LEN >> (8 * i));
	memcpy(head + 0x10 + 0x18, "test.nca", 8);
	snprintf(path, sizeof path, "%s/test.nsp", argv[1]);
	FILE *nf = fopen(path, "wb");
	REQUIRE(nf);
	CHECK(fwrite(head, 1, sizeof head, nf) == sizeof head);
	CHECK(fwrite(enc, 1, NCA_LEN, nf) == NCA_LEN);
	fclose(nf);

	snx_hac_free(hac);
	free(plain);
	free(enc);
	free(back);
	return CHECK_DONE();
}
