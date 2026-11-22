# taw-nx: XCI/NSP transcoder

`taw-nx` turns Nintendo Switch dumps (XCI gamecard images and NSP packages)
into ordinary `.awseek` archives and back. Decoding gives back the original
dump byte for byte, checked against its SHA-256.

```sh
taw-nx enc [-l 1..9] [-j threads] [-B block_MiB] [-k prod.keys] game.xci game.awseek
taw-nx dec [-j threads] [-k prod.keys] game.awseek game.xci
taw i game.awseek                  # lists /game.xci and /.meta
taw x game.awseek /.meta -         # prints the metadata
```

Both directions read the keys from `~/.switch/prod.keys` (or `-k`). Title keys
come from the dump's own common tickets. No key is ever written into an
archive or printed.

## The archive

No container of its own: a `.awseek` (format.md §7) with a file directory
(§7.1) holding two files.

| file | contents |
|------|----------|
| `/<dump name>` | the dump with everything taw-nx could decrypt in plain form |
| `/.meta` | text: what decoding needs besides the image |

`/.meta`, one `key value` per line, the first line `tadamw-nx 1`:

| key | value |
|-----|-------|
| `kind` | `nsp` or `xci` |
| `name` | the dump's file name (informational) |
| `size` | the dump's size in bytes |
| `sha256` | SHA-256 of the original dump, hex |
| `nca` | `offset size` of one NCA stored plain; one line per NCA |
| `cardinfo` | an XCI's gamecard info (0x70 bytes at 0x190), decrypted, hex |

Readers ignore keys they do not know.

## Why decrypt

NCAs are AES-encrypted, and ciphertext does not compress (xz -6 leaves an
18.5 MB NSP at 18.5 MB). So nothing that can be decrypted is stored encrypted:

- Every NCA becomes a **plain NCA** (hactool's `snx_hac_plain`, as ScarletNX
  uses it): header and sections decrypted where they lie, every other byte as
  the dump has it.
- An XCI's gamecard info is decrypted (AES-128-CBC under `xci_header_key`,
  IV from the header) into `/.meta`, and its 0x70 bytes in the image are
  zeroed. Without `xci_header_key` it stays as it is.

Decoding encrypts all of it again. `snx_hac_encrypt`
(`tools/hactool/snx_hac.c`) is the inverse added for tadamw:

- It XTS-encrypts the header under `header_key` the way hactool decrypts it.
- It lets hactool derive the section keys from that header as usual.
- It reads each section's raw storage over the plain bytes, which yields them
  encrypted. AES-CTR and AES-CTR-EX (an update's BKTR sections) are their own
  inverses; the update's bucket tree loads from its plain tables. XTS sections
  go through an encrypting XTS storage.

The encoder relies on nothing it has not checked: every NCA, and the gamecard
info, must encrypt back to the dump's exact bytes, or it stays as the dump has
it. That covers missing keys, personalized tickets, NCA0, sparse sections, and
an NCA2 whose unused FS headers decrypt to zeros.

Only the NCAs listed by the container are handled: an NSP's files, or the files
of an XCI's secure partition. The XCI's other partitions (update, normal, logo)
stay as they are; they are small next to the game.

## Compression

`taw_seek_encode` (libtadamw) does the work. One model is fitted for the whole
dump, sampled evenly across it. The blocks (64 MiB by default) are coded on
all threads with bounded memory, and blocks that do not shrink (already
compressed assets) are stored. The archive is the same for any thread count.
Decoding opens one `taw_seek` per thread and expands blocks in parallel. The
default level is 1, as dumps are large.

Measured on an i5-9400 (6 cores), each decoded byte-identical to the dump:

| dump | size | NCAs decrypted | archive | enc | dec |
|------|------|----------------|---------|-----|-----|
| small NSP, 4 MiB blocks | 18.5 MB | 4 of 4 | 72.2% | 17.6 s | 13.7 s |

Earlier runs, with a container of separate per-segment models (the update
NSP's BKTR sections and an XCI's 32 NCAs), also round-tripped exactly; see
the git history.

## Building

`taw-nx` builds when CMake finds OpenSSL 3 (libcrypto) and LZ4, which the
hactool copy needs (`-DTADAMW_NX=OFF` skips it). Its `enc` command needs the
encoder half of libtadamw and `dec` the decoder half. The tests `nx_crypto`
and `cli_nx` use random keys and a synthetic NSP with CTR, XTS and unencrypted
sections. They check that hactool's decryptor inverts `snx_hac_encrypt`
exactly, that the gamecard info round-trips, that `taw-nx` round-trips and
writes an ordinary `.awseek` that `taw` can list and read, and that wrong keys
fail without output. Real dumps are tested by hand.

## License

hactool's copy keeps its licenses (see `tools/hactool/README.md`); ScarletNX's
and tadamw's code is GPL-3.0-or-later, which makes tadamw as a whole
GPL-3.0-or-later (`LICENSE`).
