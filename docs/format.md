# tar_aw container format, version 1

Status: draft. Normative for everything implemented; reserved values are
listed so that decoders can refuse them.

All integers are unsigned and little-endian unless stated otherwise.
`crc32c` is CRC-32C (Castagnoli, reflected polynomial `0x82F63B78`, init
`0xFFFFFFFF`, final xor `0xFFFFFFFF`).

Two variants share one header and one model section:

| extension | kind | purpose |
|-----------|------|---------|
| `.aw`     | 0    | stream: chunks are decoded strictly in order |
| `.awseek` | 1    | seekable: independent fixed-size blocks + seek index |

## 1. File header (32 bytes)

| off | size | field          | notes |
|-----|------|----------------|-------|
| 0   | 4    | magic          | `54 41 57 1A` ("TAW\x1A") |
| 4   | 1    | version        | `1` |
| 5   | 1    | kind           | `0` = aw, `1` = awseek |
| 6   | 1    | arith_profile  | §2 |
| 7   | 1    | quant          | §3 |
| 8   | 1    | level          | encoder effort 1..9, informational |
| 9   | 1    | flags          | must be `0` |
| 10  | 2    | reserved       | must be `0` |
| 12  | 4    | block_size     | awseek: raw bytes per block; aw: `0` |
| 16  | 8    | model_size     | bytes in the model section |
| 24  | 4    | model_crc      | crc32c of the model section |
| 28  | 4    | header_crc     | crc32c of bytes 0..27 |

A decoder MUST refuse a file if any of these hold: wrong magic, unknown
version or kind, non-zero flags or reserved, bad `header_crc` or
`model_crc`, or an unsupported `(arith_profile, quant)` pair. Decoders never
fall back to a different arithmetic: an archive decodes exactly or not at all.

For awseek, `block_size` MUST be a power of two in `[2^12, 2^30]`.

## 2. Arithmetic profiles

The profile fixes how the decoder turns model weights and history into
16-bit bit probabilities. It covers the inference arithmetic, rounding,
and the squash function. Two implementations of one profile MUST produce
identical probabilities for every bit.

| id | name            | model section | status |
|----|-----------------|---------------|--------|
| 0  | `INT_ORDER0`    | empty (`model_size = 0`, `quant = 0`) | implemented, placeholder |
| 1  | `F32_STRICT_MIX`| bitwise mixer weights, §5 | reserved, not yet specified |
| 2..255 | — | — | reserved, refuse |

### 2.1 Profile 0: `INT_ORDER0`

An integer adaptive order-0 bit model. It exists so that the container,
coder and seeking can be tested before the AdamW model lands.

- State: `uint16 p[256]`, every entry `32768` at the start of each coding
  unit (the whole stream for aw, each block for awseek).
- Each byte is coded MSB first as 8 binary decisions. The context is
  `c0`, which starts at `1` and becomes `c0 = 2*c0 + bit` after each bit.
- The probability that bit = 1 is `p[c0]`.
- After coding `bit`: if `bit == 1`, `p += (65536 - p) >> 4`; otherwise
  `p -= p >> 4`. (`p` stays inside `[15, 65521]`.)

## 3. Quantization formats

The format in which model weights are stored. Encoders may pick any format
the profile allows. F16 is the standard format.

| id | name | encoding |
|----|------|----------|
| 0  | `NONE` | no weights |
| 1  | `F16`  | IEEE 754 binary16 |
| 2  | `BF16` | bfloat16 (upper 16 bits of binary32) |
| 3  | `F32`  | IEEE 754 binary32 |
| 4  | `Q8`   | int8 rows with one F16 scale per row (layout TBD with profile 1) |
| 5..255 | — | reserved, refuse |

## 4. Binary range coder

Every coding unit (an aw chunk or an awseek block) is one independently
flushed range-coder segment. The coder is normative:

```
encoder state: x1 = 0, x2 = 0xFFFFFFFF            (uint32)
encode(bit, p):          p = P(bit = 1) in [1, 65535]
    xmid = x1 + (uint32)(((uint64)(x2 - x1) * p) >> 16)
    if bit: x2 = xmid  else: x1 = xmid + 1
    while ((x1 ^ x2) & 0xFF000000) == 0:
        emit(x2 >> 24); x1 <<= 8; x2 = (x2 << 8) | 0xFF
flush:  emit the 4 bytes of x1, most significant first

decoder state: x1, x2 as above; x = first 4 bytes, big-endian
decode(p):
    xmid as above; bit = (x <= xmid)
    update x1/x2 exactly as the encoder
    while ((x1 ^ x2) & 0xFF000000) == 0:
        x1 <<= 8; x2 = (x2 << 8) | 0xFF; x = (x << 8) | next_byte()
```

Reading past the end of a segment yields `0` bytes. A probability outside
`[1, 65535]` is a bug in the profile, not a property of the data.

## 5. Model section

`model_size` bytes immediately follow the header. Their layout belongs to
the arithmetic profile (profile 0: empty). The model is fitted once over the
whole input and is shared by every chunk or block.

Constraint: `model_size <= floor(raw_size / 32)`. Encoders that cannot
beat profile 0 inside that budget use profile 0.

## 6. aw body (kind 0)

A sequence of chunks follows the model section:

```
chunk:  u32 raw_len   u32 coded_len   u32 raw_crc   coded[coded_len]
end:    u32 0         u32 0           u32 stream_crc
trailer: u64 total_raw_len
```

- `raw_crc` is crc32c of the chunk's decoded bytes, and `stream_crc` is
  crc32c of the whole decoded stream.
- `1 <= raw_len <= 2^26`.
- Chunks only frame the data. The predictor state carries across chunks,
  and only the range coder is restarted.
- The decoded size MUST equal `total_raw_len`.

## 7. awseek body (kind 1)

```
header | model | block[0] ... block[n-1] | index | footer
```

Block `i` holds raw bytes `[i*block_size, min((i+1)*block_size, total))`.
Each block is one range-coder segment, and the predictor state is reset at
the start of each block. That is what makes random access possible.

Index:

| size | field |
|------|-------|
| 4    | magic `54 41 57 49` ("TAWI") |
| 4    | block_count |
| 8    | total_raw_len |
| 16 × block_count | entries: `u64 coded_offset` (absolute), `u32 coded_len`, `u32 raw_crc` |

`block_count` MUST equal `ceil(total_raw_len / block_size)`.

Footer (the last 16 bytes of the file):

| size | field |
|------|-------|
| 8    | index_offset (absolute) |
| 4    | index_crc (crc32c of the whole index) |
| 4    | magic `54 41 57 45` ("TAWE") |

A random-access read of `[off, off+len)` decodes only the blocks that
overlap it.
