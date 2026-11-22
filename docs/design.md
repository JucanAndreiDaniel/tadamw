# tar_aw design

## Goal

A general-purpose compressor in the spirit of zstd/xz, with a different
predictor. The encoder fits a small model to the input it is compressing,
using AdamW on the bitwise cross-entropy, and embeds the frozen weights
in the archive. The decoder only runs inference. The bit probabilities
drive a binary range coder (format.md §4).

## Decisions so far

| topic | decision |
|-------|----------|
| fitting | offline, encoder only; the decoder uses frozen weights |
| predictor | bitwise mixer: hashed order-N context features → small MLP → P(bit=1) |
| model budget | `model_size <= raw_size / 32`; a heuristic picks the size |
| level | effort only (epochs, number of candidate sizes), not a size knob |
| weights | stored in a tagged quant format; F16 is standard |
| arithmetic | tagged arithmetic profile; decoders refuse unknown profiles/quants |
| variants | `.aw` stream, `.awseek` seekable with one global model |
| language | C23, libc only, CMake + CTest |

## Encoder pipeline

1. Buffer the input. Fitting is offline, so the encoder sees all of it.
2. Pick candidate model sizes up to `floor(n/32)` bytes. The level sets
   how many candidates are tried and how many AdamW epochs each one gets.
3. For each candidate, fit with AdamW in whatever precision is
   convenient. This happens only in the encoder, so it doesn't need to be
   deterministic.
4. Quantize the weights to the chosen quant format, then **dequantize
   them back**. The encoder codes with exactly the weights the decoder
   will see.
5. Estimate the total size (model + coded residual) and keep the best
   candidate. Fall back to profile 0 if nothing beats it.
6. Code the input as chunks (aw) or blocks (awseek) with the profile's
   inference arithmetic.

## Decoder

1. Validate the header and refuse unsupported `(profile, quant)` pairs.
2. Load and dequantize the model once.
3. Decode chunks in order (aw). For awseek, decode only the blocks that
   overlap the requested range, after resetting the history at each block.

## Determinism

Fitting may be sloppy, but inference may not. Every probability must be
bit-identical between the encoder and every decoder. The rules:

- Each arithmetic profile fixes its operation order, accumulator widths,
  rounding and squash function.
- The build disables FP contraction and fast-math
  (`-ffp-contract=off -fno-fast-math`).
- Profiles must not call libm transcendentals (`expf`, `tanhf`, …),
  because their results differ between libcs. Squash and stretch use
  tables or integer formulas defined in the spec.
- The probability handed to the coder is a 16-bit integer, and the
  float-to-integer conversion is defined by the profile.

## Seeking

`.awseek` resets the predictor history at every block but shares the one
model. A seek costs reading the header, model and index, plus decoding
the overlapping blocks. Blocks are independent, so encoding and decoding
can run in parallel later.

## Source layout

| path | role |
|------|------|
| `include/taw/taw.h` | public API |
| `src/crc32c.c` | CRC-32C |
| `src/rc.c` | binary range coder (format.md §4) |
| `src/predictor.c` | arithmetic profiles (profile 0 for now) |
| `src/header.c` | header (de)serialization and validation |
| `src/aw.c` | `.aw` stream encode/decode |
| `src/awseek.c` | `.awseek` encode/decode/random access |
| `tools/taw.c` | CLI |
| `tests/` | CTest executables + CLI roundtrip script |

## Open items

- Specify profile 1 (`F32_STRICT_MIX`): feature hashing, table and MLP
  shapes as functions of the model budget, the squash table, and the
  F16/Q8 weight layout.
- The size heuristic and the level → effort mapping.
- Streaming encode for inputs larger than memory. (Fitting needs a pass
  over the data, so this probably means fitting on a sample.)
- Parallel block coding for awseek.
- A stored (uncoded) mode for incompressible chunks/blocks. Profile 0 grows
  random data by about 2.4%.
