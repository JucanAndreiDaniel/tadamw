# tar_aw

A general-purpose compressed archive format whose byte predictor is a small
model fitted to the input with AdamW and embedded in the archive. Two
variants:

- `.aw`: a single compressed stream
- `.awseek`: fixed-size blocks plus a seek table for random access

Status: working draft. The container, range coder, seeking and both
arithmetic profiles work: profile 0 is an integer order-0 placeholder, and
profile 1 (`F32_STRICT_MIX`) is the fitted mixer. Adaptive integer
context models feed a small MLP that the encoder fits with AdamW and
stores as F16 or Q8. The decoder has scalar, AVX2/F16C and AVX-512 paths
that produce identical bits. See `docs/`.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Sanitizer build: `cmake -S . -B build-san -DTAW_SANITIZE=ON -DTAW_WERROR=ON`.

## Usage

```sh
taw c [-l 1..9] in out.aw              # stream
taw c -p mix -w q8 in out.aw           # force the mixer, Q8 weights
taw c [-B block_size] in out.awseek    # seekable (or -s)
taw d archive out
taw x archive.awseek OFFSET LENGTH out # random-access range
taw i archive                          # header info
```

Requires a C23 compiler (clang >= 18 or gcc >= 14) and CMake >= 3.25.

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). `tools/hactool` is a copy of SciresM's
hactool (ISC) as ScarletNX carries it, with ScarletNX's and tadamw's changes
under GPL-3.0-or-later; see `tools/hactool/README.md`.
