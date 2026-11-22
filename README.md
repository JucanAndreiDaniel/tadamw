# tar_aw

A general-purpose compressed archive format whose byte predictor is a small
model fitted to the input with AdamW and embedded in the archive. Two
variants:

- `.aw`: a single compressed stream
- `.awseek`: fixed-size blocks plus a seek table for random access

Status: scaffold. The container, range coder and seeking work, but the
predictor is still an integer order-0 placeholder (arith profile 0). The
AdamW mixer comes next. See `docs/`.

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
taw c [-B block_size] in out.awseek    # seekable (or -s)
taw d archive out
taw x archive.awseek OFFSET LENGTH out # random-access range
taw i archive                          # header info
```

Requires a C23 compiler (clang >= 18 or gcc >= 14) and CMake >= 3.25.
