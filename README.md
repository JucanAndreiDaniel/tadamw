# tar_aw

A general-purpose compressed archive format whose byte predictor is a small
model fitted to the input with AdamW and embedded in the archive. Two
variants:

- `.aw`: a single compressed stream
- `.awseek`: fixed-size blocks plus a seek table for random access

Status: scaffold. See `docs/`.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires a C23 compiler (clang >= 18 or gcc >= 14) and CMake >= 3.25.
