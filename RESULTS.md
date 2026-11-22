# Experiment: quantization-aware training (QAT) for the profile-1 mixer

Branch `exp/qat`, rebased onto `origin/main` at `e64cc83` (was `a427e7f`).
`bench_local.sh` reproduces the table. Archive bytes are the primary signal.

> **Stale-numbers note.** The pre-rebase table (text L1 190610 / L5 187956,
> libc L1 388792 / L5 385741) was measured on `a427e7f` with the old prior,
> one-bit-per-byte sampling and only F16/Q8 formats. It is **invalid** for
> this base. The new clean `-p mix` baseline is text L1 181557 / L5 179166,
> libc L1 379197 / L5 375291 (reproduced byte-for-byte, see below).

## What changed (rebase resolution)

Upstream moved the trainable net into `src/fit_internal.h`, all weight formats
into `src/quant.c`/`quant.h` (Q4/Q2/FP8/NVFP4/IQ8/IQ4/IQ2, ids 5–11), added
bit-position sampling, a fused AVX2 decoder, parallel awseek decode and opt-in
Vulkan fitting. The rebase conflict was confined to `src/fit.c`; both upstream
code paths and the QAT code were preserved:

- `tnet` gained `qat`, `quant` and per-row scale arrays in `fit_internal.h`;
  `tnet_init` takes the QAT mode and quantizer id. GPU warm-start nets carry
  QAT scale state so serialization stays self-consistent (the GPU trainer
  itself does not simulate quantization — a caveat).
- `quant.c`/`quant.h` gained encoder-only helpers that **reuse** the upstream
  level tables and F16 scale encoding: `taw_quant_row_scale` (cheap
  amax/max-level scale), `taw_quant_simulate` (dequantized row for the
  training forward pass; O(1) path for Q8/Q4, binary search for the other
  level formats), and `taw_quant_row_encode_with_scale` (write a row with a
  prescribed scale, used by the fixed/EMA scale experiments). Q2/NVFP4 have
  no simple level set and skip QAT. No decoder file, format constant or
  archive layout was changed by this experiment.
- The training forward pass uses the simulated dequantized weights (F16 round
  trip, level-format quantization) with a straight-through estimator; AdamW
  updates the latent float weights. Q8 scale policies: recomputed each step,
  fixed at init, fixed at untie, EMA(0.98). `TAW_QAT = off|on|q8fixed|
  q8fixed2|q8ema|f16only`, default `off`.
- During the experiment the Q8/Q4 simulation was changed from a per-weight
  binary search to an O(1) round-and-clamp (identical except on exact ties)
  because the first implementation made level-5 fits ~14× slower. All final
  `on`-q8, `q8fixed`, `q8fixed2` and `q8ema` numbers below use that final
  build; the FP8 and Q4 `on` numbers were taken before this change (see
  caveats).

## Baseline verification (QAT off)

`TAW_QAT=off QUANTS=auto ./bench_local.sh` reproduced all four supplied
baseline archives byte-for-byte:

| corpus | L | auto baseline (bytes) | model |
|--------|---|----------------------:|------:|
| text1m | 1 | 181557 | 608 |
| text1m | 5 | 179166 | 608 |
| libc1m | 1 | 379197 | 608 |
| libc1m | 5 | 375291 | 608 |

Forced-format baselines (QAT off) differ at L5 because `-w` restricts the
candidate set (the auto winner is not always the best *forced* format):

| corpus | L | `-w q8` | `-w f16` | `-w fp8` | `-w q4` |
|--------|---|--------:|---------:|---------:|--------:|
| text1m | 1 | 181557 | 182067 | 181588 | 183261 |
| text1m | 5 | 180159 | 179159 | 180185 | 182071 |
| libc1m | 1 | 379197 | 379704 | 379175 | 380969 |
| libc1m | 5 | 375998 | 377366 | 376107 | 377705 |

## QAT results

Archive bytes; Δ against the **same-format QAT-off** baseline in the first
column. `—` = configuration missing (background jobs were cancelled by a
server restart; not guessed).

| corpus | L | format | QAT off | `on` (recomputed) | `q8fixed` (init) | `q8fixed2` (untie) | `q8ema` |
|--------|---|--------|--------:|------------------:|-----------------:|-------------------:|--------:|
| text1m | 1 | q8  | 181557 | 181516 (−41)   | 184280 (+2723) | 181780 (+223)  | 181568 (+11)  |
| text1m | 5 | q8  | 180159 | 180183 (+24)   | 184242 (+4083) | 179078 (−1081) | —             |
| libc1m | 1 | q8  | 379197 | 379216 (+19)   | 379363 (+166)  | 379092 (−105)  | —             |
| libc1m | 5 | q8  | 375998 | 376006 (+8)    | 377853 (+1855) | 376502 (+504)  | —             |
| text1m | 1 | f16 | 182067 | 182066 (−1)    | —              | —              | —             |
| text1m | 5 | f16 | 179159 | 179137 (−22)   | —              | —              | —             |
| libc1m | 1 | f16 | 379704 | —              | —              | —              | —             |
| libc1m | 5 | f16 | 377366 | —              | —              | —              | —             |
| text1m | 1 | fp8 | 181588 | 181442 (−146)  | —              | —              | —             |
| text1m | 5 | fp8 | 180185 | 179064 (−1121) | —              | —              | —             |
| libc1m | 1 | fp8 | 379175 | —              | —              | —              | —             |
| libc1m | 5 | fp8 | 376107 | —              | —              | —              | —             |
| text1m | 1 | q4  | 183261 | 183360 (+99)   | —              | —              | —             |
| text1m | 5 | q4  | 182071 | 182389 (+318)  | —              | —              | —             |
| libc1m | 1 | q4  | 380969 | —              | —              | —              | —             |
| libc1m | 5 | q4  | 377705 | —              | —              | —              | —             |

Notable comparisons against the **auto** baseline (which can choose any
format): `q8fixed2` text L5 = 179078 (−88 B) and `fp8 on` text L5 = 179064
(−102 B). Both are small wins over the best standard encoder output.
`q8fixed` is catastrophically worse on text (+2.7 kB / +4.1 kB) with the new
SSE-optimizing quantizer: freezing the tiny initial scale saturates most
weights.

**Missing configurations (not measured, not estimated):** f16 QAT on libc1m
(both levels); fp8 QAT on libc1m (both levels); q4 QAT on libc1m (both
levels); q8 EMA on text L5 and libc1m (both levels). Fixed-scale modes were
only run for q8, as in the pre-rebase experiment. IQ8/IQ4/IQ2, Q2 and NVFP4
were not benchmarked at all.

## Verdict

**Negative overall, with two small text-L5 wins — not enough to adopt QAT as
a default.** On the new base:

- **q8, scale recomputed**: neutral (Δ = −41, +24, +19, +8 B; +10 B total).
  F16 is similarly neutral where measured (−1, −22 B).
- **q8, scale fixed at init**: clearly harmful (+0.2% to +2.3%), unlike on the
  old base where it gave the largest single win. The upstream quantizer now
  searches for an SSE-optimal scale; the initial amax scale is a much worse
  choice, so this mode should be dropped.
- **q8, scale fixed at untie**: mixed — best q8 result at text L5 (−1081 B vs
  forced q8, −88 B vs auto) but +504 B at libc L5 and +223 B at text L1.
- **EMA**: only text L1 measured (+11 B), inconclusive.
- **fp8 QAT**: the largest observed gain on text L5 (−1121 B vs forced fp8,
  −102 B vs auto), but no libc data, so it cannot be called robust.

Recommendation: keep the opt-in `TAW_QAT` knob and implementation, leave the
default `off`, and do not promote any QAT mode without completing the missing
corpus/level matrix and adding a coder/APM-aware candidate score (the current
`est` ranks by raw-mixer cross-entropy sampled without the APM stages).

## Caveats

- QAT is encoder-only and opt-in; default `off` is byte-identical to the
  provided baseline (verified for all four auto cases). Decoder determinism
  and golden vectors are untouched.
- The training simulation uses the *cheap* amax/max-level scale while the
  final `taw_quant_row_encode` searches for the SSE-optimal scale; a small
  train/serialize mismatch remains by design (a faithful search per forward
  was ~14× slower).
- FP8 and Q4 `on` rows were produced by the build *before* the Q8/Q4 O(1)
  fast-path edit. FP8's code path is unchanged by that edit, so its numbers
  stand; Q4's tie handling changed, so its two numbers are only indicative.
  Neither was re-measured after the edit (no new long jobs were started).
- GPU-trained warm-start weights skip QAT training; only serialization uses
  the QAT scale mode. No GPU was present in the benchmark environment.
- Q2 and NVFP4 are not simulated (no single level set); QAT leaves them alone.
- Encoder time rises substantially with QAT; wall-clock numbers are inflated
  by many parallel agents, so only sizes are reported.
- Tested at levels 1 and 5 on the two supplied 1 MiB corpora; `ctest` 12/12
  passes on the final build.

## Reproduce

```sh
cd <this worktree>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build -j1
ctest --test-dir build --output-on-failure          # 12/12
TAW_QAT=off QUANTS=auto OUT=bench-out/auto ./bench_local.sh
TAW_QAT=on  QUANTS="q8 f16 fp8 q4" DBG=1 OUT=bench-out/on ./bench_local.sh
TAW_QAT=q8fixed  QUANTS=q8 OUT=bench-out/q8fixed ./bench_local.sh
TAW_QAT=q8fixed2 QUANTS=q8 OUT=bench-out/q8fixed2 ./bench_local.sh
TAW_QAT=q8ema    QUANTS=q8 OUT=bench-out/q8ema ./bench_local.sh
```
