# Experiment: quantization-aware training (QAT) for the profile-1 mixer

Branch `exp/qat` (no push). Only `src/fit.c` changed; the archive format, model
section layout and decoder are untouched. `bench_local.sh` reproduces the table.

## What changed

The encoder fits the mixer with AdamW in full precision and only then applies
F16 or per-row Q8 quantization (`put_row`). Small weights that round to zero
(Q8) or lose precision (F16) were therefore never optimized against. This
experiment simulates the decoder's dequantizer **inside the training forward
pass**, with a straight-through estimator (STE) for rounding/scale:

- F16: forward uses `h2f(f2h(clamp_half(w)))` for every weight (both layers and
  the output bias); gradients pass through the round-trip unchanged.
- Q8: forward uses `q = clamp(round(w / scale), -127, 127); wq = q * scale`
  per row, with `scale` the same F16-rounded per-row absmax/127 that the
  serializer (and hence the decoder) uses. The backward pass uses `wq` for the
  hidden-unit gradient and STE (`d wq / d w = 1`) for the weight gradients.
  The derivative of the scale through absmax is deliberately dropped.
- Scale policies for Q8: (a) **recomputed** from current weights every forward;
  (b) **fixed at init**; (c) **fixed at untie** (end of tied warm-up);
  (d) **EMA** of the per-row absmax scale (α=0.98), updated per forward.
  Fixed/EMA scales are written into the ordinary Q8 scale header at
  serialization; the model is a standard profile-1 Q8 model.

Everything is behind `TAW_QAT` (default `off` => original behaviour):
`off | on | q8fixed | q8fixed2 | q8ema | f16only`. `on` applies QAT matching
each candidate's quantizer; `q8fixed*` additionally apply F16 QAT to F16
candidates; `f16only` applies QAT only to F16 candidates.

## Results

Corpora: `text1m`, `libc1m` (1 MiB each), `-p mix`, levels 1 and 5, `-j` default
1. Sizes in bytes; Δ = archive size minus the QAT-off baseline in the same row.
`model` is the embedded model-section size, identical in every mode because
QAT changes no shape.

| corpus | L | quant | QAT off (baseline) | on (recomputed) | q8fixed (init) | q8fixed2 (untie) | q8ema |
|--------|---|-------|-------------------:|----------------:|---------------:|-----------------:|------:|
| text1m | 1 | q8  | 190610 (m1888) | 190624 (+14)  | **189686 (-924)** | 190615 (+5)   | 190750 (+140) |
| text1m | 5 | q8  | 187956 (m2696) | 187957 (+1)   | 188418 (+462)  | **187823 (-133)** | 187958 (+2)  |
| libc1m | 1 | q8  | 388792 (m1888) | 388948 (+156) | 388761 (-31)   | 388948 (+156)  | 388940 (+148) |
| libc1m | 5 | q8  | 385741 (m2696) | **385631 (-110)** | 385676 (-65) | 385693 (-48) | **385558 (-183)** |
| text1m | 1 | f16 | 190870 (m1752) | 190869 (-1)   | 190869 (-1)    | 190869 (-1)    | 190869 (-1)   |
| text1m | 5 | f16 | 186911 (m1752) | 186917 (+6)   | 186917 (+6)    | 186917 (+6)    | 186917 (+6)   |
| libc1m | 1 | f16 | 389041 (m1752) | 389042 (+1)   | 389042 (+1)    | 389042 (+1)    | 389042 (+1)   |
| libc1m | 5 | f16 | 385282 (m1752) | 385278 (-4)   | 385278 (-4)    | 385278 (-4)    | 385278 (-4)   |

Verification:

- QAT-off archives are **byte-identical** to the supplied baselines for all
  four q8 cases; f16-off also measured as a control.
- Every generated archive decodes back to the original input (`taw d`); the
  header still reports profile 1 with quant 4 (q8) or 1 (f16).
- Full `ctest` (10/10) passes with `TAW_QAT` unset and with `on`, `q8fixed`,
  `q8fixed2`, `q8ema`. A `TADAMW_WERROR=ON` build is warning-free.
- Sizes are reproducible across repeated runs; encoder wall-clock timings are
  not reliable here (many parallel agents on 12 cores), so they are omitted.

## Verdict

**Negative / inconclusive — do not enable by default.** The effect is small
(≤ 0.5%) and not consistent across corpora or levels:

- **F16 QAT is a no-op in practice** (|Δ| ≤ 6 B, i.e. ≤ 0.003%): mixer weights
  are far from F16 subnormal/precision limits, so simulation changes almost
  nothing.
- **Q8, scale recomputed each step** is essentially neutral (+61 B total over
  the four q8 cases), with a small L5 libc win (-110 B) offset by L1 libc
  regression (+156 B).
- **Q8, scale fixed from init** gives the single largest win (text L1: -924 B,
  -0.49%) but the largest regression too (text L5: +462 B, +0.25%); net -558 B.
  It saturates many more weights (e.g. text L1: 222 vs 72 of 1736 weights at
  ±127), i.e. it behaves like an accidental ternary-net regularizer rather
  than a principled quantization fix.
- **Q8, scale fixed at untie** wins at both L5 cases (-133, -48) but regresses
  L1 libc (+156); net -20 B.
- **EMA scale** wins the best L5 libc result (-183 B) but regresses both L1
  cases (+140, +148); net +107 B.

The only reasonably consistent signal is a modest L5 libc gain (-48 to -183 B)
from any scale-stabilizing Q8 variant; text L1 and libc L1 are unstable. A
selector that chose the QAT mode per input would need a better proxy than the
current `est` heuristic: it ranks candidates by raw-mixer cross-entropy on
sampled bits and ignores the two APM stages, so in this experiment it often
ranks a QAT model worse even when its archive is smaller (e.g. fixed-init text
L1: bpc 0.9722 vs 0.9669, archive 924 B smaller).

Recommendation: keep the opt-in `TAW_QAT` knob and the implementation, but
leave the default `off`. A follow-up would need to score candidates with the
real coder (including APM) before adopting QAT as default, and to test on more
corpora/levels.

## Caveats

- Fitting is encoder-only and need not be reproducible, but here all modes are
  deterministic for fixed seed/thread count; the observed deltas are due to
  QAT alone, not run-to-run noise.
- The `bpc` printed by `TAW_FIT_DEBUG` is raw-mixer bits/byte on the sampled
  bits; it is only a candidate-selection heuristic and does not predict the
  final archive size (the APM stages and full-data coding are excluded).
- Wall-clock encoder time roughly rises with QAT (extra per-row scale scans,
  ~1.3–2× on the noisy shared machine); no isolation timing was possible.
- EMA/fixed modes serialize their frozen/smoothed scale rather than the final
  absmax scale; still a valid standard Q8 model, but the weights can clip at
  ±127 by design.
- `q8ema` mutates training state (the EMA) inside the forward pass; this is
  encoder-only and does not affect decoder determinism or golden vectors.
- Tested only at levels 1 and 5 and on the two supplied 1 MiB corpora.

## Reproduce

```sh
cd <this worktree>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build -j1
ctest --test-dir build --output-on-failure
TAW_QAT=on       OUT=bench-out/on       DBG=1 ./bench_local.sh
TAW_QAT=q8fixed  OUT=bench-out/q8fixed  DBG=1 ./bench_local.sh
TAW_QAT=q8fixed2 OUT=bench-out/q8fixed2 DBG=1 ./bench_local.sh
TAW_QAT=q8ema    OUT=bench-out/q8ema    DBG=1 ./bench_local.sh
```
