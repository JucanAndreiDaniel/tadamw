# MALT / MALTER on the current leader (`exp/malt`, base `d41bd57`)

Encoder-only implementation of **MALT: Lightweight Curvature-Aware Muon via
Diagonal Preconditioning** (arXiv:2608.05088, Wu, Dong, Sun, Ma) and its
noise-adaptive variant **MALTER**. Format, decoder, quantization and golden
vectors are untouched. No push. Corpora in `/tmp/opencode/tadamw-bench` were
read-only; all archives live in `bench/arcs/` in this worktree.

## TL;DR / verdict

**Verified win over the tuned-Muon leader.** Best configuration:

```
TAW_MALT=malter \
TAW_MALT_LR0=0.28 TAW_MALT_LR1=0.0056 \
TAW_MALT_B2=0.9 TAW_MALT_EPS=1e-8 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015
```

| corpus / level | shipped AdamW | tuned AdamW | tuned Muon (base) | best plain MALT | **best MALTER** |
|---|---:|---:|---:|---:|---:|
| text1m L1 | 181557 | 179913 | 179678 | 179635 | **179405** |
| text1m L5 | 179166 | 178886 | 177816 | 177804 | **177138** |
| libc1m L1 | 379197 | 373887 | 371373 | 371743 | **370898** |
| libc1m L5 | 375291 | 371894 | 366997 | 366329 | **366504** |
| **sum L1+L5** | 1,115,211 | 1,104,580 | 1,095,864 | 1,095,511 | **1,093,945** |

MALTER beats tuned Muon by **−1,919 B (−0.18 %)** overall and is
**Pareto-better on all four cells** (−273 / −678 / −475 / −493 B). It beats
tuned AdamW by −10,635 B (−0.96 %) and shipped AdamW by −21,266 B (−1.91 %).
Plain MALT is also slightly better in sum (−353 B) but *not* Pareto: it loses
370 B on libc1m L1. MALTER is the clear winner, matching the paper's claim
that its noise-adaptive step-size adds a complementary gain over MALT.

Winner reproduces byte-identically on a rerun; four decode checks returned
`ok=1`; knob-off paths reproduce all four baseline archives exactly;
`ctest` = **12/12**.

## Paper summary

Muon orthogonalizes the momentum matrix (Newton–Schulz) to flatten its
singular spectrum, but it ignores the *curvature* anisotropy of the loss: with
an ill-conditioned Hessian, a fixed-magnitude update is too aggressive along
sharp directions and too timid along flat ones. MALT addresses this by
preconditioning **before** orthogonalization with two lightweight diagonal
factors built from row/column squared-gradient-norm EMAs (a diagonal surrogate
for the Shampoo/Kronecker curvature factors), then mapping the orthogonalized
direction back and grafting the update magnitude. MALTER adds a scalar
Adam/NAMO-style signal-to-noise step-size in the preconditioned space.
The paper proves stochastic non-convex convergence for MALT and reports
GPT-2 Small/Medium/Large pretraining wins over Muon at nearly identical memory
and wall-clock.

## Exact equations (paper Algorithms 1–2, §3.3, §3.5)

For a matrix parameter `W ∈ R^{m×n}`, gradient `G_t`, momentum decay `μ1`,
preconditioner EMA `μ2`, damping `ε`, learning rate `η`:

```
M_t  = μ1 M_{t-1} + (1-μ1) G_t
l_t  = μ2 l_{t-1} + (1-μ2) [ ||G_t[1,:]||², ..., ||G_t[m,:]||² ]
r_t  = μ2 r_{t-1} + (1-μ2) [ ||G_t[:,1]||², ..., ||G_t[:,n]||² ]
L_t  = Diag((l_t + ε)^(-1/8)),   R_t = Diag((r_t + ε)^(-1/8))
M~_t = L_t M_t R_t
O_t  = Orth(M~_t)                     # NS polar factor, singular values -> 1
D_t  = L_t O_t R_t                    # map back to parameter space
MALT:    W_t = W_{t-1} - η · (||O_t||_F / (||D_t||_F + ε)) · D_t

MALTER adds:
ν_t  = μ2 ν_{t-1} + (1-μ2) ||L_t G_t R_t||_F²
α_t  = [ sqrt(1-μ2^t) / (1-μ1^t + ε) ] · ||M~_t||_F / (sqrt(ν_t) + ε)
W_t  = W_{t-1} - η · α_t · (||O_t||_F / (||D_t||_F + ε)) · D_t
```

`Orth` is the same 5-step Newton–Schulz polynomial used by this codebase's
Muon (`A = X Xᵀ`, `B = −4.7750 A + 2.0315 A²`, `X = 3.4445 X + B X`), with the
preconditioned momentum first Frobenius-normalized. `(l+ε)^(-1/8)` is computed
as `1/sqrt(sqrt(sqrt(l+ε)))` — no libm dependency.

Two deliberate adaptations to the existing fitter (documented in the code):

* **Momentum convention.** The codebase uses `M = μ1 M + G` (no `(1-μ1)`),
  which Muon's normalize-then-orthogonalize path is invariant to. MALTER's
  `α` is *not* invariant to it, so `||M~||` is multiplied by `(1-μ1)` before
  forming `α`, recovering the paper's signal-to-noise ratio. Without this,
  `α` is ~20× too large and training is unstable.
* **Nesterov.** We keep the codebase's Nesterov extrapolation of the
  momentum before preconditioning, so MALT/MALTER differ from the tuned Muon
  only in the two-sided diagonal preconditioning and (for MALTER) the step
  rescale.

W1 is one 16×26 matrix; W2 (8×17 vectors) stays on AdamW exactly as in the
base. Per-set `l`/`r` vectors and the scalar `ν` are copied at untying like
the other W1 state; the extra state is `O(m+n)` per matrix.

## Environment knobs (all unset = shipped behaviour)

| variable | default | meaning |
|---|---|---|
| `TAW_MALT` | `off` | `off`/`0`: shipped AdamW; `1`/`on`/`malt`: MALT on W1; `malter`: adaptive MALTER. Overrides `TAW_HYBRID`. |
| `TAW_MALTER` | `0` | `1` promotes `TAW_MALT=malt` to MALTER. |
| `TAW_MALT_LR0` / `TAW_MALT_LR1` | `0.055` / `0.0011` | linear W1 LR endpoints (Muon-tuned defaults). |
| `TAW_MALT_B2` | `0.999` | preconditioner EMA decay `μ2` (also MALTER's `ν` decay). |
| `TAW_MALT_EPS` | `1e-8` | damping `ε` in `(l+ε)^(-1/8)`, graft and MALTER denominator. |
| `TAW_MALT_GRAFT` | `o` | graft source: `o` = `||O||_F` (paper), `m` = `||M~||_F`, `d` = none (`||D||_F`). |
| `TAW_MUON_MOM` / `TAW_MUON_NS` / `TAW_MUON_NEST` | `0.95` / `5` / `1` | shared with Muon: momentum `μ1`, NS steps, Nesterov. |
| `TAW_VEC_LR0` / `TAW_VEC_LR1` | `TAW_LR0` / `TAW_LR1` | W2 AdamW LR (winner uses `0.003` / `0.00015`). |
| `TAW_MALT_DEBUG` | unset | print final `l`/`r`/`ν` statistics to stderr. |

## Sweep table

Full sensitivity grid (all four cells, `-p mix`, bytes; W2 fixed to
`0.003→0.00015` unless noted). Baseline controls: Muon = 1,095,864 B;
tuned AdamW = 1,104,580 B; shipped = 1,115,211 B.

| W1 mode | LR0 → LR1 | b2 | eps | W2 | sum bytes | vs Muon |
|---|---|---:|---:|---|---:|---:|
| MALTER | 0.28 → 0.0056 | 0.90 | 1e-8 | 0.003→1.5e-4 | **1,093,945** | **−1,919** |
| MALTER | 0.28 → 0.0070 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,003 | −1,861 |
| MALTER | 0.26 → 0.0052 | 0.90 | 1e-8 | 0.005→2.5e-4 | 1,094,044 | −1,820 |
| MALTER | 0.25 → 0.0050 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,284 | −1,580 |
| MALTER | 0.40 → 0.0080 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,526 | −1,338 |
| MALTER | 0.35 → 0.0070 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,094,536 | −1,328 |
| MALTER | 0.30 → 0.0060 | 0.90 | 1e-4 | 0.003→1.5e-4 | 1,094,605 | −1,259 |
| MALTER | 0.30 → 0.0030 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,087 | −777 |
| MALT | 0.055 → 0.0011 | 0.90 | 1e-8 | 0.003→1.5e-4 | 1,095,511 | −353 |
| MALT | 0.065 → 0.00065 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,600 | −264 |
| MALT | 0.055 → 0.0011 | 0.99 | 1e-4 | 0.003→1.5e-4 | 1,095,958 | +94 |
| MALTER | 0.25 → 0.0050 | 0.90 | 1e-8 | 0.055→5.5e-4, b1=.95 | 1,100,016 | +4,152 |

Ablations (earlier proxy sweep with default W2, text1m+libc1m L1 sum;
Muon proxy 551,051): graft `o` = 550,647; graft `m` = 552,544; graft `d`
= 599,926 (diverges). With MALTER's winner: `eps` 1e-9 → 1,095,313,
1e-7 → 1,095,480, 1e-6 → 1,094,358 (all worse than 1e-8);
`b2` 0.85 → 1,094,576, 0.95 → 1,095,790 (both worse than 0.90);
final-LR ratio 1/30 → 1,094,374, 1/60 → 1,094,343, 1/75 → 1,094,214.
W2 tuned-Adam settings (b1=.95, wd=1e-3) are much worse for MALTER than
the leader's small W2 LR.

## Verification

* `bench.sh` winner run: 179405 / 177138 / 370898 / 366504, all four
  `decode_ok=1` (byte-exact vs the shared corpora).
* Knob-off: `TAW_MALT=off TAW_MALTER=0` reproduces all four baseline archives
  byte-for-byte (`cmp`), including with the env vars explicitly set to their
  off values.
* Determinism: rerunning the winner archive is byte-identical.
* `ctest --test-dir build --output-on-failure`: **12/12 passed**.
* Clean GCC 16 `-Werror`-compatible build; no libm added. A separate
  `-DTADAMW_WERROR=ON -DTADAMW_SANITIZE=ON` build compiles cleanly and passes
  11/12; the lone `gpu` failure is LeakSanitizer reporting leaks inside
  `libdbus-1.so` from the Vulkan runtime (no MALT code on that path).

## Caveats

* **Scale transfer is weak.** The paper's GPT-2 matrices are large; our W1
  is a single 16×26 matrix. With `l ~ 1e-6…1e-3`, `(l+ε)^(-1/8)` only varies
  by ~1.3–2.2×, so the diagonal preconditioner is much milder here than in
  the paper's regime. The gain is correspondingly small (~0.18 % over Muon).
* **Benchmark-specific tuning.** The sweep is over this one fitter/corpus
  pair; several near-best MALTER configurations differ by only a few hundred
  bytes and the top configuration may not generalize to other shapes. Treat
  the numbers as an existence proof, not a universal hyperparameter recipe.
* **LR does not transfer from the paper.** MALTER's `α` is below 1 in this
  noisy, lazily-updated regime, so it needs ~5–10× the W1 LR of MALT/Muon
  (0.25–0.4 vs 0.055). Using paper-like LRs with MALTER is much worse.
* **Not Pareto for plain MALT.** Its sum win comes from libc1m L5; it loses
  on libc1m L1. Only MALTER improves every cell.
* **Single W1 set.** `sel1=NONE` on the winning shape, so per-set routing and
  the multi-set aspects of the hybrid code are not exercised by MALT here.
* **CPU/encoder only.** The Vulkan trainer is unchanged; MALT is not
  implemented there. Memory overhead is `O(m+n)` extra floats per matrix,
  plus one scalar for MALTER; compute is one NS pass plus row/column sums,
  i.e. essentially Muon's cost.

## H-Net boundary feature (profile 14) — branch `exp/hnet-feature`

Rebased onto `origin/main` `1aa3dd7`. New arithmetic profile **14**
`HNET_BOUNDARY`, CLI `-p hnet`, public predictor `TAW_PREDICTOR_HNET`.

### 1. Design

H-Net's dynamic chunking ([arXiv:2507.07955](https://arxiv.org/abs/2507.07955))
scores a boundary by the dissimilarity of adjacent context representations,
`p_t = ½(1 − cos(q_t, k_{t−1}))`. This profile keeps that one idea but builds
the signal entirely from tadamw's existing integer context state: no libm, no
cosine, no extra learned weights, no extra archived fields.

* **Sampled representation.** Four stretched estimates that profile 1 already
  computes: `st[0]` (o0), `st[1]` (o1), `st[7]` (word), `st[24]` (match).
  `x[i] = st[i]/256` is exact in binary32, so `u[j] = x[P[j]] * 256` recovers
  the integer stretch value with no rounding.
* **Sampling instant.** Once per byte, on the inputs of `bitpos = 7` (last bit
  of the byte), before that bit is coded. The value is held for that bit and
  the first seven bits of the next byte, so it spans the byte-to-byte
  transition. Sampling at `bitpos = 0` instead (`c0 = 1`, a near-constant o0
  context) was measured first and was clearly worse: it cost 243/29/16/108
  bytes against the new baseline on text1m L1/L5 and libc1m L1/L5.
* **Score.** With `prev[4]` the previous sample (zero at a coding-unit start),

  ```
  d[j] = |u[j] − prev[j]|;  s = Σ d[j];  b = min(s, 4*2*2047 = 16376);  prev = u
  x[26] = b/256   (exact; b is a non-negative integer in stretch units)
  ```

  Four subtractions, four absolute values and one clamp per byte. `b` lives
  in decoder state and is reset with the context state per coding unit
  (whole `.aw` stream, each `.awseek` block).
* **Mixer.** Profile 1's frozen MLP with `F = 27` instead of 26: the same
  direct inputs, the same bias at `x[25]`, the same fixed `dot4` order,
  hidden clamp ±8, output clamp ±2047/256, integer squash and the same two
  APM stages. No new selector, table or weight format; only the weight rows
  are one element wider (27 instead of 26).
* **Feature pass.** `fit.c` calls the same `phnet_boundary()` on the same
  first-of-byte inputs, so the fitter trains on exactly the signal the decoder
  sees; `tests/test_hnet.c` checks this bit for bit, including after a
  coding-unit reset.

Rationale, why the last bit rather than the first, and the full arithmetic are
in `docs/format.md` §2.14/§5.1 and `docs/design.md` "H-Net boundary profile".

### 2. Format and implementation additions

| item | addition |
|------|----------|
| header | `arith_profile = 14` (`TAW_PROFILE_HNET_BOUNDARY`); same header and model-section layout as profile 1 |
| model head | identical 8 bytes; reserved bytes 6–7 must be zero |
| weight rows | `len = 27` for layer-1 rows and for the single-layer output row; quant formats unchanged (F16, Q8, Q4, Q2, FP8, NVFP4, IQ8, IQ4, IQ2) |
| loader | width is derived from `arith_profile`; a 26-wide payload under 14 (or 27-wide under 1) is `TAW_E_FORMAT`; all existing head/quant checks apply |
| state | `prev[4] = 0`, `b = 0` at each coding-unit reset; `b` constant within a byte |
| files | `src/phnet.{h,c}` (new), `src/p1.{h}`/`p1_net.c` (parameterised row width, shared APM), `src/predictor.c`, `src/fit.c`, `src/fit_internal.h`, `src/format.h`, `include/tadamw/tadamw.h`, `src/header.c`, `tools/taw.c`, `src/CMakeLists.txt` |
| tests | `tests/test_hnet.c` (new), registered in `tests/CMakeLists.txt` |

**SIMD note.** The fused AVX2 per-bit loop (`p1_run_avx2.c`) and the AVX2 /
AVX-512 mixer kernels (`p1_simd_out.h` has `static_assert(P1_F == 26)`)
hardcode `F = 26`. Profile 14 therefore dispatches to the **scalar mixer and
the generic per-bit loop** on every CPU. The context-model kernels are shared
and bit-identical, and ISA-forcing tests confirm the same archive bytes from
each ISA; no new SIMD kernel was added.

GPU fitting is not available for the wider rows: `-g auto` silently fits on
the CPU and `-g on` with `-p hnet` returns `TAW_E_ARG`.

### 3. Results

Corpora and baseline archives are the read-only `/tmp/opencode/tadamw-bench`
set (1 MiB `text1m`, 1 MiB `libc1m`). Bytes are primary; encode/decode times
were taken best-of-3 for decode with the same clang Release build. The "new
baseline" is profile 1 at `e64cc83`; "entropy gate" is sibling profile 11 as
given in the task.

| profile / level | text1m | Δ vs new base | libc1m | Δ vs new base | sum |
|-----------------|-------:|--------------:|-------:|--------------:|----:|
| new baseline 1 / 5 | 181557 / 179166 | — | 379197 / 375291 | — | 1115211 |
| entropy gate 11, 1 / 5 | 181473 / 179098 | −84 / −68 | 378070 / 374705 | −1127 / −586 | 1113346 |
| **hnet 14, 1 / 5** | **181307 / 179426** | **−250 / +260** | **377294 / 371256** | **−1903 / −4035** | **1109283** |

* Total: **−5928 bytes (−0.53 %)** versus the new baseline; **4063 bytes
  better than the entropy-gate profile** on the same four points.
* Wins on text L1 (−0.14 %), libc L1 (−0.50 %) and libc L5 (−1.08 %); loses
  on text L5 (+0.15 %). Model section grew from 608 to 624 bytes (wider
  rows) in all four runs.
* `.awseek` sanity check with 64 KiB blocks (16 independent units): profile 14
  229421 / 422506 bytes versus profile 1 229962 / 422664 bytes for
  text1m / libc1m at level 1 (−541 / −158), with the per-block boundary reset
  exercised.

Timing (same machine, other agents benchmarking concurrently, so relative
only): profile 14 encode ≈ 9 s (L1) / 19–21 s (L5) and decode ≈ 6 s per MiB,
versus ≈ 1.8 s / 7.4 s and ≈ 0.9 s for the fused profile 1. This is the cost
of the generic loop, not of the boundary arithmetic itself.

### Correctness / verification

* Rebased onto upstream `origin/main` `1aa3dd7` (merged PRs #3 MALTER and
  #4 Muon). Upstream's optimizer code and bench results are kept verbatim;
  only the optimizer's W1 width was generalised from the constant `P1_F` to
  the net's `t->f` (26 for profiles 1/11-style fits, 27 for profile 14) in
  the MALT/Muon allocation, untie, update and debug paths. With the tuning
  env vars unset, upstream's default fit behaviour is unchanged.
* `ctest`: **13/13 passed** (12 baseline tests + the new `hnet` test),
  including all profile-1 golden vectors unchanged and the new profile's
  F16/Q8 golden vectors (len/CRC `57761 / 0xEE2926BD`, `48573 /
  0x482211FB`). The CLI roundtrip script now also exercises `-p hnet`
  stream + `.awseek` + range extraction. A GCC 16 `-Werror` build is clean
  and its tests pass too.
* Decode of all four new archives reproduces the original input byte for byte.
  The existing profile-1 baseline archives also still decode byte for byte
  with the modified decoder. After the rebase, all four profile-14 archives
  were re-encoded and came out **byte-identical** to the pre-rebase ones
  (181307 / 179426 / 377294 / 371256 bytes), and a clang `-Werror` Release
  rebuild passes the suite.
* Fit-pass and decoder boundary signals match for every byte of a skewed
  input across a mid-stream coding-unit reset.
* Boundary checksum over 4096 fixed text bytes: `sum=8352596`,
  `sum2=1113733144`, `min=0`, `max=11517`, 9 zero samples.

### 4. Verdict

**Keep as an experimental, opt-in profile.** The feature is deterministic,
cheap in integer operations, adds only one mixer weight per row, and gives a
net byte win on the bench (−0.53 % overall, −1.08 % on libc L5), beating the
sibling entropy-gate profile in total bytes on these four points. It does not
win everywhere (text L5 is slightly worse) and it is not a default: use
`taw c -p hnet`.

### 5. Caveats

* Speed: no fused AVX2 path; encode/decode are several times slower than
  profile 1. Adding a SIMD kernel requires reworking `p1_simd_out.h` and
  `p1_run_avx2.c` for `F = 27`, which was out of scope.
* The clamp bound is the natural saturating width of four stretch values and
  does not actually truncate any valid sum (max 16376). It is part of the
  spec to pin the register width; loosening it to 4095/8191 or tightening it
  to 2047 changed the fitter's candidate choices and produced mixed results
  (the 2047 variant lost ~700 bytes overall against the new baseline).
* The first sample of a coding unit compares against the zero vector, which
  acts like an artificial boundary at the start of each `.awseek` block.
  Its effect is one byte per block; the 64 KiB-block check still improved.
* Only 1 MiB text and 1 MiB libc at levels 1 and 5 were measured. Fitting is
  deterministic on the CPU but the chosen quant/shape can jump between
  levels, so small differences here should not be over-read.
* Profile 14 is only available with CPU fitting; GPU `-g on` is rejected.
