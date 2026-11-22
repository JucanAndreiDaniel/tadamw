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

---

# Experiment: learned probability calibration (arithmetic profile 6)

Branch `exp/calibration`, rebased onto `origin/main` `1aa3dd7`
(upstream PR #3 MALTER and PR #4 Muon, opt-in env knobs; previously e64cc83
with GPU fitting, new quant formats, bit-position sampling, fused AVX2
decoder and parallel awseek decode). Upstream's own MALT report is preserved
verbatim above; this section is only the profile-6 delta.

> **Stale-numbers note.** The first version of this report (base `a427e7f`)
> quoted text L1 190610 → 186508, text L5 187956 → 186098, libc L1
> 388792 → 384244, libc L5 385741 → 382830. Those archives came from the old
> sampler/fitter/decoder and are **invalid** after the rebase. The table
> below is the only current result.

## Design

Profile 6 (`CALIBRATED_MIX`) is profile 1 with five fitted integer scalars;
the context models, hashing, MLP, `squash`, squash/stretch tables, APM tables
and all state updates are byte-for-byte unchanged. Only two points in the
per-bit path are touched:

1. **Logit temperature + bias** applied to the mixer's integer stretch output
   `t ∈ [-2047, 2047]` (after the existing clamp/truncate, before `squash`):
   `t' = clamp(trunc_Q8(temp·t) + bias, -2047, 2047)`, where
   `trunc_Q8` rounds `(temp·t)/256` half away from zero using integer
   division only. No libm, no float arithmetic is added.
2. **Learned APM mix**, replacing profile 1's `(2·p + q1 + q2) >> 2` with
   `(w0·p + w1·q1 + w2·q2) / (w0 + w1 + w2)`, `w*` in `0..15`, with integer
   division. `p` is the calibrated `p_mix`; the APM bin lookup and APM
   updates still use `p_mix`.

With `temp = 256`, `bias = 0`, `(w0,w1,w2) = (2,1,1)`, profile 6 produces
exactly the same coded bytes as profile 1. A test asserts this.

The upstream fused AVX2 per-bit loop is **not** used for profile 6: it
hardcodes profile 1's probability, so encode/decode/skip dispatch to the
generic loop when `m->calibrated` is set. Profile 1 keeps the fast path.

### Fitting (`src/fit.c`)

* The network is trained on 7/8 of the sampled bits. Every 8th (position-
  evenly-spread) sample is held out. Fitting temperature/bias on the training
  samples was tried first and was badly overconfident: it applied heavy
  tempering (`temp=112`, `bias=-272`) and made archives ~12 KB larger under
  the old base. Holding out samples fixed it; new-base fits are mild
  (`temp=220..254`, `|bias| <= 24`).
* `temp`/`bias` are found by a two-stage grid search minimising cross-entropy
  on a 4096-bin histogram of the held-out samples' integer `t`. Cost per
  grid point is one pass over the bins; no libm is used.
* The APM weights are chosen from a fixed 14-triple set by measuring actual
  coding cost on a 16 KiB probe with the real predictor (`taw_pred_prob` /
  `taw_pred_advance`, including live APM adaptation) and unit resets.
* AUTO fitting evaluates profile 1 and profile 6 candidates together over
  all nine quant formats and picks the smaller estimated archive; `-p mix`
  and `-p cal` force one. Profile 6 candidates always fit on the CPU even
  with `-g on`/`-g auto`, because they need the held-out-sample path.
  Fitted archives are deterministic and thread-count independent.

A larger 26-triple APM grid was also tried on the old base: it overfit the
16 KiB probe and made three of four benchmarks worse, so the smaller grid
was kept.

## Format additions

`docs/format.md` gained profile 6 in the profile table, §2.3 (arithmetic) and
§5.2 (model layout). The model section is the profile 1 head (8 bytes) plus:

| off | size | field | range |
|-----|------|-------|-------|
| 8   | 2 | `temp` u16 LE | 64..512 |
| 10  | 2 | `bias` i16 LE | -1024..1024 |
| 12  | 1 | `w0` | 0..15 |
| 13  | 1 | `w1` | 0..15 |
| 14  | 1 | `w2` | 0..15, not all zero |
| 15  | 1 | reserved | 0 |

followed by the unchanged profile 1 weight payload (all nine quant formats
are accepted). Overhead versus profile 1 is **8 bytes**. Loaders validate all
ranges and the exact size; profile 1 loaders reject profile 6 models (and
vice versa). Public API: `TAW_PREDICTOR_CALIBRATED`, CLI `-p cal`.

## Results (post-rebase, new clean baseline)

Archives made with `-p mix` (baseline, profile 1) and `-p cal` (profile 6),
clang Release build at `e64cc83`, read-only corpora in
`/tmp/opencode/tadamw-bench`. Size is the whole `.aw` file in bytes. All
profile-6 archives decompress back to the exact input.

| corpus | level | new baseline | profile 6 | delta | quant base → cal | model base → cal |
|--------|-------|--------------|-----------|-------|------------------|------------------|
| text1m | 1 | 181557 | 180874 | **-683** (-0.38%) | q8 → q8 | 608 → 616 |
| text1m | 5 | 179166 | 178997 | **-169** (-0.09%) | q8 → fp8 | 608 → 616 |
| libc1m | 1 | 379197 | 375305 | **-3892** (-1.03%) | q8 → q8 | 608 → 616 |
| libc1m | 5 | 375291 | 373312 | **-1979** (-0.53%) | fp8 → fp8 | 608 → 616 |

**APM weights after the rebase:** `w=(0,1,1)` was still chosen for all four
runs, so the APM-dominant blend also wins under the new bit-position sampler.
Fitted temperatures/biases were `(250,12)`, `(250,0)`, `(254,-6)` and
`(220,20)` / `(224,24)` / `(206,22)` (three L5 candidates) respectively.
The 8-byte head cost is paid back 21×–486×.

The text L5 gain is small partly because the selected quant changed from q8
to fp8 in the profile-6 candidate set; the libc L5 gain is at the same quant
and is a cleaner comparison.

## Verification

* `ctest --test-dir build --output-on-failure`: **12/12 pass** on the rebased
  tree (including upstream's quant and GPU suites).
* Existing profile 1 golden vectors are unchanged; profile 6 goldens
  (F16 `len=43914 crc=0xD593C4D1`, Q8 `len=39189 crc=0xEBF4AF55`) still pass
  across every available ISA, so the upstream context/decoder changes did not
  disturb the pinned arithmetic.
* Added validation tests for out-of-range temp/bias/weights and reserved
  bytes, cross-profile model rejection, and identity-equivalence with
  profile 1.
* All four benchmark archives round-trip to their original corpora.
* With MALT/Muon knobs off (`TAW_HYBRID`/`TAW_MALT` unset), profile 1 output
  is byte-identical to the pre-MALT base on all four corpora/levels; all four
  profile-6 archives are likewise byte-identical to the pre-MALT rebase run.
  Upstream's opt-in optimizers therefore change neither baseline nor the
  calibration result.

## Caveats

* The APM weight probe is a single 16 KiB prefix of the input. A broader grid
  overfit it; results on data with very different local statistics may vary.
  A larger or spread-out probe would be more robust but costs fit time.
* The APM mixture fitting depends on the source bytes through `read_at`;
  for seekable archives only the first block is probed. It remains
  deterministic for a given input and independent of `threads`.
* The chosen `w0 = 0` discards the frozen mixer from the final probability
  on all tested inputs, so the profile-6 gain is mostly a statement about
  the APM stages being under-weighted in profile 1, not about calibration
  alone. On very small inputs, where APM adaptation has no time to converge,
  a non-zero `w0` may be preferable; the probe chooses per file.
* Profile 6 does not use the new fused AVX2 per-bit loop, so its decode is
  slower than profile 1 on the same machine. Only archive bytes were measured
  here; a throughput evaluation is outstanding.
* Profile 6 candidates are excluded from GPU training even with `-g on` to
  keep the held-out-sample calibration path; on GPU-equipped machines this
  makes `-p cal` fitting slower than `-p mix`.
* Fit time grows by roughly one predictor pass over 16 KiB per APM triple
  (14 triples) plus a 4096-bin grid search per candidate; measured wall time
  at L1/L5 was comparable to the baseline within noise on this shared box.
* The format now reserves profile id 6 for an experimental profile. Shipping
  it requires agreeing to keep the extension permanently.
