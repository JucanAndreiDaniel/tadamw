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

# C2 / C3 / NC9 AdamW stabilizers (branch `exp/eps-gate`)

> **Rebase note (2026-11-23):** this branch is rebased onto upstream `1aa3dd7`
> (merged PRs #3 MALTER and #4 Muon). Upstream's optimizer code is preserved
> unchanged: the C2/C3 delta wraps **only AdamW vector updates** that carry an
> `(m, v)` second moment, so the Muon, MALT and MALTER W1 paths are never
> touched. Knob-off output is byte-identical to the new upstream baseline on
> all four cells (181557 / 179166 / 379197 / 375291), and the tuned-AdamW C3
> numbers below reproduce exactly after the rebase (tuned control
> 179913 / 178886 / 373887 / 371894; tuned + C3 shrink
> 180047 / 178687 / 372296 / 368712; decode byte-identical). Upstream's
> merged MALTER leader totals 1,093,945 B, which is stronger than
> tuned-AdamW + C3 (1,099,742 B); C3 shrink is therefore an **AdamW-specific
> add-on**, not a new overall leader.

**Bottom line:** the *specified* NC9 recipe (per-coordinate ε **plus a mean-preserving
sign gate**) is **harmful** here (+678…+1060 B vs the tuned control on text1m L5).
The useful ingredient is the **shrink-only** sign-consistency gate (C3) applied to
**layer 2 only**: `TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink`.
On top of the strong tuned-AdamW control (`exp/lr-baseline` 9a08442) it gives
**−4,838 B total (−0.438%)** over the four benchmark cells and **−15,469 B
(−1.387%)** vs the shipped baseline. C2 alone gives a small, fragile −474 B.
None of this appears to be a cold-set effect: the gains are largest on hot sets.

## Method

Encoder-only change in `src/fit.c`; format and decoder untouched.

* **C2** — replace the scalar `eps = 1e-6` in the lazy AdamW update by
  `eps_i = max(c · (v_i/(1-β₂ᵗ))^α, 1e-8)`. Bias-corrected second moment;
  `α ∈ {½, 1}`, `c ∈ {1e-3, 1e-2}` (plus 3e-4/5e-4 probes). `TAW_EPS_GRAN=set`
  uses the row-mean second moment instead of `v_i`. Default off.
* **C3** — keep one sign-EMA per weight, `s_i ← β s_i + (1-β) sign(g_i)`, and
  multiply the AdamW coordinate update by `|s_i|^p`. Two variants:
  **mean-preserving** (divide by the row mean of the gate, so the tensor's
  total step magnitude is unchanged — the NC9 paper's recipe) and
  **shrink-only** (multiply directly). `β ∈ {0.9, 0.95}`, `p ∈ {1, 2}`.
  The sign state is mirrored from set 0 at un-tie, like `w/m/v`. Default off.
* **NC9** — both at once (`TAW_EPS_MODE=nc9`).
* Scope knobs: `TAW_EPS_LAYERS=1|2|12` (default **2**), `TAW_EPS_SCOPE=all|cold|warm`
  (cold/warm split by the median expected selector frequency over the samples).
* Control knobs: `TAW_LR0`, `TAW_LR1`, `TAW_B1`, `TAW_B2`, `TAW_WD` — defaults
  0.02 / 0.001 / 0.9 / 0.999 / 1e-5, so **the default build is unchanged**
  (verified byte-identical on all four cells).
* Diagnostic only: `TAW_FORCE_SEL1`, `TAW_FORCE_SEL2` filter the candidate
  shapes; `TAW_EPS_DEBUG=1` prints per-layer touch counts, mean `v`, `sqrt(v)`,
  `eps` and gate.

All archives were written under `/tmp/opencode/eps-gate-bench/`; the read-only
benchmark corpus directory was never written to. Build: clang Release;
`ctest` 12/12 after the final edit. Decode was checked byte-identical to the
sources for every finalist (`taw d` + `cmp`) on text1m L5 and libc1m L5.

## Sweep

Sizes are compressed archive bytes (1 MiB inputs), `-p mix`; Δ in parentheses.

### Baselines and controls

| configuration | text L1 | text L5 | libc L1 | libc L5 | total Δ |
|---|---|---|---|---|---|
| shipped baseline `e64cc83` | 181557 | 179166 | 379197 | 375291 | — |
| original control `lr 0.003→6e-5` | 183686 (+2129) | 179165 (−1) | 378209 (−988) | 374221 (−1070) | **+70** |
| **tuned AdamW control** (0.055→0.00055, β₁=0.95, WD=1e-3) | 179913 (−1644) | 178886 (−280) | 373887 (−5310) | 371894 (−3397) | **−10631** |

The 0.003 control is essentially neutral on the new base; the 9a08442 control
is the one that matters, so all stabilizer arms are reported both ways.

### Finalists, four cells

| arm | text L1 | text L5 | libc L1 | libc L5 | Δ vs baseline | Δ vs tuned control |
|---|---|---|---|---|---|---|
| C2 (α=½, c=1e-3) | 181544 (−13) | 179180 (+14) | 379197 (0) | 375372 (+81) | **+82** | — |
| C3 shrink (β=.9, p=1) | 180645 (−912) | 180050 (+884) | 379082 (−115) | 372601 (−2690) | **−2833** | — |
| NC9 shrink (C2 above + C3 shrink) | 180560 (−997) | 179574 (+408) | 379087 (−110) | 373466 (−1825) | **−2524** | — |
| C2 **on tuned** | 179999 (+86) | 178759 (−127) | 373561 (−326) | 371787 (−107) | −10105 | **−474** |
| **C3 shrink on tuned** | 180047 (+134) | 178687 (−199) | 372296 (−1591) | 368712 (−3182) | **−15469** | **−4838** |
| NC9 shrink on tuned | 180169 (+256) | 178555 (−331) | 372482 (−1405) | 368748 (−3146) | −15257 | −4626 |

### Parameter sweep, tuned control, text1m L5 (base = 178886)

| variant | size (Δ) |
|---|---|
| C2 α=½, c=3e-4 | 179377 (+491) |
| C2 α=½, c=5e-4 | 178970 (+84) |
| **C2 α=½, c=1e-3** | **178759 (−127)** |
| C2 α=½, c=1e-2 | 179348 (+462) |
| C2 α=1, c=1e-3 | 179416 (+530) |
| C2 α=1, c=1e-2 | 179358 (+472) |
| C3 mean β=.9 p=1 | 179519 (+633) |
| C3 mean β=.9 p=2 | 180443 (+1557) |
| C3 mean β=.95 p=1 | 179713 (+827) |
| C3 mean β=.95 p=2 | 181208 (+2322) |
| **C3 shrink β=.9 p=1** | **178687 (−199)** |
| C3 shrink β=.9 p=2 | 179001 (+115) |
| C3 shrink β=.95 p=1 | 178895 (+9) |
| C3 shrink β=.95 p=2 | 179542 (+656) |
| NC9 mean α=½,c=1e-3,β=.9,p=1 | 179792 (+906) |
| NC9 mean α=½,c=1e-2,β=.9,p=1 | 179564 (+678) |
| NC9 mean α=1,c=1e-2,β=.9,p=1 | 179946 (+1060) |
| **NC9 shrink (α=½,c=1e-3,β=.9,p=1)** | **178555 (−331)** |

The C2 optimum is narrow and its overall effect is at the noise floor; α=1
(the "variance-proportional" choice) is consistently worse. The mean-preserving
gate is uniformly harmful; shrink-only with β=.9, p=1 is the only gate setting
that helps. β=0.9 beats 0.95 and p=1 beats p=2 throughout (the latter shrinks
too aggressively in our very short schedule).

### Layer scope, tuned control, L5

| gate placement | text L5 | libc L5 |
|---|---|---|
| layer 2 only | **178687** | **368712** |
| layer 1 only | 181701 | 377388 |
| layers 1 + 2 | 178891 | 371521 |

Applying the gate to layer 1 is strongly harmful and cancels the layer-2 gain:
the layer-1 set has one shared 416-weight row driving every hidden unit, so
shrinking its coordinates is equivalent to an uncontrolled lr reduction.

### Touch-count stratification (forced `sel2=C0`, 256 output sets, text1m L5)

The shipped L1/L5 shapes have `sel1=NONE, sel2=BITPOS`: one hot layer-1 set and
only 8 hot output sets (~20k touches each) — **no cold sets**. To test the
hypothesis I forced `sel2=C0`, giving 110 touched sets out of 256, with
10 / 8,000 / 20,480 min / median / max touches per set (`TAW_EPS_DEBUG=1`).

| arm (forced shape, tuned control) | size (Δ vs 187511) |
|---|---|
| no stabilizer | 187511 |
| C2 (α=½, c=1e-2) | 187527 (+16) |
| C3 mean p=1 | 187514 (+3) |
| NC9 mean | 187415 (−96) |
| C3 shrink, all sets | 186812 (−699) |
| C3 shrink, **cold** sets only | 187078 (−433) |
| C3 shrink, **warm** sets only | **186758 (−753)** |
| NC9 shrink, cold sets only | 187072 (−439) |
| NC9 shrink, warm sets only | **186736 (−775)** |

Both cold and warm strata benefit, warm slightly more. The claimed cold-set
mechanism is **not** supported: the sign gate behaves as a general
sign-stability/adaptive-lr regularizer on the hot output rows.

### Mechanism check

On libc1m L5 the chosen candidate stays `H=16, sel1=NONE, sel2=BITPOS, fp8`
for both base and C3-shrink; the fp8 candidate's bpc improves 2.8692 → 2.8374
(−0.0318 bpc ≈ −3,182 B). The gain is from the fit itself, not from switching
candidates or quantizers.

## Verdict

* **Adopt?** C3 shrink-only (β=0.9, p=1) on layer 2 is a real incremental gain
  **only in combination with the tuned high-LR / high-WD control**: −4,838 B
  there, but on the shipped baseline it is a wash overall (−2,833 B, driven by
  libc L5, with a +884 B regression on text L5). It is cheaper than further LR
  tuning and does not touch the decoder, so it is a reasonable add-on to the
  tuned control, **not** a replacement for it.
* **Reject NC9 as specified.** The mean-preserving gate makes text1m L5 worse
  in every tested (β, p) combination, and in combination with C2 as well. The
  paper's mean-preserving rationale does not transfer to a 17-parameter output
  row trained for 6–8 epochs: dividing by the row mean pushes *more* step into
  sign-unstable coordinates, which the high LR then amplifies.
* **C2 alone is not worth shipping** — a single narrow setting gives −474 B on
  the tuned control, comparable to sweep noise, and is neutral on the raw base.
* **Layer scope matters:** layer 2 only. Layer 1 must be left on plain AdamW.
* **Cold-set targeting is unsubstantiated** in the benchmark shapes; the
  forced cold-set diagnostic shows warm sets benefit at least as much.

## Caveats / reproduction

* One seed per cell; no error bars. Effects below ~100 B are within
  candidate-selection and quantization noise (the C2 arms all live there).
* The gate starts at `|s|≈0`; with shrink-only this damps the first few touches
  of a set. Cold sets with <10 touches never build a reliable sign EMA — one
  reason cold-only gating is weaker.
* `eps_i` statistics are taken from the same short gradient history used for
  the update, exactly as in C2; β₂=0.999 bias correction is applied.
* The big C3 gain is concentrated on libc1m; the text1m L1 cell regresses
  slightly (+134 B). Portfolio-level total is still strongly positive, but a
  per-corpus regression guard may be warranted.
* All experiments used `-p mix`, CPU fitter (`-g off`), 1 MiB corpora, clang
  Release. Rebuild + run:
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
  cmake --build build -j1 && ctest --test-dir build --output-on-failure
  LEVELS="1 5" CORPORA="text1m libc1m" ./sweep_eps.sh tune_base tune_c3_b09_p1s
  ```
  Reproduce the winning arm directly:
  ```sh
  TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 \
  TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink \
  build/tools/taw c -l 5 -p mix corpus input.aw
  ```
