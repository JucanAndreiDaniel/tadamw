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

# Appendix: AdamW LR/WD baseline sweep (PR #7)

# Tuning the plain AdamW baseline — v2, rebased on `origin/main` 1aa3dd7

> **STALE NOTICE.** The earlier sweep (a427e7f, 51 configs, "best 0.003→6e-5")
> is preserved in `sweep_table.md` / `results.tsv` / `arcs/` / `phase*.log`,
> but is **invalid** after upstream's fit.c rewrite (GPU fitting, bit-position
> sampling, new quant formats, FIT_BATCH). Numbers below were re-measured from
> scratch on e64cc83 and re-verified byte-for-byte after rebasing onto
> 1aa3dd7 (upstream PRs #3 MALTER and #4 Muon). Upstream optimizer code is
> untouched; this PR keeps only docs/sweep data and the `TAW_LR1=flat`
> convenience.

## TL;DR — best plain AdamW configuration (v2)

| knob | value |
|---|---|
| `TAW_LR0` (initial LR) | **0.055** |
| `TAW_LR1` (final LR) | **0.00055** (= LR0/100) |
| `TAW_B1` | **0.95** (upstream ships 0.9) |
| `TAW_B2` | 0.999 (unchanged) |
| `TAW_WD` | **1e-3** (upstream ships 1e-5) |
| eps | 1e-6 (unchanged, no override) |

Actual archive bytes with `-p mix`:

| corpus / level | new shipped baseline | tuned | Δ |
|---|---:|---:|---:|
| text1m L1 | 181557 | 179913 | **−1644 (−0.905%)** |
| text1m L5 | 179166 | 178886 | **−280 (−0.156%)** |
| libc1m L1 | 379197 | 373887 | **−5310 (−1.400%)** |
| libc1m L5 | 375291 | 371894 | **−3397 (−0.905%)** |
| **sum** | **1,115,211** | **1,104,580** | **−10,631 (−0.953%)** |

This configuration is **Pareto-better**: it improves every cell, including
text1m L5 where almost every high-LR variant regresses. Runner-up
`0.055→0.000275, β1=0.95, WD=1e-3` is −10,286 but loses 186 bytes on text L5.
The best LR-only (β/WD shipped) config is `0.04→0.002, β1=0.9, WD=1e-5`
with −6,829 bytes and all four cells improved.

## What changed in `src/fit.c` after the 1aa3dd7 rebase

- Upstream PRs #3/#4 already ship the `TAW_LR0`, `TAW_LR1`, `TAW_B1`,
  `TAW_B2`, `TAW_WD` overrides and the hybrid/Muon/MALTER optimizers. All of
  that upstream code was preserved verbatim.
- The entire PR delta is now six lines in `tnet_train()`: `TAW_LR1=flat`
  keeps the LR constant (otherwise upstream's default LR0/20 schedule is
  used), plus a comment. `git diff origin/main -- src/fit.c` shows exactly
  this hunk and nothing else.
- **The shipped default was NOT changed.** With no overrides, all four
  baseline archives are byte-identical (`cmp`) to the pre-rebase v2 archives
  and match 181557 / 179166 / 379197 / 375291; the tuned config likewise
  reproduces 179913 / 178886 / 373887 / 371894 byte-for-byte.
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`,
  `cmake --build build -j1`, `ctest --test-dir build --output-on-failure`:
  **12/12 passed** (53.09 s) on 1aa3dd7.

## Method (v2)

- Same 1 MiB corpora (`/tmp/opencode/tadamw-bench/{text1m,libc1m}`, read-only),
  `-p mix`, distinct `arcs_v2/<config>_<corpus>_L<level>.aw` outputs, archive
  bytes as the only metric; `JOBS=3` on the shared 12-core box.
- Phase A: LR0 ∈ {0.02, 0.01, 0.005, 0.003, 0.002, 0.001}, LR1=LR0/20.
  B: {0.05, 0.03, 0.015, 0.008, 0.004, 0.0035, 0.0025}.
  F: {0.025…0.07}. G: LR1 ratios at LR0=0.04 (1/10, 1/20, 1/50, 1/100, flat,
  absolute 0.003/0.01). H/I: lower final LR and neighbouring LR0.
  D: β1 {0.9,0.95} × β2 {0.999,0.99} × WD {0,1e-5,1e-4,1e-3} at
  LR0=0.06, LR1=0.0006. J/K: cross LR × β1 × WD, final refinement.
- 71 configurations × 2 corpora × L1/L5 = 284 measurements
  (`results_v2.tsv`, `phase{A,B,D,F,G,H,I,J,K}_v2.log`, `arcs_v2/`).
  No L9 runs: the task asks for a focused L1/L5 sweep and L9 costs ~2 min/run
  on this box.

## Full v2 sweep table

Cells are `bytes (Δ vs 181557 / 179166 / 379197 / 375291)`; `total Δ` is the
sum of the four deltas. Note phase D used LR0=0.06, LR1=0.0006 for all rows.

| phase | config (env overrides) | text L1 | text L5 | libc L1 | libc L5 | total Δ |
|---|---|---:|---:|---:|---:|---:|
| A | `A_lr0p001`<br>TAW_LR0=0.001 | 185630 (+4073) | 181126 (+1960) | 380195 (+998) | 374222 (-1069) | +5962 |
| A | `A_lr0p002`<br>TAW_LR0=0.002 | 184311 (+2754) | 180054 (+888) | 378780 (-417) | 374636 (-655) | +2570 |
| A | `A_lr0p003`<br>TAW_LR0=0.003 | 183652 (+2095) | 179082 (-84) | 378148 (-1049) | 374195 (-1096) | -134 |
| A | `A_lr0p005`<br>TAW_LR0=0.005 | 183263 (+1706) | 180956 (+1790) | 379396 (+199) | 376815 (+1524) | +5219 |
| A | `A_lr0p01`<br>TAW_LR0=0.01 | 183632 (+2075) | 180604 (+1438) | 379266 (+69) | 376703 (+1412) | +4994 |
| A | `A_lr0p02`<br>TAW_LR0=0.02 | 181557 (+0) | 179166 (+0) | 379197 (+0) | 375291 (+0) | +0 |
| B | `B_lr0p0025`<br>TAW_LR0=0.0025 | 183922 (+2365) | 179373 (+207) | 378426 (-771) | 374483 (-808) | +993 |
| B | `B_lr0p0035`<br>TAW_LR0=0.0035 | 183558 (+2001) | 179954 (+788) | 377957 (-1240) | 374357 (-934) | +615 |
| B | `B_lr0p004`<br>TAW_LR0=0.004 | 183434 (+1877) | 181079 (+1913) | 377843 (-1354) | 375426 (+135) | +2571 |
| B | `B_lr0p008`<br>TAW_LR0=0.008 | 183390 (+1833) | 182178 (+3012) | 379299 (+102) | 376476 (+1185) | +6132 |
| B | `B_lr0p015`<br>TAW_LR0=0.015 | 183108 (+1551) | 179496 (+330) | 379254 (+57) | 376576 (+1285) | +3223 |
| B | `B_lr0p03`<br>TAW_LR0=0.03 | 180561 (-996) | 179095 (-71) | 376676 (-2521) | 373585 (-1706) | -5294 |
| B | `B_lr0p05`<br>TAW_LR0=0.05 | 180226 (-1331) | 179857 (+691) | 376894 (-2303) | 374205 (-1086) | -4029 |
| D | `D_b10p95_b20p999_wd0`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0 | 180150 (-1407) | 179247 (+81) | 375513 (-3684) | 371610 (-3681) | -8691 |
| D | `D_b10p95_b20p999_wd1em3`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 180106 (-1451) | 179551 (+385) | 374790 (-4407) | 371744 (-3547) | -9020 |
| D | `D_b10p95_b20p999_wd1em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-4 | 180202 (-1355) | 178860 (-306) | 376541 (-2656) | 371687 (-3604) | -7921 |
| D | `D_b10p95_b20p999_wd1em5`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-5 | 180084 (-1473) | 179365 (+199) | 376531 (-2666) | 371907 (-3384) | -7324 |
| D | `D_b10p95_b20p99_wd0`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.99 TAW_WD=0 | 180678 (-879) | 179602 (+436) | 376913 (-2284) | 373328 (-1963) | -4690 |
| D | `D_b10p95_b20p99_wd1em3`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.99 TAW_WD=1e-3 | 180718 (-839) | 179462 (+296) | 376945 (-2252) | 374132 (-1159) | -3954 |
| D | `D_b10p95_b20p99_wd1em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.99 TAW_WD=1e-4 | 180591 (-966) | 179449 (+283) | 377738 (-1459) | 373131 (-2160) | -4302 |
| D | `D_b10p95_b20p99_wd1em5`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.99 TAW_WD=1e-5 | 180694 (-863) | 179596 (+430) | 377153 (-2044) | 373264 (-2027) | -4504 |
| D | `D_b10p9_b20p999_wd0`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.999 TAW_WD=0 | 180176 (-1381) | 179278 (+112) | 375480 (-3717) | 372220 (-3071) | -8057 |
| D | `D_b10p9_b20p999_wd1em3`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.999 TAW_WD=1e-3 | 180142 (-1415) | 179566 (+400) | 375435 (-3762) | 372496 (-2795) | -7572 |
| D | `D_b10p9_b20p999_wd1em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.999 TAW_WD=1e-4 | 180056 (-1501) | 179592 (+426) | 375424 (-3773) | 372530 (-2761) | -7609 |
| D | `D_b10p9_b20p999_wd1em5`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.999 TAW_WD=1e-5 | 180128 (-1429) | 179280 (+114) | 375476 (-3721) | 372280 (-3011) | -8047 |
| D | `D_b10p9_b20p99_wd0`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.99 TAW_WD=0 | 181254 (-303) | 179456 (+290) | 375499 (-3698) | 373543 (-1748) | -5459 |
| D | `D_b10p9_b20p99_wd1em3`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.99 TAW_WD=1e-3 | 181428 (-129) | 179914 (+748) | 376032 (-3165) | 373594 (-1697) | -4243 |
| D | `D_b10p9_b20p99_wd1em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.99 TAW_WD=1e-4 | 180685 (-872) | 179707 (+541) | 376121 (-3076) | 375833 (+542) | -2865 |
| D | `D_b10p9_b20p99_wd1em5`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.9 TAW_B2=0.99 TAW_WD=1e-5 | 180730 (-827) | 179560 (+394) | 375582 (-3615) | 373555 (-1736) | -5784 |
| F | `F_lr0p025`<br>TAW_LR0=0.025 | 181406 (-151) | 179298 (+132) | 376859 (-2338) | 373254 (-2037) | -4394 |
| F | `F_lr0p035`<br>TAW_LR0=0.035 | 181516 (-41) | 178950 (-216) | 376643 (-2554) | 373731 (-1560) | -4371 |
| F | `F_lr0p04`<br>TAW_LR0=0.04 | 179809 (-1748) | 179144 (-22) | 376648 (-2549) | 372781 (-2510) | -6829 |
| F | `F_lr0p045`<br>TAW_LR0=0.045 | 180356 (-1201) | 179284 (+118) | 376399 (-2798) | 374097 (-1194) | -5075 |
| F | `F_lr0p06`<br>TAW_LR0=0.06 | 180315 (-1242) | 179844 (+678) | 375972 (-3225) | 373715 (-1576) | -5365 |
| F | `F_lr0p07`<br>TAW_LR0=0.07 | 180055 (-1502) | 180646 (+1480) | 376034 (-3163) | 373916 (-1375) | -4560 |
| G | `G_lr00p04_abs0p003`<br>TAW_LR0=0.04 TAW_LR1=0.003 | 180195 (-1362) | 179863 (+697) | 376853 (-2344) | 374119 (-1172) | -4181 |
| G | `G_lr00p04_abs0p01`<br>TAW_LR0=0.04 TAW_LR1=0.01 | 182405 (+848) | 182963 (+3797) | 378108 (-1089) | 376592 (+1301) | +4857 |
| G | `G_lr00p04_flat`<br>TAW_LR0=0.04 TAW_LR1=flat | 201303 (+19746) | 196060 (+16894) | 385454 (+6257) | 386069 (+10778) | +53675 |
| G | `G_lr00p04_r10`<br>TAW_LR0=0.04 TAW_LR1=0.004 | 180255 (-1302) | 180052 (+886) | 377065 (-2132) | 374630 (-661) | -3209 |
| G | `G_lr00p04_r100`<br>TAW_LR0=0.04 TAW_LR1=0.0004 | 179873 (-1684) | 178568 (-598) | 376377 (-2820) | 373165 (-2126) | -7228 |
| G | `G_lr00p04_r20`<br>TAW_LR0=0.04 TAW_LR1=0.002 | 179809 (-1748) | 179144 (-22) | 376648 (-2549) | 372781 (-2510) | -6829 |
| G | `G_lr00p04_r50`<br>TAW_LR0=0.04 TAW_LR1=0.0008 | 179886 (-1671) | 178709 (-457) | 376455 (-2742) | 373418 (-1873) | -6743 |
| H | `H_lr00p035_lr10p00035`<br>TAW_LR0=0.035 TAW_LR1=0.00035 | 181369 (-188) | 178598 (-568) | 376370 (-2827) | 373261 (-2030) | -5613 |
| H | `H_lr00p03_lr10p0003`<br>TAW_LR0=0.03 TAW_LR1=0.0003 | 181220 (-337) | 178661 (-505) | 376466 (-2731) | 373174 (-2117) | -5690 |
| H | `H_lr00p04_lr10p00004`<br>TAW_LR0=0.04 TAW_LR1=0.00004 | 180074 (-1483) | 178750 (-416) | 376338 (-2859) | 373017 (-2274) | -7032 |
| H | `H_lr00p04_lr10p0001`<br>TAW_LR0=0.04 TAW_LR1=0.0001 | 180022 (-1535) | 178746 (-420) | 376357 (-2840) | 373170 (-2121) | -6916 |
| H | `H_lr00p04_lr10p0002`<br>TAW_LR0=0.04 TAW_LR1=0.0002 | 179965 (-1592) | 178751 (-415) | 376358 (-2839) | 373184 (-2107) | -6953 |
| H | `H_lr00p05_lr10p0004`<br>TAW_LR0=0.05 TAW_LR1=0.0004 | 179810 (-1747) | 178876 (-290) | 376559 (-2638) | 373333 (-1958) | -6633 |
| H | `H_lr00p05_lr10p0005`<br>TAW_LR0=0.05 TAW_LR1=0.0005 | 180118 (-1439) | 178860 (-306) | 376469 (-2728) | 373407 (-1884) | -6357 |
| H | `H_lr00p06_lr10p0006`<br>TAW_LR0=0.06 TAW_LR1=0.0006 | 180128 (-1429) | 179280 (+114) | 375476 (-3721) | 372280 (-3011) | -8047 |
| I | `I_lr00p04_lr10p0006`<br>TAW_LR0=0.04 TAW_LR1=0.0006 | 179881 (-1676) | 178751 (-415) | 376453 (-2744) | 373015 (-2276) | -7111 |
| I | `I_lr00p055_lr10p00055`<br>TAW_LR0=0.055 TAW_LR1=0.00055 | 179962 (-1595) | 179478 (+312) | 375551 (-3646) | 372197 (-3094) | -8023 |
| I | `I_lr00p065_lr10p00065`<br>TAW_LR0=0.065 TAW_LR1=0.00065 | 180103 (-1454) | 179330 (+164) | 375774 (-3423) | 372151 (-3140) | -7853 |
| I | `I_lr00p06_lr10p0002`<br>TAW_LR0=0.06 TAW_LR1=0.0002 | 180423 (-1134) | 179622 (+456) | 375634 (-3563) | 372261 (-3030) | -7271 |
| I | `I_lr00p06_lr10p0003`<br>TAW_LR0=0.06 TAW_LR1=0.0003 | 180452 (-1105) | 179189 (+23) | 375452 (-3745) | 372334 (-2957) | -7784 |
| I | `I_lr00p06_lr10p001`<br>TAW_LR0=0.06 TAW_LR1=0.001 | 180178 (-1379) | 179461 (+295) | 375532 (-3665) | 372414 (-2877) | -7626 |
| I | `I_lr00p07_lr10p0007`<br>TAW_LR0=0.07 TAW_LR1=0.0007 | 180523 (-1034) | 179536 (+370) | 375490 (-3707) | 372465 (-2826) | -7197 |
| I | `I_lr00p08_lr10p0008`<br>TAW_LR0=0.08 TAW_LR1=0.0008 | 180101 (-1456) | 179383 (+217) | 375121 (-4076) | 374963 (-328) | -5643 |
| J | `J_lr00p03_wd1em3`<br>TAW_LR0=0.03 TAW_LR1=0.0003 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 179888 (-1669) | 178362 (-804) | 375979 (-3218) | 373173 (-2118) | -7809 |
| J | `J_lr00p04_wd1em3`<br>TAW_LR0=0.04 TAW_LR1=0.0004 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 179920 (-1637) | 179195 (+29) | 376739 (-2458) | 373235 (-2056) | -6122 |
| J | `J_lr00p04_wd1em4`<br>TAW_LR0=0.04 TAW_LR1=0.0004 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-4 | 180041 (-1516) | 179076 (-90) | 376682 (-2515) | 373309 (-1982) | -6103 |
| J | `J_lr00p055_wd1em3` **BEST**<br>TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 179913 (-1644) | 178886 (-280) | 373887 (-5310) | 371894 (-3397) | -10631 |
| J | `J_lr00p06_wd3em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=3e-4 | 180056 (-1501) | 179365 (+199) | 376606 (-2591) | 371711 (-3580) | -7473 |
| J | `J_lr00p06_wd5em4`<br>TAW_LR0=0.06 TAW_LR1=0.0006 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=5e-4 | 180121 (-1436) | 179247 (+81) | 376497 (-2700) | 371728 (-3563) | -7618 |
| K | `K_lr00p0525_lr10p000525_b10p95_wd1em3`<br>TAW_LR0=0.0525 TAW_LR1=0.000525 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 180257 (-1300) | 179354 (+188) | 373802 (-5395) | 371721 (-3570) | -10077 |
| K | `K_lr00p055_lr10p00011_b10p95_wd1em3`<br>TAW_LR0=0.055 TAW_LR1=0.00011 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 180253 (-1304) | 179396 (+230) | 373724 (-5473) | 371873 (-3418) | -9965 |
| K | `K_lr00p055_lr10p000275_b10p95_wd1em3`<br>TAW_LR0=0.055 TAW_LR1=0.000275 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 180103 (-1454) | 179352 (+186) | 373827 (-5370) | 371643 (-3648) | -10286 |
| K | `K_lr00p055_lr10p00055_b10p94_wd1em3`<br>TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.94 TAW_B2=0.999 TAW_WD=1e-3 | 180022 (-1535) | 179430 (+264) | 375160 (-4037) | 371823 (-3468) | -8776 |
| K | `K_lr00p055_lr10p00055_b10p95_wd2em3`<br>TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=2e-3 | 180020 (-1537) | 179377 (+211) | 373861 (-5336) | 371821 (-3470) | -10132 |
| K | `K_lr00p055_lr10p00055_b10p95_wd7em4`<br>TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=7e-4 | 180139 (-1418) | 179372 (+206) | 373949 (-5248) | 371632 (-3659) | -10119 |
| K | `K_lr00p0575_lr10p000575_b10p95_wd1em3`<br>TAW_LR0=0.0575 TAW_LR1=0.000575 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 | 180182 (-1375) | 179468 (+302) | 375206 (-3991) | 370585 (-4706) | -9770 |

Interpretation:

- The upstream fit rewrite completely changed the LR landscape: on v1 the
  tuned LR was 0.003→6e-5; on v2 low LR badly hurts text L1 (e.g. LR0=0.001
  gives 185630, +4073) while high initial LR with a low final LR wins.
- A flat LR is catastrophic (LR0=LR1=0.04: +53,675 bytes total); the decay
  schedule matters more than the exact LR0. LR1≈LR0/100 is best.
- β1=0.95 and WD=1e-3 interact strongly with LR: at LR0=0.055 they give the
  best result, but at LR0=0.04 WD=1e-3 regresses text L5 (+29).
- β2=0.99 is consistently worse than 0.999 on text L5 (up to +436 bytes).

## Pixel-perfect reproduction (v2)

```sh
cd /home/andrew/.local/share/opencode/worktree/8500c0/exp-lr-baseline
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang > /dev/null
cmake --build build -j1
ctest --test-dir build --output-on-failure
mkdir -p arcs_v2
D=/tmp/opencode/tadamw-bench

# BEST CONFIG
TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 \
  build/tools/taw c -l 1 -p mix $D/text1m arcs_v2/ok_text_L1.aw  # 179913
TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 \
  build/tools/taw c -l 5 -p mix $D/text1m arcs_v2/ok_text_L5.aw  # 178886
TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 \
  build/tools/taw c -l 1 -p mix $D/libc1m  arcs_v2/ok_libc_L1.aw  # 373887
TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3 \
  build/tools/taw c -l 5 -p mix $D/libc1m  arcs_v2/ok_libc_L5.aw  # 371894

# NEW SHIPPED BASELINE CONTROL (no env overrides)
build/tools/taw c -l 1 -p mix $D/text1m arcs_v2/base_text_L1.aw  # 181557
build/tools/taw c -l 5 -p mix $D/text1m arcs_v2/base_text_L5.aw  # 179166
build/tools/taw c -l 1 -p mix $D/libc1m  arcs_v2/base_libc_L1.aw  # 379197
build/tools/taw c -l 5 -p mix $D/libc1m  arcs_v2/base_libc_L5.aw  # 375291

stat -c '%n %s' arcs_v2/ok_*.aw arcs_v2/base_*.aw
```

All eight commands were run and the sizes above verified; the best-config
archives are byte-identical to `arcs_v2/J_lr00p055_wd1em3_*` and round-trip
back to the originals via `taw d`.

## Decision on the shipped default

**Not changed.** The tuned setting would shift the baseline for every other
experiment in the portfolio (sum −10,631 bytes, −0.953%), so it is delivered
as a reproducible opt-in via env vars. If a later decision adopts it, the
change in `src/fit.c` is `lr0 = 0.055f, lr1 = 0.00055f, b1 = 0.95f,
wd = 1e-3f` and `ctest` must be re-run.

## STALE v1 results (a427e7f — do not use)

From the pre-rebase branch: best single config was LR 0.003→6e-5 with
β1=0.9, β2=0.999, WD=1e-5, giving text 190610→189020 (L1), 187956→188079
(L5), libc 388792→385681 (L1), 385741→384120 (L5), aggregate −6,199 bytes.
The sibling `exp/muon` control numbers (188984/188087/385592/384178) were a
per-cell mix of LR0=0.002 and LR0=0.003, not one schedule. All of this is
invalid on e64cc83; the full stale table is in `sweep_table.md` and raw data
in `results.tsv`. Branch rebased onto e64cc83 and then onto 1aa3dd7; all
v2 bytes re-verified after the final rebase.

Commits on `exp/lr-baseline`, force-pushed to `myfork` only.
