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

# NAMO / NAMO-D (branch `exp/namo`, rebased onto merged `origin/main` `1aa3dd7`)

Original base `d41bd57`; this section is rebased on top of the merged MALT/
MALTER + tuned-Muon upstream. The upstream MALT/MALTER implementation and its
results (section above) are preserved verbatim; the only delta below is the
NAMO/NAMO-D optimizer (`TAW_HYBRID=namo|namod`, `TAW_NAMO_*`). Format,
decoder, quantization and golden vectors are untouched. Corpora in
`/tmp/opencode/tadamw-bench` were read only. 84 new NAMO/C3 configurations
(336 encodes/decodes) are in `bench/results.tsv` alongside the 48 upstream
MALT rows and the base controls; every decode was `cmp`-verified
(`decode_ok=1` for all 600 retained measurements).

## TL;DR / verdict

**Both paper algorithms beat the tuned-Muon and tuned-AdamW controls**, but
after the rebase the merged MALTER leader (1,093,945 B, section above) is
80 B ahead of NAMO-D. Best configurations (sum of the four L1+L5 archive
sizes, lower is better):

| config | text1m L1 | text1m L5 | libc1m L1 | libc1m L5 | sum | vs tuned Muon |
|---|---:|---:|---:|---:|---:|---:|
| controls: shipped AdamW | 181557 | 179166 | 379197 | 375291 | 1,115,211 | +19,347 |
| controls: tuned AdamW¹ | 179913 | 178886 | 373887 | 371894 | 1,104,580 | +8,716 |
| controls: tuned Muon² | 179678 | 177816 | 371373 | 366997 | 1,095,864 | 0 |
| upstream MALTER (merged) | 179405 | 177138 | 370898 | 366504 | 1,093,945 | −1,919 |
| **NAMO** (scalar)³ | 179273 | 177228 | 370921 | 367034 | **1,094,456** | **−1,408 (−0.13 %)** |
| **NAMO-D** (diagonal)⁴ | 179144 | 177145 | 371009 | 366727 | **1,094,025** | **−1,839 (−0.17 %)** |

vs tuned AdamW: NAMO −10,124 B (−0.92 %), NAMO-D −10,555 B (−0.96 %).
vs shipped AdamW: −20,755 / −21,186 B (−1.86 % / −1.90 %).
vs the merged MALTER leader: NAMO +511 B, NAMO-D +80 B (NAMO-D wins
libc1m L5 by 223 B but loses the other three cells).
NAMO-D is Pareto-better than tuned AdamW on all four cells and better than
tuned Muon on three of four (libc1m L5 is +270 B); NAMO likewise loses only
libc1m L5 (+37 B). The two NAMO variants are within 431 B of each other — the
paper's diagonal gain over the scalar version is visible but small here.

Reproduce the two winners:

```sh
# NAMO
TAW_HYBRID=namo TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.006 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01 ./bench.sh
# NAMO-D (best)
TAW_HYBRID=namod TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.015 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01 \
TAW_NAMO_BETA=0.999 TAW_NAMO_CLAMP=0.9 ./bench.sh
```

¹ `TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_WD=0.001` (from
`exp/lr-baseline`, reproduced exactly here).
² `TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015` (reproduced exactly here, and
byte-identical archives to the base branch).
³ `TAW_HYBRID=namo`, μ1=0.95, μ2=0.99, λ=0.01, no Nesterov, NS5.
⁴ `TAW_HYBRID=namod`, μ1=0.95, μ2=0.999, λ=0.01, c=0.9, no Nesterov, NS5.

## Paper summary (arXiv:2602.17080v2, Zhang / Liu / Schaeffer)

*Title:* "Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized
Momentum" ([abs](https://arxiv.org/abs/2602.17080),
[HTML](https://arxiv.org/html/2602.17080v2), code
[github.com/minxin-zhg/namo](https://github.com/minxin-zhg/namo)).

The paper's premise: an optimizer should combine a good *deterministic*
direction (Muon's orthogonalized momentum `Orth(M_t)`, the spectral-norm
steepest-descent direction) with an *Adam-type noise-adaptation* mechanism.
NAMO ("Norm-Based Adaptive Moment Estimation with Orthogonalized Momentum")
keeps Muon's direction and rescales it with a single norm-based SNR; NAMO-D
adds a right-multiplied clamped diagonal for column-wise ("neuron-wise")
adaptation, motivated by near block-diagonal Hessians. Under smoothness +
bounded-noise assumptions both obtain the optimal deterministic O(T^-1/2)
rate and a stochastic O(T^-1/4 + σ b^-1/4 T^-1/8) rate that becomes optimal
for large batches.

GPT-2 (124M / 355M) on OpenWebText, 4×H100, 2000-step warmup then constant
LR: AdamW β=(0.9,0.95), Muon β=0.95, NAMO/NAMO-D μ=(0.95,0.99), λ=0.01
for all; grid-searched optimal η: AdamW 0.0013/0.0009, Muon 0.0013/0.0009,
NAMO 0.012/0.007, NAMO-D 0.009 (c=0.1) / 0.009 (c=0.9). Validation losses
(124M / 355M): AdamW 3.0643 / 2.9914; Muon 3.0435 / 2.9684;
NAMO 3.0351 / 2.9516; NAMO-D 3.0246 / 2.9507. NAMO-D's extra gain appears
with longer training (50K steps) and with the clamped c tuned.

## Exact update rules implemented

Notation: `G_t` = minibatch gradient matrix (our `g` already divided by the
batch size), `M_t = μ1 M_{t-1} + (1−μ1) G_t`, `v_t = μ2 v_{t-1} +
(1−μ2)||G_t||_F²`, bias-corrected `M̂_t = M_t/(1−μ1^t)`, `v̂_t =
v_t/(1−μ2^t)`, `ε_t = ε/√(1−μ2^t)`, `O_t ≈ Orth(M_t)` via Newton–Schulz
(5 steps, the existing Muon polynomial). Paper equations (3), Algorithm 1,
Algorithm 2 and its GPT-2 weight-decay variant:

```
NAMO:     α_t   = ||M̂_t||_F / (√v̂_t + ε_t)
          Θ_t   = Θ_{t-1} − η α_t (O_t + λ Θ_{t-1})

NAMO-D:   [v_t]_j = μ2 [v_t−1]_j + (1−μ2) ||[G_t]_{:j}||²
          d_t     = N_c(M̂_t) ⊘ (√v̂_t + ε_t),     [N_c(A)]_j = ||A_{:j}||
          d̄_t     = ||d_t||_1 / n
          d̃_t     = min{ max{ d_t, c d̄_t 1 }, d̄_t / c },    c ∈ (0,1]
          Θ_t     = Θ_{t-1} − η (O_t + λ Θ_{t-1}) diag(d̃_t)
```

Implementation mapping (important for exactness):

* The existing Muon momentum is `m_t = μ1 m_{t-1} + g_t`, i.e. the paper's
  `M_t/(1−μ1)`. Orthogonalization is scale invariant, so `O_t` is identical.
  Folding the bias corrections into one constant gives
  `mscale = (1−μ1)/(1−μ1^t) · √(1−μ2^t)` and
  `α_t = mscale·||m_t||_F/(√v_t + ε)`, `d_j = mscale·||(m_t)_{:j}||/(√v_j+ε)`,
  which is algebraically the paper's `α_t` / `d_j`.
* The adaptive scale multiplies the **decoupled weight decay** as in the
  paper's GPT-2 experiments (`ηα(O+λW)`, `η(O+λW)D`), unlike our Muon path
  where WD is scaled by `η` only.
* NAMO defaults to **no Nesterov extrapolation** (paper uses `M_t`, not
  `μ1 M_t + g_t`) and no `gain·√(maxdim)` factor: the scalar `α_t` carries
  the update magnitude. `TAW_NAMO_SCALE` multiplies the whole NAMO step
  (default 1.0), `TAW_MUON_NS`/`TAW_MUON_MOM` are reused for NS steps and μ1.
* `TAW_NAMO_CLAMP=0` is our extension meaning "no clamp" (paper restricts
  `c ∈ (0,1]`); `c=1` forces all diagonal entries to `d̄`, not "off".
* W2 (the 8 vector sets) stays on the untouched AdamW path; only W1's single
  16×26 matrix uses NAMO/NAMO-D. `v2s` (one scalar per W1 set), `v2c`
  (per-column v) and the optional C3 sign EMA are new W1-only state in
  `tnet`, allocated/copied at untie like the other momentum buffers.

## Environment knobs (all unset = shipped AdamW, bit-identical)

| variable | default | meaning |
|---|---|---|
| `TAW_HYBRID` | `adam` | `adam`, `muon`, `blend`, `alt`, `route`, **`namo`**, **`namod`** (`namo-d` also accepted) |
| `TAW_MUON_LR0` / `TAW_MUON_LR1` | `0.01` / `LR0/20` | W1 LR endpoints, also used by NAMO modes (winner: 0.3 → 0.015) |
| `TAW_VEC_LR0` / `TAW_VEC_LR1` | `TAW_LR0` / `TAW_LR1` | W2 AdamW LR endpoints (winner: 0.003 → 1.5e-4) |
| `TAW_NAMO_BETA` | `0.99` | μ2 (second-moment decay for the norm/column norm) |
| `TAW_NAMO_SCALE` | `1.0` | multiplier of the adaptive W1 step |
| `TAW_NAMO_CLAMP` | `0.1` | NAMO-D clamp c; `0` disables clamping |
| `TAW_NAMO_WD` | `1e-3` | W1 λ (adaptive-scaled); paper uses 0.01 |
| `TAW_NAMO_D_AXIS` | `col` | `row` = ablation: scale rows instead of right-diagonal columns |
| `TAW_NAMO_GATE` | `none` | `c3` = sign-EMA shrink gate on W1 (comparison with sibling work) |
| `TAW_NAMO_GATE_BETA` / `_P` | `0.9` / `1` | C3 EMA decay / exponent |
| `TAW_MUON_MOM` / `_NS` / `_NEST` | `0.95` / `5` / `1` (0 for NAMO modes) | reused Muon internals |

Knob-off check after the rebase: built `origin/main` (`1aa3dd7`) separately
and compared archives on all four corpora/levels. Plain defaults,
`TAW_HYBRID=muon` with the tuned-Muon LRs, and the upstream MALTER winner all
produced **byte-identical** archives on both builds (12/12 `cmp` checks).
The published NAMO (179273/177228/370921/367034) and NAMO-D
(179144/177145/371009/366727) results also reproduced exactly after the
rebase (tags `rebase_namo_winner` / `rebase_namod_winner`).

## Sweep and ablations

All rows are archive-byte sums over the four cells; full per-cell numbers and
environment strings are in `bench/results.tsv` (tag = first column). Archive
bytes are deterministic; decode timings are discarded as noise.

**NAMO LR sweep** (μ2=0.99, λ=1e-3, W2 0.003→1.5e-4, lr1=lr0/50):

| lr0 | sum |
|---:|---:|
| 0.02 | 1,106,760 |
| 0.04 | 1,103,342 |
| 0.055 | 1,101,470 |
| 0.08 | 1,099,333 |
| 0.12 | 1,096,771 |
| 0.20 | 1,095,754 |
| 0.25 | 1,096,003 |
| **0.30** | **1,095,122** |
| 0.35 | 1,095,886 |
| 0.40 | 1,096,736 |

The optimum is ~5× the tuned-Muon LR because our Muon's gain·√maxdim ≈ 1.02
is absent from NAMO (paper: η_NAMO ≈ 9× η_Muon for square GPT-2 matrices).
At lr0=0.055, `TAW_NAMO_SCALE=2` gave 1,098,535 (vs 1,101,470 at scale 1),
and Nesterov-on gave 1,098,941 (vs 1,099,333) — neither closes the LR gap.

**Weight decay is the biggest single lever** (NAMO, lr0=0.3):

| λ | sum |
|---:|---:|
| 0 | 1,095,920 |
| 1e-3 | 1,095,122 |
| **0.01** | **1,094,456** |
| 0.02 | 1,095,231 |
| 0.03 | 1,095,740 |

λ=0.01 is exactly the paper's setting; applying it *inside* the adaptive
scale matters (the shipped Muon path uses 1e-3 unscaled).

**NAMO-D clamp sweep** (lr0=0.3, lr1=0.006, λ=1e-3, μ2=0.99):

| c | sum |
|---:|---:|
| 0 (no clamp) | 1,096,044 |
| 0.1 (paper, 124M) | 1,096,006 |
| 0.4 | 1,095,937 |
| 0.6 | 1,096,298 |
| 0.75 | 1,095,748 |
| 0.9 (paper, 355M) | **1,095,311** |
| 0.95 | 1,095,113 |
| 0.99 | 1,095,248 |

With λ=0.01 and μ2=0.999 the optimum stays near c=0.9: c=0.75 → 1,095,545,
c=0.9 → 1,094,679, c=0.95 → 1,095,346, c=0.99 → 1,095,023.
A row-wise diagonal ablation (`TAW_NAMO_D_AXIS=row`, c=0.9) gave 1,094,304 —
slightly worse than the paper's right-diagonal, which we keep as default.
Strong clamping (c≈0.9) is best: our 16×26 W1 benefits from staying close to
the scalar direction; the paper likewise needed c=0.9 for the larger model.

**Other NAMO-D ablations** (around lr0=0.3, λ=0.01, c=0.9, μ2=0.999,
lr1=0.006 unless noted):

| change | sum |
|---|---:|
| **reference (lr1=0.006)** | **1,094,218** |
| lr1=0.015 (1/20) | **1,094,025** ← best overall |
| lr1=0.020 (1/15) | 1,094,426 |
| lr1=0.030 (1/10) | 1,094,686 |
| lr1=0.012 (1/25) | 1,094,995 |
| lr0=0.25 / 0.35 (lr1=lr0/50) | 1,094,649 / 1,094,657 |
| μ2=0.99 / 0.995 / 0.9999 | 1,094,679 / 1,094,541 / 1,095,319 |
| λ=0.005 / 0.015 / 0.02 | 1,094,504 / 1,095,498 / 1,095,627 |
| μ1=0.90 / 0.98 | 1,096,574 / 1,094,969 |
| NS=3 / 7 | 1,095,222 / 1,094,465 |
| Nesterov on | 1,094,865 |
| W2 0.004→2e-4 | 1,095,137 |
| c=0.85 / 0.95 with lr1=0.015 | 1,095,139 / 1,094,822 |

**Comparison with our C3 shrink gate.** The sign-EMA coordinate shrink from
`exp/eps-gate` (`β=0.9`, `p=1`, shrink-only, applied to W1) was ported behind
`TAW_NAMO_GATE=c3` so that both mechanisms are measured on the *same* base,
same corpora and same W2 schedule:

| configuration | sum | vs its ungated counterpart |
|---|---:|---:|
| tuned Muon + C3 (p=1) | 1,104,807 | +9,943 vs 1,095,864 |
| tuned Muon + C3 (p=2) | 1,108,953 | +13,089 |
| shipped AdamW + C3 (W1) | 1,118,968 | +3,757 vs 1,115,211 |
| NAMO (best) + C3 (p=1) | 1,101,714 | +7,258 vs 1,094,456 |
| NAMO-D (best) + C3 (p=1) | 1,101,596 | +7,571 vs 1,094,218 |
| NAMO-D (best) + C3 (p=2) | 1,107,587 | +13,569 |

The per-coordinate sign gate is strongly counterproductive here: on this
single 16×26 matrix the noise-adaptive information is better exploited at the
granularity of the whole matrix (NAMO) or of a few clamped columns (NAMO-D)
than by shrinking individual weights. This is consistent with the paper's
argument that preserving the orthogonalized direction (or keeping `D_t`
well-conditioned) matters.

## Caveats

* This is a mechanical reproduction of the *update rules*, not of GPT-2
  pretraining: our "loss" is compressed bits of a 608-byte mixer fitted on
  1 MiB corpora, W1 is one 16×26 matrix, W2 remains AdamW, the LR schedule
  is linear decay (the paper uses warmup + constant), and orthogonalization
  is the same 5-step Newton–Schulz approximation as our Muon baseline. The
  0.13–0.17 % margins are real and deterministic (same seed/build), but are
  far smaller than the paper's validation-loss gains and may not transfer.
* Our μ2=0.999 for the winning NAMO-D differs from the paper's 0.99; the
  paper's exact NAMO-D setting (μ2=0.99, λ=0.01, c=0.1, lr 0.3→0.006) gave
  1,095,898 here, i.e. +34 B vs tuned Muon (essentially tied); with our
  default λ=1e-3 it gave 1,096,006. The best c=0.9 matches the
  paper's 355M sweep, λ=0.01 matches both models, and c=0.1 (124M) was worse
  on this tiny W1.
* No single setting is Pareto-best: both winners are slightly worse than
  tuned Muon on libc1m L5. Margins between neighbouring sweep points
  (hundreds of bytes) should not be over-interpreted as smooth trends.
* All archives (upstream MALT + NAMO) were decoded and `cmp`-checked against
  the source (600/600 retained rows `decode_ok=1`); format/decoder were never
  changed, so
  byte-identical decode is expected and observed. Decode timings on the
  shared 12-core box are noise and are not used for any conclusion.
* `ctest` passes 12/12 both with defaults and with the winning environment;
  the code builds cleanly with clang and with GCC `-Werror`.
* The shipped default (`TAW_HYBRID=adam`) is untouched and bit-identical to
  the base; adopting NAMO/NAMO-D is a policy decision for the coordinator.
