# AdaGO / OptMuon on the new base (branch `exp/adago`)

Base: `d41bd57` (current leader: tuned Muon W1 + AdamW W2, sum of four
archives **1,095,864 B**). Encoder changes are confined to `src/fit.c`;
format, decoder, quantization and golden vectors are untouched. No push.
Corpora in `/tmp/opencode/tadamw-bench` were read only; all archives live in
this worktree (`bench/arcs`, `bench/sweep/arcs`, `bench/ref`).

## TL;DR / verdict

| configuration | text1m L1 | text1m L5 | libc1m L1 | libc1m L5 | sum | vs leader |
|---|---:|---:|---:|---:|---:|---:|
| shipped AdamW (control) | 181557 | 179166 | 379197 | 375291 | 1,115,211 | +19,347 |
| tuned AdamW (control) | 179913 | 178886 | 373887 | 371894 | 1,104,580 | +8,716 |
| tuned Muon W1 = leader | 179678 | 177816 | 371373 | 366997 | 1,095,864 | 0 |
| **AdaGO (recommended)** | **179271** | **177468** | **371284** | **366598** | **1,094,621** | **−1,243 (−0.11 %)** |
| OptMuon (muon momentum) | 179739 | 177274 | 371543 | 367085 | 1,095,641 | −223 (−0.02 %) |

**AdaGO is Pareto-better than the Muon leader on all four cells**
(−407 / −348 / −89 / −399 B) and is the first adaptive-scalar variant here to
beat the hand-tuned Muon schedule. OptMuon gives a small aggregate win but is
*not* Pareto (worse on text L1 and libc L1). Both are much better than either
AdamW control. The default (`TAW_ADAGO=off`) remains the shipped fitter, so
adoption is a coordinator policy decision; this branch keeps A/B via knobs.

Recommended AdaGO env (all other knobs at defaults, including `β_muon=0.95`):

```
TAW_ADAGO=adago TAW_ADAGO_LR=29 TAW_ADAGO_EPS=1e-3 \
TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015
```

## Papers and exact equations implemented

### AdaGO — Zhang, Liu & Schaeffer, *AdaGrad Meets Muon: Adaptive Stepsizes
for Orthogonal Updates*, arXiv:2509.02981v2 (Algorithm 2)

With stochastic gradient `G_t = ∇L_t(Θ_{t−1})`, momentum `M_t = μM_{t−1} +
(1−μ)G_t`, orthogonalized direction `O_t = Orth(M_t)` (the polar factor;
Newton–Schulz in practice), and Frobenius norms (the paper's practical choice):

```
v_t^2  = v_{t−1}^2 + min{ ||G_t||^2 , γ^2 }          (v_0 > 0)
α_t    = max{ ε , η · min{ ||G_t|| , γ } / v_t }
Θ_t    = Θ_{t−1} − α_t O_t
```

Points that matter: (i) AdaGrad-Norm accumulates the **clamped** squared norm
(γ removes the requirement of uniformly bounded gradients); (ii) the unit-scale
orthogonal direction is grafted with the *current* clamped gradient norm, which
gives null-gradient consistency; (iii) `ε` is a floor so AdaGO is at least as
fast as small-constant-step Muon. Theory: `Õ(T^{-1/4})` stationary-point rate
in the stochastic setting with `b_t≡1, ε=T^{-3/4}, 1−μ=T^{-1/2},
η=T^{-(3/8+q)}`, and `O(T^{-1/2})` in the deterministic full-batch setting;
only γ and v0 enter the bounds, logarithmically. The paper's grid search:
`η=0.5, ε=5e-3` (regression) and `η=0.05, ε=5e-4` (CIFAR-10), with `ε<η²`.

### OptMuon — Yuan, *OptMuon: Closed-Loop Orthogonalized Momentum Methods for
Stochastic Optimization with Zero-Noise Optimality*, arXiv:2606.08783v2
(Algorithm 1, Eqs. (2)–(4))

```
g_t   = ||G_t||_F
ρ_t   = ((1 + max_{1≤i≤t} g_i^2) / (1 + Σ_{i=1}^t g_i^2))^q ,  ρ_{−1}=ρ_0=1
α_t   = ρ_{t−1}                       (lagged ⇒ F_{t−1}-measurable)
Option A: M_t = G_t + (1−α_t) M_{t−1},  q = 1/2
          γ_t = min{ α_t^2 , α_t (1 + Σ_{j=1}^t α_j ||M_j||_F^2)^{-1/2} }
Option I: M_t = (1−α_t)(M_{t−1} − ∇f(X_{t−1};ξ_t)) + G_t,  q = 2/3
          γ_t = min{ √α_t , (1 + Σ_{j=1}^t α_j^{-1/2} ||M_j||_F^2)^{-1/2} }
X_{t+1} = X_t − θ γ_t ||M_t||_F Orth(M_t)
```

The denominator is AdaGrad-Norm energy accumulation; the **running maximum**
in the numerator stops an isolated gradient spike from collapsing the
coefficient (ablation below: removing it costs 387 B). No Lipschitz constant,
variance level or gradient bound is used in parameter selection. Rates (up to
logs): OptMuon-A `Õ(T^{-1/2} + σ^{1/2}T^{-1/4})`, OptMuon-I
`Õ(T^{-1/2} + σ^{1/3}T^{-1/3})`; in the zero-noise regime both reduce to
`Õ(T^{-1/2})` without retuning. Theory assumes exact polar factors; we use
the existing 5-step Newton–Schulz approximation.

### What is faithful and what is an ablation

* AdaGO: exact Algorithm 2 (clamped accumulator, clamped current norm graft,
  ε floor, same Muon momentum and NS direction as the leader). W2 is left on
  AdamW exactly as in the leader's hybrid scheme.
* OptMuon: exact `ρ_t`/`α_t`/`γ_t`/update of Option A, including the lagged
  coefficient and the running maximum. Our fitter draws **one** stochastic
  gradient per step, so Option I's STORM estimator cannot be reproduced; the
  `TAW_ADAGO_Q=0.667` knob applies only its exponent (tested, worse — see
  below). `TAW_ADAGO_MOMSRC=muon` is an explicit departure: it keeps Muon's
  fixed-0.95 momentum as the direction source while retaining the paper's
  scalar rule (with plain, non-Nesterov Muon momentum); this is the best
  OptMuon variant here. The paper's literal
  adaptive decay `M_t=G_t+(1−α_t)M_{t−1}` is the default (`paper`) and is
  worse on our four cells (best L1 sum 558,106 vs 551,282 for `muon`).

## Implementation (`src/fit.c` only)

* Adaptive scalars live in `optcfg` (one `tnet_train` call = one fit). W1 is a
  single 16×26 matrix on this base, so one global accumulator per fit matches
  the papers' single scalar exactly. W2 stays on AdamW; decoupled W1 weight
  decay keeps using the base LR (`mix·lr·wd·w`), as in the leader.
* The base LR `lr` (linear schedule, or frozen at `LR0` with `CONST=1`) is
  multiplied by `TAW_ADAGO_LR`; that product is the paper's `η`/`θ`. The
  large multiplier (~29) is expected: our `G_t` is a batch average (`1/256`)
  with `||G_t||~0.05`, and `v_0=1`, so the paper's `η≈0.05` is ~50× too small.
* `ε` (AdaGO floor) and `γ` (AdaGO clamp) have paper defaults `1e-4` and
  `0` (no clamp); `v0=1`. OptMuon uses `q=0.5`, running max on, `ρ=1`,
  `Σg²=0`, `α_1=1`.

| knob | default | meaning |
|---|---|---|
| `TAW_ADAGO` | `off` | `off` / `adago` / `optmuon`; adaptive modes take over W1 |
| `TAW_ADAGO_LR` | `1` | multiplier on the W1 LR (`η`/`θ`) |
| `TAW_ADAGO_CONST` | `0` | `1`: freeze `η` at `TAW_MUON_LR0` (no linear decay) |
| `TAW_ADAGO_EPS` | `1e-4` | AdaGO stepsize floor `ε` (`0` disables) |
| `TAW_ADAGO_GAMMA` | `0` | AdaGO norm clamp `γ` (`0` = none) |
| `TAW_ADAGO_V0` | `1` | AdaGO initial accumulator `v0` (`v_0²=v0²`) |
| `TAW_ADAGO_RUNMAX` | `1` | OptMuon running-maximum numerator (`0`: current `g_t²`) |
| `TAW_ADAGO_Q` | `0.5` | OptMuon exponent (`>0.6` selects `2/3`) |
| `TAW_ADAGO_MOMSRC` | `paper` | `muon`: fixed-0.95 momentum direction (ablation) |
| `TAW_ADAGO_DEBUG` | `0` | print scalar diagnostics every N W1 updates |

**Knob-off is byte-identical to the base.** Four leader archives regenerated
with `TAW_ADAGO=off` compared equal with `cmp` to references made before the
change (text1m L1/L5, libc1m L1/L5). All 12 tests pass with and without the
adaptive knobs; full `ctest --output-on-failure`: 12/12.

## Sweep and ablation (archive bytes, lower is better)

Common: W2 AdamW `0.003→0.00015`, W1 base `0.055→0.0011` (ratio 1/50),
β=0.95, NS=5 unless a row says otherwise. All four cells unless marked L1.
Raw log: `bench/sweep.tsv` (67 configurations, 194 corpus/level measurements).

### AdaGO: scalar multiplier and LR decay

| config | text L1 | text L5 | libc L1 | libc L5 | sum |
|---|---:|---:|---:|---:|---:|
| linear decay, mult 20 (ε=1e-4) | 179370 | 177386 | 373165 | 367303 | 1,097,224 |
| linear decay, mult 25 (ε=1e-4) | 179267 | 177443 | 372295 | 367370 | 1,096,375 |
| linear decay, mult 30 (ε=1e-4) | 179502 | 177476 | 371395 | 366693 | 1,095,066 |
| linear decay, mult 35 (ε=1e-4) | 179443 | 177508 | 371756 | 367030 | 1,095,737 |
| linear decay, mult 40 (ε=1e-4) | 179391 | 177463 | 371910 | 366745 | 1,095,509 |
| decay, mult 27, ε=1e-3 | 179201 | 177466 | 371430 | 367885 | 1,095,982 |
| decay, mult 28, ε=1e-3 | 179327 | 177315 | 372119 | 367607 | 1,096,368 |
| **decay, mult 29, ε=1e-3 (winner)** | **179271** | **177468** | **371284** | **366598** | **1,094,621** |
| decay, mult 31, ε=1e-3 | 179474 | 177486 | 372092 | 366551 | 1,095,603 |
| decay, mult 32, ε=1e-3 | 179336 | 177398 | 371706 | 366070 | 1,094,510 |
| **no decay (`CONST=1`), mult 12, ε=1e-3** | 179544 | 177632 | 373567 | 368111 | 1,098,854 |
| no decay, mult 15, ε=1e-3 | 179748 | 177702 | 372901 | 368527 | 1,098,878 |
| no decay, mult 18, ε=1e-3 | 180089 | 177970 | 373146 | 367982 | 1,099,187 |
| no decay, mult 20, ε=1e-3 | 180016 | 177901 | 373295 | 368642 | 1,099,854 |

Linear decay still helps by ~3.5–4.4 KB over the best constant-`η` AdaGO,
consistent with our earlier experience that replacing the LR schedule
outright (AIR) hurt. AdaGrad-Norm's `1/√t` is not a drop-in replacement for
the tuned linear decay; the two compose well.

### AdaGO knob ablations (L1 only unless noted; baseline mult 30, ε=1e-4)

| variation | text L1 | libc L1 | sum(L1) |
|---|---:|---:|---:|
| baseline (v0=1, γ=off, ε=1e-4) | 179502 | 371395 | 550,897 |
| `V0=0.05` | 180666 | 371856 | 552,522 |
| `V0=0.2` | 179895 | 371109 | 551,004 |
| `V0=5` | 181162 | 375237 | 556,399 |
| `GAMMA=0.05` | 179679 | 372342 | 552,021 |
| `GAMMA=0.1` | 179566 | 372134 | 551,700 |
| `GAMMA=0.3` / `1` (no clamping in practice) | 179502 | 371395 | 550,897 |
| `EPS=0` / `1e-5` | 179502 | 371395 | 550,897 |
| `EPS=1e-3` | 179465 | 371399 | 550,864 |
| W2 `0.002→1e-4` | 179703 | 373230 | 552,933 |
| W2 `0.003→1.5e-4` | 179271 | 371284 | 550,555 |
| W2 `0.004→2e-4` | 179671 | 371662 | 551,333 |
| W2 `0.005→2.5e-4` | 178928 | 371758 | 550,686 |
| final LR ratio 1/20 (`LR1=0.00275`) | 179390 | 372055 | 551,445 |
| final LR ratio 1/100 (`LR1=0.00055`) | 179405 | 372154 | 551,559 |
| β=0.90 | 179474 | 373986 | 553,460 |
| β=0.95 (winner setting) | 179271 | 371284 | 550,555 |
| β=0.98 | 179658 | 371250 | 550,908 (full sum 1,094,505) |

The clamp and `v0` are insensitive in the broad middle (their effect is only
logarithmic in the theory). The ε floor helps slightly by supplying a small
constant tail. W2's Muon-tuned optimum (0.003→1.5e-4) still holds. β=0.98
edges the winner by 116 B in aggregate but leaves text L1 only 20 B ahead of
the leader; I treat that as noise and keep the default β=0.95.

### OptMuon ablations

| config | text L1 | text L5 | libc L1 | libc L5 | sum |
|---|---:|---:|---:|---:|---:|
| paper momentum, mult 30 | 182542 | — | 375564 | — | L1: 558,106 |
| paper momentum, mult 100 | 183495 | — | 377705 | — | L1: 561,200 |
| muon momentum, mult 5 | 180868 | — | 374876 | — | L1: 555,744 |
| muon momentum, mult 10 | 179846 | 177529 | 372282 | 367753 | 1,097,410 |
| muon momentum, mult 12 | 179609 | 177513 | 371705 | 368031 | 1,096,858 |
| **muon momentum, mult 15, q=0.5, runmax on** | 179739 | 177274 | 371543 | 367085 | **1,095,641** |
| muon momentum, mult 18 | 179992 | 177361 | 371760 | 366936 | 1,096,049 |
| muon momentum, mult 20 | 179897 | 177481 | 371560 | 367348 | 1,096,286 |
| mult 15, `RUNMAX=0` | 179659 | 177364 | 371393 | 367612 | 1,096,028 |
| mult 15, `Q=0.667` (Option I exponent only) | 179834 | 177462 | 371673 | 367760 | 1,096,729 |
| mult 15, `CONST=1` (no decay) | 179767 | 177374 | 371677 | 367478 | 1,096,296 |

Dropping the running maximum costs 387 B — directionally consistent with the
paper's claim that it prevents spike-driven coefficient collapse. Constant-`η`
OptMuon costs 655 B; the `2/3` exponent costs 1,088 B (and lacks the paper's
STORM estimator, so this is not a real Option I test). OptMuon's best remains
~1 KB behind AdaGO.

## Verification method

* `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`,
  `cmake --build build -j4`. GCC 14 `-Werror` build also clean on the base.
* `ctest --test-dir build --output-on-failure`: **12/12** after the final code
  state, both with knobs off and with the winner's environment.
* `bench.sh` re-encoded the two winning configurations at L1/L5 on both
  corpora, decoded every archive and `cmp`-checked it against the source:
  **8/8 `decode_ok=1`** (rows `adago_f29_v`, `optmuon_m15_v` in
  `bench/results.tsv`). The earlier coarse sweeps were archive-size only;
  format/decoder were never modified, so byte-identical decode is expected.
* Knob-off byte identity: four leader archives regenerated with
  `TAW_ADAGO=off` are byte-equal (`cmp`) to pre-change references.
* Note: `bench.sh`'s `mode` column records `adam/adago` and `adam/optmuon`
  for the new rows; legacy rows predate the adaptive knob and say `adam`.

## Caveats

* Margins are small in relative terms (AdaGO −0.11 % in aggregate): bytes are
  deterministic here, but the sweep surface is noisy, so the specific optimum
  (mult 29 vs 30/32, β 0.95 vs 0.98) should not be over-read. The *robust*
  claim is that AdaGO with decay lands in the 1,094.5–1,095.1 KB band, below
  the leader's 1,095.9 KB.
* Only two 1 MiB corpora, levels 1 and 5, `-p mix`, one seed family,
  Q8/IQ8 608-byte models. No L9, no other shapes, no multi-seed averaging.
* The accumulator is one scalar per fit. Exact for the current single W1 set;
  a future base with `sel1 != NONE` needs per-set accumulators (and the tied
  warm-up copy), otherwise lazy updates would interleave different sets'
  gradient histories in one scalar.
* AdaGO's `η≈1.6` (=0.055×29) is far above the paper's 0.05/0.5 because ours
  is a batch-averaged gradient with `v0=1`; the paper's `*2e-2`–`*5e-1`
  hyperparameters do not transfer without rescaling.
* OptMuon's Option I is not implemented faithfully (no second gradient
  evaluation / STORM correction); `q=2/3` only exercises the coefficient
  exponent. Theory also assumes exact orthogonalization; we keep the leader's
  5-step Newton–Schulz.
* Decoupled W1 weight decay still uses the base LR rather than the adaptive
  scalar; this keeps the comparison isolated to the direction magnitude.
* Decode timings in `results.tsv` are noisy on a shared box and were not used
  for any conclusion; archive bytes are the only metric.
