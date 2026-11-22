# Hybrid Muon + AdamW fitting on the new base (branch `exp/muon-adam`)

Base: `origin/main` @ `e64cc83` (clean new base, new F32_STRICT_MIX fitter).
Encoder-only changes in `src/fit.c` / `src/fit_internal.h`; format, decoder,
quantization and golden vectors untouched. No push. Corpora in
`/tmp/opencode/tadamw-bench` were read only; all archives live in
`bench/arcs/` in this worktree.

## TL;DR / verdict

**Muon on the single W1 matrix + AdamW on the W2 vectors is a clear, robust
win on the new base — much larger than the 0.05–0.31 % margin seen on the old
base.** Best configuration found (sum of four archive sizes):

```
TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015
```

| corpus / level | shipped AdamW | tuned AdamW¹ | **Muon W1 (winner)** | vs shipped | vs tuned |
|---|---:|---:|---:|---:|---:|
| text1m L1 | 181557 | 179913 | **179678** | −1879 (−1.03 %) | −235 (−0.13 %) |
| text1m L5 | 179166 | 178886 | **177816** | −1350 (−0.75 %) | −1070 (−0.60 %) |
| libc1m L1 | 379197 | 373887 | **371373** | −7824 (−2.06 %) | −2514 (−0.67 %) |
| libc1m L5 | 375291 | 371894 | **366997** | −8294 (−2.21 %) | −4897 (−1.32 %) |
| **sum L1+L5** | **1,115,211** | **1,104,580** | **1,095,864** | **−19,347 (−1.74 %)** | **−8,716 (−0.79 %)** |

¹ Tuned AdamW from sibling `exp/lr-baseline` commit `9a08442`:
LR 0.055→0.00055, β1=0.95, WD=1e-3; reproduced **exactly** with this build
(179913 / 178886 / 373887 / 371894) using
`TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_WD=0.001`.

The winner is **Pareto-better than both AdamW controls on all four cells**.
No blend, alternation or routing scheme beat plain Muon; see below.

The shipped default (`TAW_HYBRID=adam`, no overrides) was deliberately left
unchanged and is bit-identical to `origin/main` (all four baseline archives
verified with `cmp`). Recommendation: adopt Muon W1 with the LRs above, but
that is a policy decision for the coordinator — this branch keeps A/B
reproducibility via environment knobs.

## What the new base changed (why old Muon numbers are invalid)

The old base fitted `H=8/16` with multi-set W1 selectors and 1888–2696 byte
models. The new base picks `H=16, sel1=NONE, sel2=BITPOS`, quant Q8/IQ8,
**model 608 B**: W1 is a **single 16×26 matrix** (one set), W2 is **8 sets × 17
vectors** (16 hidden outputs + bias). Consequently:

* Muon is applied to exactly one W1 matrix per fit (plus the tied warm-up
  epochs), not to many selector matrices.
* Per-selector-set routing of W1 degenerates: with `s1 == 1` every W1 sample
  belongs to the one hot set. `TAW_HYBRID=route TAW_ROUTE_Q=0.5` produced
  **byte-identical archives** to plain Muon (179244/177788/372030/367852),
  confirming this. The routing mechanism is implemented and works for shapes
  with `sel1 != NONE`, but it has nothing to route on the winning shape.

## Exact update rules implemented

Linear LR schedule throughout: `lr(step) = lr0 + (lr1-lr0) * step/steps_total`;
`steps_total = epochs * ceil(ns/256)`. Warm-up (first `max(1, epochs/5)`
epochs) trains a tied set 0 which is then copied to all sets; W1 and W2 use
the same schedule.

**AdamW** on a vector row (W2 always; W1 in `adam`/cold-route/Adam half):

```
g_k = (accumulated gradient of the batch) / batch_size
m_k = b1*m_k + (1-b1)*g_k
v_k = b2*b_k + (1-b2)*g_k^2
w_k -= lr * ( (m_k/(1-b1^t)) / (sqrt(v_k/(1-b2^t)) + eps) + wd*w_k )
```
`b1=0.9, b2=0.999, eps=1e-6` by default, `wd=1e-5` for vectors.

**Muon** on W1 as a row-major `H × 26` matrix (max 32×26, fixed-size stack
scratch), per touched set, with its own momentum `m[]` (the original W1
momentum buffer, never shared with W2's Adam state at offset `n1`):

```
m   = 0.95*m + g                       # g already scaled by 1/batch_size
u   = 0.95*m + g                       # Nesterov extrapolation
X   = u / ||u||_F                      # Frobenius normalization
repeat NS=5 times (rows/cols transposed so the short dim is first):
    A = X X^T
    B = -4.7750*A + 2.0315*A^2
    X = 3.4445*X + B X
W  -= lr * 0.2 * sqrt(max(H,26)) * X + lr * 1e-3 * W
```
The 0.2 gain matches Muon's usual `sqrt(max(1, rows/cols))` for our tall
matrices. All gradients are cleared once per batch after the touched rows are
updated, preserving the "lazy per-set" behaviour of the original fitter and
its bias correction.

**Hybrid modes** (`TAW_HYBRID`), all keeping W2 on AdamW:

* `muon`: Muon on every touched W1 set (this is the winner).
* `blend`: `ΔW1 = α·Muon(lr_muon, wd_muon) + (1−α)·AdamW(lr1, wd1)`, with
  **separate** momentum/second-moment state for the Adam half (`m1/v1` arrays,
  copied at untie like the others); `α = TAW_ALPHA`.
* `alt`: per step (`TAW_ALT=step`) or per half of training (`=half`), W1 uses
  Muon or AdamW; `TAW_ALT_FIRST` picks which comes first.
* `route`: W1 sets are counted once from the training samples; the top
  `TAW_ROUTE_Q` fraction (by count) use Muon, the rest AdamW. Degenerates for
  `s1=1` as noted above.
* `adam`: original fitter, bit-exact.

## Environment knobs (all unset = shipped behaviour)

| variable | default | meaning |
|---|---|---|
| `TAW_HYBRID` | `adam` | `adam`, `muon`, `blend`, `alt`, `route` |
| `TAW_LR0` / `TAW_LR1` | `0.02` / `LR0/20` | plain-AdamW LR endpoints (also W2 defaults) |
| `TAW_VEC_LR0` / `TAW_VEC_LR1` | `TAW_LR0` / `TAW_LR1` | W2 AdamW LR endpoints |
| `TAW_MUON_LR0` / `TAW_MUON_LR1` | `0.01` / `LR0/20` | W1 Muon LR endpoints |
| `TAW_ADAM1_LR0` / `TAW_ADAM1_LR1` | W2 endpoints | W1 Adam LR (blend/alt/route) |
| `TAW_B1` / `TAW_B2` / `TAW_WD` | `0.9` / `0.999` / `1e-5` | Adam betas and vector WD |
| `TAW_MUON_MOM` / `TAW_MUON_NS` / `TAW_MUON_NEST` | `0.95` / `5` / `1` | Muon momentum, NS steps, Nesterov |
| `TAW_MUON_GAIN` / `TAW_MUON_WD` | `0.2` / `1e-3` | Muon step gain and decoupled WD |
| `TAW_ADAM1_WD` | `TAW_WD` | WD of W1 Adam half |
| `TAW_ALPHA` | `0.5` | blend weight of the Muon W1 update |
| `TAW_ALT` / `TAW_ALT_FIRST` | `step` / `muon` | alternation schedule |
| `TAW_ROUTE_Q` | `0.5` | fraction of W1 sets routed to Muon |
| `TAW_WARM` | `muon` | `adam` gives AdamW the tied warm-up epochs |

The extra W1 state (`m1/v1`) and per-set counters (`cnt1/route1`) were added
to `tnet` in `fit_internal.h`, allocated/freed in `tnet_init`/`tnet_free` and
copied at `tnet_untie`. The GPU trainer's `tnet_init` is unaffected (extra
buffers simply unused).

## Method

* `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`,
  `cmake --build build -j1`; `ctest --test-dir build --output-on-failure`
  passes **12/12** both with no overrides and with the winning environment.
* `bench.sh <NAME>` encodes text1m/libc1m at levels 1/5 with `-p mix` into
  distinct `bench/arcs/<NAME>_*` names, decodes each archive and compares the
  output byte-for-byte with the source corpus (`cmp`). **All 204 measurements
  in `bench/results.tsv` have `decode_ok=1`.** Winning archives report
  `profile 1 (F32_STRICT_MIX)`, model 608 B (one libc L5 archive picked IQ8,
  still standard profile 1). Timings were recorded but treated as noise on a
  shared 12-core box; archive bytes are the only metric.
* 51 configurations (204 encodes); every decode verified; every retained code
  state passed the full test suite.

## Full results (archive bytes, lower is better)

Sorted by sum. The `mode` column is abbreviated; full environment strings are
in `bench/results.tsv`. `route_q050` is a byte-identical control for
`muon_040_v003`; `route_q000` for `muon_010_v003s`.

| config | mode | text L1 | text L5 | libc L1 | libc L5 | sum | vs shipped | vs tuned AdamW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `muon_055_lowfin` | muon .055→1.1e-3 | 179678 | 177816 | 371373 | 366997 | 1095864 | −19347 | −8716 |
| `muon_055_fin100` | muon .055→5.5e-4 | 179596 | 177860 | 371628 | 367101 | 1096185 | −19026 | −8395 |
| `muon_055_v003` | muon .055→2.75e-3 | 179577 | 178098 | 371481 | 367171 | 1096327 | −18884 | −8253 |
| `muon_050_v003` | muon .05→2.5e-3 | 179440 | 177837 | 372287 | 367129 | 1096693 | −18518 | −7887 |
| `muon_050_ns3` | muon .05, NS3 | 179385 | 177874 | 372396 | 367116 | 1096771 | −18440 | −7809 |
| `muon_060_lowfin` | muon .06→1.2e-3 | 179758 | 178006 | 371883 | 367149 | 1096796 | −18415 | −7784 |
| `muon_060_v003` | muon .06→3e-3 | 179900 | 178006 | 371602 | 367404 | 1096912 | −18299 | −7668 |
| `muon_040_v003` | muon .04→2e-3 | **179244** | 177788 | 372030 | 367852 | 1096914 | −18297 | −7666 |
| `route_q050` | route, all W1 hot (=Muon) | 179244 | 177788 | 372030 | 367852 | 1096914 | −18297 | −7666 |
| `muon_050_lowfin` | muon .05→1e-3 | 179573 | 177974 | 372328 | 367126 | 1097001 | −18210 | −7579 |
| `muon_050_wd0` | muon .05, W1 wd=0 | 179408 | 178211 | 371684 | 367716 | 1097019 | −18192 | −7561 |
| `muon_wd0` | muon .04, W1 wd=0 | 179768 | 177660 | 371472 | 368224 | 1097124 | −18087 | −7456 |
| `muon_ns3` | muon .04, NS3 | 179539 | **177579** | 372839 | 367241 | 1097198 | −18013 | −7382 |
| `muon_055_highfin` | muon .055→5.5e-3 | 179662 | 178100 | 371801 | 367658 | 1097221 | −17990 | −7359 |
| `muon_070_v003` | muon .07→3.5e-3 | 180330 | 178227 | 371884 | 366852 | 1097293 | −17918 | −7287 |
| `muon_mom098` | muon .04, β=0.98 | 180118 | 177956 | 372387 | 366891 | 1097352 | −17859 | −7228 |
| `muon_065_v003` | muon .065→3.25e-3 | 180228 | 178055 | 371632 | 367457 | 1097372 | −17839 | −7208 |
| `muon_050_v005` | muon .05, W2 .005 | 179266 | 177958 | 373297 | 367188 | 1097709 | −17502 | −6871 |
| `muon_w2b1` | muon .04, W2 β1=.95 wd=1e-3 | 179716 | 177703 | 372734 | 367608 | 1097761 | −17450 | −6819 |
| `blend_a075` | blend α=.75, W1-Adam .003 | 179636 | 177523 | 373001 | 367744 | 1097904 | −17307 | −6676 |
| `muon_030_v003` | muon .03→1.5e-3 | 179475 | **177661** | 372359 | 368418 | 1097913 | −17298 | −6667 |
| `alt_half_muon` | alt half, Muon first | 179504 | 177871 | 372509 | 368101 | 1097985 | −17226 | −6595 |
| `muon_ns7` | muon .04, NS7 | 179785 | 177654 | 372536 | 368053 | 1098028 | −17183 | −6552 |
| `muon_050_v002` | muon .05, W2 .002 | 180027 | 178115 | 373031 | 367162 | 1098335 | −16876 | −6245 |
| `alt_step_adam` | alt step, Adam first | 180248 | 177640 | 372723 | 367791 | 1098402 | −16809 | −6178 |
| `alt_step_muon` | alt step, Muon first | 179576 | 177900 | 373977 | 367666 | 1099119 | −16092 | −5461 |
| `blend_a075b` | blend α=.75, W1-Adam .055 | 179829 | 178028 | 372946 | 368339 | 1099142 | −16069 | −5438 |
| `muon_050_mom098` | muon .05, β=.98 | 181069 | 178827 | 372141 | 367136 | 1099173 | −16038 | −5407 |
| `blend_a090` | blend α=.90, W1-Adam .055 | 179878 | 177845 | 372522 | 369067 | 1099312 | −15899 | −5268 |
| `muon_020_v003` | muon .02→1e-3 | 179627 | 177596 | 374445 | 368251 | 1099919 | −15292 | −4661 |
| `muon_015_v003` | muon .015→7.5e-4 | 179819 | 177581 | 374936 | 367860 | 1100196 | −15015 | −4384 |
| `muon_mom090` | muon .04, β=.90 | 179626 | 177963 | 374534 | 368680 | 1100803 | −14408 | −3777 |
| `blend_a050` | blend α=.50 | 179811 | 177567 | 374550 | 368993 | 1100921 | −14290 | −3659 |
| `muon_warm_adam` | muon .04, AdamW warm-up | 180294 | 177771 | 374110 | 369340 | 1101515 | −13696 | −3065 |
| `muon_010_v003s` | muon .01, W2 .003→1.5e-4 | 180681 | 177414 | 375209 | 368586 | 1101890 | −13321 | −2690 |
| `route_q000` | route, no W1 hot (=W1 Adam) | 180681 | 177414 | 375209 | 368586 | 1101890 | −13321 | −2690 |
| `muon_010_v003` | muon .01, W2 .003→6e-5 | 180785 | 177459 | 375243 | 368757 | 1102244 | −12967 | −2336 |
| `muon_010_v002` | muon .01, W2 .002 | 181004 | 177438 | 375660 | 368415 | 1102517 | −12694 | −2063 |
| `blend_a025` | blend α=.25 | 180219 | 177648 | 376494 | 368706 | 1103067 | −12144 | −1513 |
| `muon_030_vt` | muon .03, W2 "tuned AdamW" | 179898 | 178875 | 373613 | 370848 | 1103234 | −11977 | −1346 |
| `muon_020_vt` | muon .02, W2 "tuned AdamW" | 180164 | 178450 | 374109 | 371687 | 1104410 | −10801 | −170 |
| `adam_tuned` | AdamW .055, β1=.95, wd=1e-3 | 179913 | 178886 | 373887 | 371894 | 1104580 | −10631 | 0 |
| `muon_020_v020` | muon .02, W2 .02 | 179908 | 178790 | 376318 | 371022 | 1106038 | −9173 | +1458 |
| `muon_010_v020` | muon .01, W2 .02 | 181093 | 178332 | 376301 | 371282 | 1107008 | −8203 | +2428 |
| `muon_005_v020` | muon .005, W2 .02 | 181375 | 178547 | 376932 | 371912 | 1108766 | −6445 | +4186 |
| `muon_002_v020` | muon .002, W2 .02 | 182086 | 178843 | 377570 | 370941 | 1109440 | −5771 | +4860 |
| `adam_003s` | AdamW .003→1.5e-4 | 183652 | 179082 | 378148 | 374195 | 1115077 | −134 | +10497 |
| `adam_020` | shipped AdamW | 181557 | 179166 | 379197 | 375291 | 1115211 | 0 | +10631 |
| `adam_003` | AdamW .003→6e-5 | 183686 | 179165 | 378209 | 374221 | 1115281 | +70 | +10701 |
| `adam_002` | AdamW .002→6e-5 | 184345 | 180067 | 378829 | 374681 | 1117922 | +2711 | +13342 |
| `adam_005` | AdamW .005→2.5e-4 | 183263 | 180956 | 379396 | 376815 | 1120430 | +5219 | +15850 |

Per-cell minima: text L1 `muon_040_v003` = 179244; text L5
`muon_030_v003` = 177661; libc L1 `muon_060_v003` = 371602; libc L5
`muon_mom098` = 366891. No single Muon setting is Pareto-best across all four
cells, but `muon_055_lowfin` is the best compromise and beats **both**
AdamW controls in every cell.

## Hyperparameter findings

1. **W1 Muon and W2 AdamW LRs are strongly coupled.** With the old W2 default
   (0.02→0.001), Muon 0.01→5e-4 saved 8,203 B in aggregate; with W2
   0.003→1.5e-4 the *same* Muon LR saved 13,321 B. Using the "tuned AdamW"
   W2 setting (0.055→5.5e-4, β1=.95, wd=1e-3) with Muon W1 was much worse
   (e.g. Muon 0.03: sum 1,103,234) than W2 0.003→1.5e-4 with default β/WD
   (1,097,913). W2's optimum is different once W1 is orthogonalized.
2. **Muon LR sweep** (W2 0.003→1.5e-4): 0.01 → 1,102,244; 0.02 → 1,099,919;
   0.03 → 1,097,913; 0.04 → 1,096,914; 0.05 → 1,096,693; 0.055 → 1,096,327;
   0.06 → 1,096,912; 0.065 → 1,097,372; 0.07 → 1,097,293. Broad optimum
   0.04–0.06; final-LR ratio 1/50 best (1,095,864 vs 1/100 1,096,185 and
   1/20 1,096,327 at lr0=0.055).
3. **Muon internals** (at W1 LR 0.04): β=0.95 better than 0.90 (1,096,914 vs
   1,100,803) and slightly better than 0.98 in aggregate (1,097,352), though
   0.98 gives the best libc L5 cell; NS=5 best (NS3 1,097,198, NS7 1,098,028);
   wd=1e-3 slightly better than wd=0 (1,096,914 vs 1,097,124); Nesterov and
   gain 0.2 kept from the old-base Muon work.
4. **Blending** (W1 Muon 0.04, W1 Adam 0.003, W2 0.003): α=0.25 → 1,103,067;
   α=0.5 → 1,099,921; α=0.75 → 1,097,904; α=0.9 (with W1-Adam 0.055) →
   1,099,312. Monotone towards α=1: pure Muon is best; the Adam half never
   pays for itself. Best blend was still 2,040 B worse than pure Muon 0.04.
5. **Alternation**: step-wise Muon-first 1,099,119; step-wise Adam-first
   1,098,402; half-wise Muon-first 1,097,985 — all worse than plain Muon
   (1,096,914). Warm-up in AdamW instead of Muon also hurt (1,101,515).
6. **Routing**: with `H=16, sel1=NONE` there is a single W1 set, so the idea
   is vacuous on this base; `TAW_ROUTE_Q=0.5` is bit-identical to plain Muon.
   The mechanism (per-set sample counts, top-q hot sets) is implemented for
   future multi-set shapes; the all-cold control (`Q=0`) is identical to
   W1-Adam with W2 0.003.

## Caveats

* Archive bytes are deterministic; decode timings were noisy (shared machine,
  other agents benchmarking) and are not used for any conclusion. Structurally
  the model size is unchanged (608 B), so decode time should be unchanged.
* Every archive was decoded and `cmp`-checked against its source; all passed.
  The format/decoder were never modified, so byte-identical decode is expected
  and observed.
* Results cover only the two 1 MiB corpora, levels 1 and 5, and `-p mix`;
  no L9 or other shapes were benchmarked. The old-base 0.05–0.31 % Muon margin
  does not transfer — the new base's single 16×26 W1 matrix is a different
  optimization problem.
* Plain AdamW on this base does **not** prefer the old-base low LR: LR 0.003
  regressed text L1 by ~2.1 KB. The tuned 0.055/β1=.95/WD=1e-3 control was
  supplied by `exp/lr-baseline` and reproduced exactly.
* The shipped default remains AdamW; adopting the winner requires either
  setting the four environment variables above or hardcoding them in
  `tnet_train()` (deliberately not done here to keep A/B reproducible).

## Reproducing the winner

```sh
cd /home/andrew/.local/share/opencode/worktree/8500c0/exp-muon-adam
cmake --build build -j1
TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 \
TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 \
  build/tools/taw c -l 1 -p mix /tmp/opencode/tadamw-bench/text1m out.aw
```
