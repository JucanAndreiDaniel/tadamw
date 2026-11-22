# H-Net-style context-change weighting / ratio balancing in the encoder fitter

Branch `exp/hnet-weight`, base `origin/main` e64cc83. Encoder-only changes in
`src/fit.c`; format, quantizer and decoder untouched. No push. Research basis:
H-Net (arXiv:2507.07955) — boundary probability from the dissimilarity of
neighbouring representations, plus an MoE-style ratio loss (Eq. 10).

## Verdict (headline)

**The idea transfers weakly. One narrow configuration gives a small but
seed-robust win; it is far from the tuned-optimizer controls and does not
stack with them.**

| arm | text L1 | text L5 | libc L1 | libc L5 | sum | vs shipped |
|---|---:|---:|---:|---:|---:|---:|
| shipped AdamW (e64cc83) | 181557 | 179166 | 379197 | 375291 | 1,115,211 | — |
| tuned AdamW (`exp/lr-baseline`, reproduced here) | 179913 | 178886 | 373887 | 371894 | 1,104,580 | −10,631 (−0.95 %) |
| tuned Muon (`exp/muon-adam`, external reference¹) | 179678 | 177816 | 371373 | 366997 | 1,095,864 | −19,347 (−1.74 %) |
| **H-Net winner: `st2`, p=1, lag=32, C=2** | **181930** | **178717** | **377297** | **373265** | **1,111,209** | **−4,002 (−0.36 %)** |
| H-Net winner + tuned AdamW | 180160 | 179526 | 375624 | 372163 | 1,107,473 | −7,738, but **+2,893 worse than tuned AdamW alone** |

The H-Net win is real in sign (4/4 seeds improve) but small (mean −2,449 B,
−0.22 %), concentrated in libc L5, and **not Pareto**: text L1 is +373 B at
the default seed. The ratio arm alone, the boundary arm, and all other score
definitions are at or worse than baseline. Recommendation: **do not adopt** as
a replacement for optimizer tuning; it is a niche fitting regularizer at best.

¹ Muon's optimizer lives in sibling `exp/muon-adam` and is not in this
worktree; those four numbers are the published control, not reproduced here.

## Exact score and loss formulas

Every collected `sample` (`src/fit_internal.h`) holds the 25 integer context
inputs `st[0..24]`, the selector context `c0, c1, msel`, and `bit`. Samples are
stored in input-position order. For a lag `L` (`TAW_HNET_LAG`, default 1), with
`a = sm[i]`, `b = sm[i-L]` and `chg[i]=0` for `i<L`:

| `TAW_HNET_SCORE` | `chg_i` (all integer arithmetic; `st` is `int16_t`) |
|---|---|
| `st2` (default) | `abs(a.st[0]-b.st[0]) + abs(a.st[1]-b.st[1])` — direct o0 + o1 stretch |
| `all` | `sum_{q=0}^{24} abs(a.st[q]-b.st[q])` |
| `byte` | `abs((int)a.c1-(int)b.c1)` — previous-byte change |
| `c0` | `abs((int)a.c0-(int)b.c0)` — partial-byte change |
| `c0c1` | `abs((int)a.c0-(int)b.c0) + abs((int)a.c1-(int)b.c1)` |

**Weight arm** (`TAW_HNET_MODE=weight`), with `mu = max(1, mean_i chg_i)`,
`r_i = chg_i/mu`, exponent `p = TAW_HNET_W`, clamp `C = TAW_HNET_CLAMP`:

```
w_i = clamp( r_i^p / mean_j(r_j^p), 1/C, C )
```

`p ∈ {-2,-1,-0.5,0,0.5,1,2}` (negative = stable samples count more);
`r^p` uses `fsqrt` for `|p|=0.5`, multiplication for `|p|=2`, no libm. The
mean normalization keeps the total gradient scale ≈1. `w_i` multiplies the
cross-entropy gradient `gz = p_i - bit_i` in `tnet_step`; at `w=1` no FP
operation is added, so the default path is bit-identical.

**Boundary arm** (`TAW_HNET_MODE=boundary`): binary H-Net analogue. Sort
`chg`; the top `q = TAW_HNET_Q` (default 0.1) are "boundaries". `w_i = C` for
those and `1/C` otherwise, rescaled to mean 1 and clamped to `[1/C, C]`.

**Ratio arm** (`TAW_HNET_MODE=ratio`): let `k(i)` be the class of sample i,
`f_k = sum_{i in k} w_i / sum_i w_i` its share of weighted gradient mass, `K`
the class count, `t = TAW_HNET_TARGET / K` (1.0 = uniform target). One
projected step on the load-balancing objective `L = K sum_k f_k^2` (minimized
at uniform mass) is implemented as inverse-mass reweighting:

```
m_i = clamp( (t / f_{k(i)})^alpha / Z, 1/C, C ),
Z   = sum_i w_i (t/f_{k(i)})^alpha / sum_i w_i,
w_i = clamp( w_i * m_i, 1/C, C ),                 alpha = TAW_HNET_RATIO
```

`TAW_HNET_CTX` chooses classes: `sel` = the candidate's output selector
(bitpos, 8 classes, on the winning shape), `c0` = 256, `c1hi` = 16, `bitpos`
= 8. With `sel` and a second selector present, the two balance multipliers
are applied independently and multiplied.

**`both`** composes weight (or boundary) then ratio. Scoring is **never**
weighted: `evaluate()` still measures bits/bit with the decoder's real
inference on all samples, so the objective remains the true archive size.

### Knobs (all opt-in; default off ⇒ baseline byte-for-byte)

| knob | meaning |
|---|---|
| `TAW_HNET_MODE` | `off` (default), `weight`, `boundary`, `ratio`, `both` |
| `TAW_HNET_SCORE` | `st2` (default), `all`, `byte`, `c0`, `c0c1` |
| `TAW_HNET_W` | change exponent p, default 1 |
| `TAW_HNET_LAG` | compare sample i to i−lag, default 1 |
| `TAW_HNET_CLAMP` | bound C, default 2 |
| `TAW_HNET_Q` | boundary top-quantile, default 0.1 |
| `TAW_HNET_TARGET` | target class mass as a multiple of 1/K, default 1 |
| `TAW_HNET_RATIO` | balance exponent alpha, default 1 |
| `TAW_HNET_CTX` | `sel` (default), `c0`, `c1hi`, `bitpos` |

Benchmark-only knobs also ported for controls: `TAW_LR0/LR1/B1/B2/WD`
(tuned-AdamW control, defaults unchanged) and `TAW_SEED` (shifts both fitter
RNG seeds, default 0). GPU fitting is disabled while any H-Net knob is on
(the GPU trainer has no sample weights); the benchmark path is CPU anyway.

## Method

Archive bytes from `taw c -l {1,5} -p mix` on the read-only 1 MiB corpora
`/tmp/opencode/tadamw-bench/{text1m,libc1m}`, clang 23 Release, CPU fitting.
Fitting is deterministic for a fixed environment, so each cell was encoded
once; baseline cells were additionally `cmp`-verified against the reference
archives. `TAW_FIT_DEBUG=1` confirms every arm fits the same family
(H=16, sel1=NONE, sel2=BITPOS, Q8/IQ8/IQ4, 608-byte model), so deltas are
fitting effects, not shape changes. Full raw data: `bench/sweep.tsv`,
summary: `bench/report.txt`, configs: `bench/configs{,2,3,4}.tsv`.

## Sweep results

`delta` = sum − shipped baseline sum (1,115,211 B). "1m" in the header is
text1m for the first two cells and libc1m for the last two.

### Main table (all configurations, sorted by sum)

```
config                textL1  textL5  libcL1  libcL5      sum    delta
tuned_ctl             179913  178886  373887  371894  1104580  -10631
tuned_lag32           180163  178976  375594  371905  1106638   -8573
tuned_bnd_q10         180227  179487  374961  372097  1106772   -8439
tuned_w_p1_lag32      180160  179526  375624  372163  1107473   -7738
w_st2_p1_lag32        181930  178717  377297  373265  1111209   -4002  <-- winner
both_p1_lag32_sel     181881  178941  377184  375585  1113591   -1620
seed3_win             181834  178928  379819  373267  1113848   -1363
w_st2_pm2             182013  179212  379254  373710  1114189   -1022
seed2_win             181895  179661  379456  373234  1114246    -965
w_st2_p2_lag32        182418  179040  379549  373336  1114343    -868
w_st2_p1_lag32_C4     182231  178914  377537  375721  1114403    -808
w_st2_p1_lag32_C3     182304  178848  377469  375787  1114408    -803
w_st2_p1_lag32_C15    181835  178403  379269  375370  1114877    -334
bnd_st2_q05           181654  178983  379270  375213  1115120     -91
r_sel_a1              181570  179119  379159  375281  1115129     -82
seed2_base            181865  178918  379085  375327  1115195     -16
base                  181557  179166  379197  375291  1115211       0
w_st2_p05_lag32       182050  178674  379179  375317  1115220      +9
r_sel_a2              181617  179130  379165  375329  1115241     +30
seed1_win             182964  179760  379333  373243  1115300     +89
w_st2_pm2_C15         181640  178987  379190  375527  1115344    +133
w_st2_pm1_lag32       182465  180291  379212  373386  1115354    +143
bnd_st2_lag32_q05     181825  179139  379074  375372  1115410    +199
w_all_p1_lag32        181668  179292  379234  375540  1115734    +523
w_c0c1_pm1            181421  179546  379072  375753  1115792    +581
w_st2_p1_lag8         181713  179459  379291  375344  1115807    +596
bnd_st2_lag32_q10     182139  179281  379140  375325  1115885    +674
r_c1hi_a1             181762  179045  379333  375795  1115935    +724
w_st2_pm2_lag8        181722  179373  379287  375558  1115940    +729
both_pm2_c0           182041  179054  379261  375598  1115954    +743
w_all_p05             181531  179646  379165  375619  1115961    +750
both_pm2_sel          181988  179242  379246  375489  1115965    +754
w_c0_pm1              181532  179851  378815  375770  1115968    +757
w_all_pm1             181902  178836  379695  375630  1116063    +852
seed1_base            182336  179601  378859  375351  1116147    +936
w_st2_p1_lag256       182205  179101  379525  375445  1116276   +1065
w_st2_pm2_lag32       182060  179546  379292  375395  1116293   +1082
w_st2_p1_lag16        181979  179436  379370  375548  1116333   +1122
w_st2_p1_lag128       181947  179367  379536  375566  1116416   +1205
w_c0_p05              182328  179395  379398  375464  1116585   +1374
w_all_p1              181599  179650  379342  376065  1116656   +1445
w_st2_pm05            182143  179777  379440  375330  1116690   +1479
w_st2_p1_lag64        182042  179676  379477  375512  1116707   +1496
w_c0c1_p05            182307  179657  379311  375480  1116755   +1544
w_st2_p1             181838  179637  379536  375877  1116888   +1677
w_byte_pm1            182072  179571  379530  375761  1116934   +1723
r_c0_a1              182045  179850  379368  375732  1116995   +1784
bnd_st2_q25           182200  179399  379712  375862  1117173   +1962
w_c0c1_p1            182537  179813  379393  375585  1117328   +2117
w_st2_pm1            182367  179876  379653  375463  1117359   +2148
w_st2_p2             182137  179817  379583  376000  1117537   +2326
seed3_base            181989  178972  379388  377496  1117845   +2634
w_st2_p05            181554  179387  379305  377672  1117918   +2707
w_c0_p1              183353  179778  379480  375557  1118168   +2957
bnd_st2_q10           181677  179571  379300  377719  1118267   +3056
w_byte_p05            182490  180549  379367  375939  1118345   +3134
w_st2_pm2_C3          182578  180343  379543  376342  1118806   +3595
tuned_pm2_C4          182925  181666  377848  377230  1119669   +4458
w_st2_pm2_C4          183001  180869  379931  376222  1120023   +4812
w_byte_p1             182758  180723  379405  377921  1120807   +5596
```

### What the sweep says

- **Score/exponent at lag 1**: forward change weighting (`p>0`) is uniformly
  harmful (byte/p=1: +5,596 B); inverse weighting on `st2` at p=−2 is the
  only lag-1 arm that wins (−1,022 B). Byte/c0/c0c1/all scores are neutral to
  strongly negative.
- **Lag is the decisive knob**: with `st2`, p=1, lag 8/16/32/64/128/256 gives
  +596 / +1,122 / **−4,002** / +1,496 / +1,205 / +1,065. The optimum at 32 is
  sharp; neighbouring lags lose. Exponent 0.5/2 at lag 32 give +9 / −868,
  p=−1 gives +143, so the `p=1, lag=32` point is also narrow in p.
- **Clamp**: C=1.5/3/4 at that point give −334 / −803 / −808, all worse than
  C=2, so the win is not from a larger clamp.
- **Ratio arm alone** (`sel` classes = bitpos, already nearly uniform):
  −82 B at alpha=1, +30 at alpha=2; `c0`/`c1hi` skew adds +1,784 / +724.
  Load balancing is essentially a no-op on the fitted shape and harmful on
  skewed contexts.
- **Boundary arm** (top-q hard weights): −91 B at q=0.05, +3,056 at q=0.10;
  no useful signal.
- **`both`**: −1,620 B (worse than weight alone) — ratio adds nothing.
- **Stacking on tuned AdamW**: every H-Net arm tested hurts the stronger
  base: +2,893 (lag-32 winner), +2,058 (inverse lag-32), +2,192 (boundary),
  +15,089 (inverse C=4). The mechanisms do not compose.

### Seed robustness of the winner (`TAW_SEED`, same seed for base and arm)

| seed | base sum | winner sum | delta |
|---:|---:|---:|---:|
| 0 (default) | 1,115,211 | 1,111,209 | −4,002 |
| 1 | 1,116,147 | 1,115,300 | −847 |
| 2 | 1,115,195 | 1,114,246 | −949 |
| 3 | 1,117,845 | 1,113,848 | −3,997 |
| **mean** | | | **−2,449 (−0.22 %)** |

Sign is robust (4/4 seeds), magnitude is not. Per-cell across seeds, only
libc L5 improves every time (−2,026 / −2,108 / −2,093 / −4,229 B); text L1,
text L5 and libc L1 are mixed. So the aggregate win is carried by one corpus
at level 5.

## Validation

- Default (no H-Net knobs): all four baseline archives reproduce **byte for
  byte** against `/tmp/opencode/tadamw-bench/{text1m,libc1m}_L{1,5}.aw`,
  re-verified after every code change.
- Finalists decoded with `taw d` and `cmp`-verified against both sources:
  winner, ratio, both and tuned-control archives, 16/16 cells.
- `ctest --test-dir build --output-on-failure`: **12/12** (gpu skipped in the
  non-Vulkan build).
- GCC 16 `-DTADAMW_WERROR=ON` Release build: clean, no warnings.

## Caveats

1. The fitter's features are **fixed integer context models**; unlike H-Net,
   fitting never learns a representation or a boundary detector. A per-sample
   weight is therefore only a biased reweighting of a fixed objective, not a
   boundary-aware representation loss. That is the most likely reason the
   transfer is weak.
2. The win is concentrated in libc L5 and the lag-32/exponent-1 point is
   narrow; cross-corpus/cross-level generalisation is not demonstrated. A
   0.36 % aggregate delta with a mixed per-cell pattern is within the range
   that a different corpus or effort level could erase.
3. H-Net weights are computed from *collected samples*, not every input bit.
   At L5 that is ~1 bit/byte and at L1 ~1 bit/4 bytes, with a rotated bit
   position, so "context change over lag L" mixes byte distance and bit
   position; the meaning of an optimal lag is not clean.
4. All arms were fitted on CPU. With GPU fitting on, H-Net knobs force CPU
   fitting, so they cannot be combined with the 4x GPU speed-up.
5. The ratio penalty is implemented as one projected inverse-mass step at a
   fixed point, not as a learned router; the target is a mass fraction, not
   H-Net's compression ratio. No knob maps onto H-Net's ratio value.
6. The tuned-Muon control is an external number from a sibling branch, not
   reproduced in this worktree; the tuned-AdamW control was reproduced
   exactly here (179913/178886/373887/371894).
7. `TAW_SEED`, `TAW_LR*` and H-Net knobs are encoder-only benchmarking aids;
   none changes the archive format, and decoders need no support for them.
