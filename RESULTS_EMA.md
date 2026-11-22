# EMA / SWA weight averaging — EMA-experiment report

Branch `exp/ema`, rebased onto `origin/main` at **`1aa3dd7`** (merged PR #3
MALTER and PR #4 Muon). This file is the EMA experiment's report; upstream's
own `RESULTS.md` (MALT/MALTER leader) is preserved untouched in this branch.

**Verdict: KEEP the gated `late` default** — EMA(0.999) for runs with ≥6 epochs
(levels 3+), last iterate for levels 1–2. 12/12 tests pass. Format, decoder,
quantization and golden vectors are untouched.

## Rebase / integration notes

The rebase conflicted in `src/fit.c` and `src/fit_internal.h`; upstream's
merged fit code was preserved in full and only the averaging delta was layered
back on:

- `tnet` lives in the shared `fit_internal.h`; upstream added Muon momentum,
  `m1/v1`, MALT preconditioner EMAs (`malt_l/r/nu`), `cnt1`, `route1` and the
  `optcfg` optimizer dispatch. All of it is intact; the EMA experiment adds
  only `wa`, `last1`, `last2` (scratch, allocated only when averaging is on)
  and the corresponding frees.
- `tnet_train` keeps upstream's `fit_warm_epochs()`, `fit_stride()`, hybrid
  optimizer dispatch (`adam_rows` / `muon_rows` / `malt_rows`), `TAW_*` LR
  knobs and MALT debug hooks. The EMA fold-in is inserted *after* each
  optimizer update to a touched set, so it works on top of any optimizer path.
- My earlier diagnostic `TAW_FIT_LRSCALE` knob was dropped in favour of
  upstream's `TAW_LR0`/`TAW_LR1`; the control below uses those.
- GPU-trained candidates have no CPU trajectory, so they keep the last
  iterate (same caveat as before the rebase).
- I did not re-benchmark MALT/Muon configurations; only the shipped AdamW
  path (`TAW_HYBRID` unset, `TAW_MALT` unset) was measured.

## What was implemented

`TAW_FIT_AVG` env switch: `off|swa|ema9|ema99|ema999|late|…` (also `0..7`),
plus compile-time `TAW_FIT_AVG_DEFAULT` (now `late`). Parsed once in
`taw_model_fit_source`, carried in `fit_job` to each CPU fit thread.

- **SWA**: zero an accumulator (reset at un-tie), add a full `t->w` snapshot
  each epoch in the second half (`swa`) or from un-tie (`swa_all`), divide by
  the count. Full snapshots inherently respect lazy per-set updates.
- **EMA**: binary-exponentiation table `d2[k] = decay^(2^k)` (no libm/SIMD
  divergence). After an optimizer update to a touched set, with
  `gap = step - last[set]`, fold it in as
  `wa = decay^gap * wa + (1 - decay^gap) * w`, then `last[set] = step`.
  Untouched weights are constant over the gap, so decaying toward the current
  `w` is the time-weighted EMA without a full-array pass. State starts at
  un-tie (or at start for never-tied shapes). Decays: 0.9, 0.95, 0.99, 0.999.
- Averaged weights are copied into `t->w` before the existing quantize /
  serialize / evaluate path, so the stored and scored model is the quantised
  EMA/SWA solution. Trajectory, candidate set and selection rule unchanged.
- `late` = EMA(0.999) only when `epochs >= 6`; levels 1–2 keep last iterate.

## Byte-identical knob-off check

With `TAW_FIT_AVG=off`, all four cells reproduce the current upstream baseline
exactly (1 MiB, `-p mix`, model 608 B):

| corpus / level | upstream baseline | `TAW_FIT_AVG=off` |
|---|---:|---:|
| text1m L1 | 181557 | **181557** |
| text1m L5 | 179166 | **179166** |
| libc1m L1 | 379197 | **379197** |
| libc1m L5 | 375291 | **375291** |

`ctest --test-dir build --output-on-failure`: **12/12 passed**.

## Results — archive size in bytes (lower is better)

| mode | text L1 | text L5 | libc L1 | libc L5 | total Δ |
|---|---|---|---|---|---|
| upstream baseline (`off`) | 181557 | 179166 | 379197 | 375291 | — |
| SWA, last half | 182131 | 179651 | 379399 | 375751 | +1721 |
| EMA 0.9 | 181637 | 179155 | 379218 | 375289 | +88 |
| EMA 0.99 | 181651 | 179041 | 379005 | 375233 | −281 |
| EMA 0.999 | 181577 | **178993** | 379127 | **375127** | −387 |
| **`late` (shipped default)** | **181557** | **178993** | **379197** | **375127** | **−337** |

- SWA is consistently worse on this base (+202…+574 B per cell).
- EMA 0.999 is the best decay: L5 improves by **−173 B (text, −0.097%)** and
  **−164 B (libc, −0.044%)**; the gated default leaves L1 untouched.
- Default after rebase re-verified: 181557 / 178993 / 379197 / 375127.

## LR control (L5, via upstream `TAW_LR0`/`TAW_LR1`)

| LR endpoints | mode | text L5 | libc L5 |
|---|---|---:|---:|
| 0.02 → 0.001 (shipped) | off | 179166 | 375291 |
| 0.02 → 0.001 | EMA 0.999 | **178993** (−173) | 375127 (−164) |
| 0.002 → 0.0001 | off | 180054 (+888) | 374636 (−655) |
| 0.003 → 0.00015 | off | 179082 (−84) | 374195 (−1096) |
| 0.003 → 0.00015 | EMA 0.999 | 179042 (−124) | 374124 (−1167) |
| 0.005 → 0.00025 | off | 181079 (+1913) | 375426 (+135) |
| 0.008 → 0.0004 | off | 180465 (+1299) | 375201 (−90) |

The EMA gain persists under LR retuning: at 0.003→0.00015, EMA still gives
−40 (text) and −71 (libc) versus the matched LR control. LR tuning is a
stronger but data-dependent lever (MALTER upstream gains are much larger);
the two partially stack. Spot-checked after the rebase with the new knob:
`TAW_LR0=0.003 TAW_LR1=0.00015` gives 179082 (off) / 179042 (ema999), exactly
matching the table.

## Verdict

Keep `TAW_FIT_AVG_DEFAULT = late`. It is a small, deterministic,
format-neutral L5 win with no L1 regression, and it remains beneficial after
LR retuning. If the upstream optimizer leader (MALTER) is eventually shipped
as the default, re-run this A/B — the averaging delta is small enough to be
absorbed by other fitting changes.

## Caveats

- Encode/decode timings under the shared benchmark load were unusable (load
  ~35 on 12 cores); sizes are deterministic and were used instead. A clean
  encode-time measurement is advisable before merging if the guardrail is
  strict: EMA adds one blended read-modify-write per weight per step, a
  fraction of the existing forward + optimizer work.
- Averaging is validated only with the shipped AdamW path. It is applied on
  top of MALT/Muon iterates mechanically but has not been benchmarked there.
- GPU-trained candidates are not averaged; on a GPU-enabled machine, `late`
  would score averaged CPU candidates against non-averaged GPU ones.
- Levels 3–4 and 6–9 were not re-benchmarked; the `epochs >= 6` gate comes
  from earlier evidence and is unverified on this base.
- Gains are 0.04–0.1% on 1 MiB inputs on two corpora; SWA's regression shows
  the result is fit-regime dependent.
