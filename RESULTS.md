# Experiment: chained lightweight predictors with information inheritance (profile 10)

Branch `exp/chain`, rebased onto `origin/main` e64cc83 (9 upstream commits:
GPU fitting, new quant formats, bit-position sampling, fused AVX2 decoder,
parallel awseek decode). Branch commit: `872132b` plus a follow-up retuning
commit. Status: **implemented, all 12 CTest tests pass**.

> **Stale results:** the pre-rebase table in the first version of this file
> (190610 / 187956 / 388792 / 385741 for the baseline, and chain sizes
> 186708 / 186084 / 382858 / 382583) was measured against the old sampler
> and old profile-1 prior. Those numbers are invalid after e64cc83 and must
> not be compared with anything below.

## Paper

arXiv:2604.15472, *Lossless Compression via Chained Lightweight Neural Predictors with Information Inheritance* (Kim & Belyaev, 2026): a chain of predictors each with the minimum weights sufficient for a Markov source of a given order, with a low-order unit's probability estimate inherited by the next higher-order unit.

## What is implemented

Arithmetic profile **10, `F32_CHAINED_MIX`**, reuses profile 1's 12 integer context models, 25 stretched inputs + bias, squash/stretch tables, both APM stages and range coder bit-for-bit. Only the frozen mixer is replaced: three nested single-layer units (order 0 → order ≤3 → all contexts), each higher unit taking the previous unit's clamped binary32 stretch estimate as one extra input. No hidden layer; the per-unit clamps are the only nonlinearity. Final unit's estimate is truncated, squashed and passed to the unchanged APM path.

Format additions (`docs/format.md` §2.3, §5.2): profile id 10, three selector bytes in the 8-byte head, unit/subset/input order, inheritance and clamp/truncation order, F16/Q8 row payload shared with profile 1 through `src/quant.c`. New `src/p10.[ch]` (loader + scalar inference); profile 10 shares `struct taw_p1` and the APM code in `src/predictor.c` via a `chain` net pointer, with the fused AVX2 decoder disabled for it. Chain AdamW fitter (including inheritance backprop) in `src/fit.c`; golden vectors, validation and ISA-parity tests in `tests/test_p1.c`; CLI `-p chain`.

Profile 1's golden vectors and archives are unchanged; decoders that do not know profile 10 refuse it.

## Post-rebase benchmark

`/tmp/opencode/tadamw-bench/{text1m,libc1m}`, 1 MiB, `.aw` streams, levels 1 and 5. Archive bytes (primary signal); model bytes in parentheses.

| corpus | level | profile 1 (new baseline) | profile 10 (chain, retuned) | delta |
|--------|-------|--------------------------|-----------------------------|-------|
| text1m | 1 | 181557 (608) | 182555 (294) | +998 B (**+0.55% worse**) |
| text1m | 5 | 179166 (608) | 182383 (392) | +3217 B (**+1.80% worse**) |
| libc1m | 1 | 379197 (608) | **377534 (294)** | −1663 B (−0.44% better) |
| libc1m | 5 | 375291 (608) | 377455 (392) | +2164 B (**+0.58% worse**) |

The new baseline was reproduced exactly (including model sizes) on this build.

### Retuning after the upstream changes

The old chain heuristics were tuned against the old one-bit-per-byte sampler and no longer picked a winning shape. Under the new bit-position sampler:

- the selector-free 42–63-weight chain now measures 184405 / 183942 (text L1/L5) and 379273 / 378741 (libc L1/L5) — clearly worse;
- raising the model to C0-selector tables (~7–11 KB) improves the sample `bpc` but *worsens* real archive size (text L1 189212–192858), i.e. the sample score overfits many-set models;
- the best small shape uses a **bit-position selector on the final unit** (model 294–392 B), 12 fitting epochs (instead of the level's 3/6) and at least 2^19 samples;
- setting the inheritance weight's initial value to 1 and inheriting a squashed probability instead of a stretch value were both tried and were slightly worse; both were reverted.

The resulting default is: candidate cap 512 B, prior favouring `sel2 = BITPOS`, `epochs = max(12, level)`, `samples = max(2^19, level)`. `TAW_CHAIN_FORCE`, `TAW_CHAIN_EPOCHS` and `TAW_CHAIN_SAMPLES` remain experiment-only knobs.

All tests pass after the rebase: **12/12**, including the unchanged profile-1 goldens, the new quant tests, the GPU tests and the profile-10 goldens (which still produce lengths 25079 / 91360 and CRCs `0x8AB6CEC6` / `0xD0109680`).

## Verdict

**Negative overall / partial win.** After the upstream improvements the chain is no longer a consistent winner: it is smaller on libc L1 by 0.44%, but larger on the other three points, up to 1.80% on text L5. The 42–63-weight chain that beat the old MLP does not survive a stronger, better-trained profile-1 model (H=16, bit-position output selector). The paper's weight-minimization idea still produces a valid, very small model (294–392 B versus 608 B), but on this corpus it does not translate into better compression.

## Caveats

- The retuned prior, 512-byte cap and extra 12 epochs / 2^19 samples were selected from measurements on these same two corpora. They are heuristics, not a general rule.
- Chain fitting now uses more encoder effort than the level's default (12 epochs at L1/L5); decoding cost was not measured (timings are noisy in this environment). Only archive bytes are reported.
- The sample `bpc` remains a ranking proxy, not a size estimate: shapes with C0 selectors look much better on samples yet produce larger archives. Candidate choice is therefore fragile.
- Inheritance uses the clamped stretch-domain estimate rather than the paper's literal probability; the per-unit weight-minimization rule is approximated by three fixed nested subsets. No data-dependent unit count.
- Profile 10 accepts only F16/Q8; it does not use upstream's new quant formats.
- Only archive size is compared; no attempt was made to make the chain share the fused AVX2 per-bit loop.
