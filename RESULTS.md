# Experiment: chained lightweight predictors with information inheritance (profile 10)

Branch `exp/chain`, based on `origin/main` a427e7f. Status: **implemented, all CTest tests pass**.

## Paper

arXiv:2604.15472, *Lossless Compression via Chained Lightweight Neural Predictors with Information Inheritance* (Kim & Belyaev, 2026). Two claims, per the abstract:

1. a **chain** of neural predictors, each with the minimum number of weights sufficient for a Markov source of a given order, minimizes total weights as a function of the data's statistics;
2. **information inheritance** — feeding a low-order unit's probability estimate to the next, higher-order unit — improves compression.

This experiment applies both ideas to tadamw's profile 1 without touching its integer feature extraction or its range coder.

## Design

Arithmetic profile **10, `F32_CHAINED_MIX`**, keeps profile 1's 12 adaptive integer context models, the 25 stretched inputs + bias, squash/stretch tables, both adaptive APM stages and the binary range coder bit-for-bit. Only the frozen mixer is replaced.

Three single-layer units, nested by Markov order:

| unit | view | own inputs | weights/row |
|------|------|-----------|-------------|
| 0 | order 0 | `st0, hist0, bias` | 3 |
| 1 | order ≤3 | `st0,hist0, st1,hist1, st2,hist2, st3,hist3, sparse2-3 direct/hist, bias` | 12 (11 + inheritance) |
| 2 | all contexts | all 25 stretched inputs + bias | 27 (26 + inheritance) |

Each higher unit receives the previous unit's **clamped binary32 stretch-domain estimate** as one extra input (information inheritance), appended after its own inputs. Unit 0 has no predecessor. Inference is `dot4` (the profile-1 four-partial-sum order) → clamp to ±2047/256 → inherit → final unit's clamped estimate is truncated to integer stretch, squashed and fed to the unchanged APM path. There is no hidden layer; the per-unit clamps provide the only nonlinearity.

Selectors (weight-set choice by `bitpos`/`c0`) were kept as an option, but measurements favor no selector or only `bitpos` on unit 0: the chosen shape has 42–63 frozen weights versus profile 1's 1736 (L1 model). Chain inference is scalar by design; the shared context kernels still select their ISA and the whole archive is bit-identical across ISAs.

### Format additions (docs/format.md)

- §2 profile table: id 10 reserved for `F32_CHAINED_MIX`, quant `F16` or `Q8`.
- §2.3: normative unit subsets, input order, inheritance rule, clamp and truncation order, and the statement that §2.2.1–§2.2.7 and §2.2.9–§2.2.10 apply unchanged.
- §5.2: 8-byte head (same geometry bytes, three selector bytes, two reserved bytes), payload serialized unit 0→2, set→row, with the existing F16/Q8 row encodings; exact size formula and validation rules.
- `src/format.h`: `TAW_PROFILE_F32_CHAINED_MIX = 10`; `include/tadamw/tadamw.h`: `TAW_PREDICTOR_CHAIN`; CLI `-p chain`.

New files `src/p10.h`/`src/p10.c` implement head parse/write, size calculation, dequantization and scalar inference. `src/predictor.c` shares one state struct (union of the two nets) and the APM code between profiles 1 and 10. `src/fit.c` contains an AdamW fitter for the chain, including backpropagation through the inheritance weights and the clamps. `tests/test_p1.c` adds model validation, a whole-archive ISA-parity test, and golden vectors:

| quant | coded length | CRC-32C |
|-------|--------------|---------|
| F16 | 25079 | `0x8AB6CEC6` |
| Q8 | 91360 | `0xD0109680` |

Profile 1's golden vectors are unchanged: existing archives and decoders are unaffected.

## Results

`/tmp/opencode/tadamw-bench/{text1m,libc1m}`, 1 MiB each, `.aw` streams, `-p mix` vs `-p chain`. Archive bytes (primary signal); model bytes in parentheses.

| corpus | level | profile 1 (baseline) | profile 10 (chain) | delta |
|--------|-------|----------------------|--------------------|-------|
| text1m | 1 | 190610 (1888) | **186708 (91)** | −3902 B (−2.05%) |
| text1m | 5 | 187956 (2696) | **186084 (56)** | −1872 B (−1.00%) |
| libc1m | 1 | 388792 (1888) | **382858 (91)** | −5934 B (−1.53%) |
| libc1m | 5 | 385741 (2696) | **382583 (91)** | −3158 B (−0.82%) |

The chain is smaller on all four measured points. Its model section is 20–48× smaller than the MLP's while making better probability estimates on these inputs — direct evidence for the paper's "minimize total weights" claim.

Additional checks:
- `.awseek` chain roundtrip (text1m, L1): 186724 B, decodes correctly.
- Full `ctest --test-dir build --output-on-failure`: **10/10 pass** (including the new profile-10 tests).
- Clean build with `-Wall -Wextra -Wpedantic -Wconversion`: no warnings.

Explored and rejected: disjoint (non-nested) subsets (text L1 199984 B, worse than baseline), C0 selectors on the final unit (text L1 204678 B), and a plain single-layer 26-weight mixer. Nesting lower-order inputs into higher units was the decisive change: it lets the high-order unit correct the inherited estimate instead of reconstructing low-order statistics through one scalar.

## Verdict

**Positive but preliminary.** On the two benchmark corpora at levels 1 and 5, a 42–63-weight frozen chain beats a 1736-weight MLP by 0.8–2.1% in archive size while using ~2% of the model budget. This supports both of the paper's mechanisms in tadamw's setting. It is not yet a general replacement for profile 1.

## Caveats

- **Corpus-tuned heuristics.** The chain fitter's candidate prior (target `n/20000`), its 256-byte model cap and the 4× model-byte penalty in candidate scoring were chosen from measurements on these two corpora. They are not validated on other data. `TAW_CHAIN_FORCE=sel0,sel1,sel2,quant` is an internal experiment knob to fit one shape.
- **Biased sample estimator.** The shared feature sampler keeps an even spread of *bit indices*; at 1 MiB and the L1/L5 sample counts this is almost entirely MSBs, so the reported `bpc`/`score` is optimistic and is only used to rank candidates, never as a size prediction. At L9 the bias changes which shape wins and the chain regresses on libc (386330 B versus 378206 B for the MLP); the MLP also benefits more from L9's extra epochs. No conclusions about high effort levels are drawn here.
- **Fitting dynamics, not pure capacity.** The L1/L5 win is plausibly because a 42-weight model converges in 3–6 AdamW epochs while the 1736-weight MLP is under-trained at those levels. A weight-matched, longer-trained MLP comparison has not been run.
- **Inheritance domain.** The paper's estimate is a probability; this profile passes the monotone-equivalent clamped stretch value, which is exact and cheaper. The paper's per-unit weight-minimization rule is approximated by three fixed nested subsets rather than a data-dependent unit count.
- **No SIMD inference path** for the chain mixer; the tiny rows make it unnecessary, but decoder throughput was not measured (timings are noisy with many parallel agents). Only archive size is reported.
- Profile 10 is experimental and is refused by decoders that do not implement it; no silent fallback to profile 1.
