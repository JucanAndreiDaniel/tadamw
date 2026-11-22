# Experiment: PAQ-style adaptive logistic mixing (arithmetic profile 8)

Branch `exp/paq`, rebased onto `origin/main` at **`e64cc83`** (GPU fitting,
new Q4/Q2/FP8/NVFP4/IQ8/IQ4/IQ2 rows, bit-position sampling, fused AVX2
decoder, parallel awseek decode). The original commit was based on `a427e7f`.

> **Stale-numbers note:** the results in the first version of this report
> (profile 1 190610/187956/388792/385741; profile 8 189818/187829/389308/
> 385336) were measured on the old base and are **invalid**. Upstream's new
> sampler and fused decoder moved the baseline down by roughly 9 KB on text
> and 9-10 KB on libc. All numbers below were measured after the rebase.

## Design

Profile 1 fits a small MLP on the whole input and freezes its weights; only
the integer counters/APM adapt while coding. This experiment keeps all of
profile 1's feature extraction and APM machinery and replaces the output
layer with PAQ-style adaptive logistic mixing:

- The 12 adaptive integer context models (25 stretched features), the
  squash/stretch tables, the match model, the selectors and the two APM
  stages are exactly profile 1's (`src/p1_ctx.c`, `predictor.c`).
- A frozen first layer (`W1`, same shape and row quant formats as profile
  1) turns the features into `H` experts `h[j] = clamp(dot4(W1[a][j], x),
  -8, 8)`, or, when `H = 0`, the raw stretches are the experts.
- The output layer has a **frozen binary32 part** (the initial weights from
  the archive, dequantized exactly as profile 1 does) plus a **live int32
  delta** per expert, starting at zero, updated after every bit with a
  stored uint8 rate and an integer logistic gradient step (64-bit products,
  truncation toward zero, int16 saturation of the deltas).

The frozen base is computed with profile 1's exact output-layer arithmetic,
so **with all rates zero, profile 8 is bit-identical to profile 1** (verified
by re-labelling a profile 1 archive as profile 8 and decoding it: identical
coded bitstream and output). The adaptive delta is a pure add-on correction.

Selection is explicit: `taw c -p paq` (`TAW_PREDICTOR_PAQ`) fits profile 8;
`auto` and `-p mix` still produce profile 1, so existing archives and golden
vectors are unaffected.

Encoder fitting (`src/fit.c`) fits `W1`/`W2` with the existing AdamW code,
quantizes them into the new model section, then fits one shared rate from
`{0,1,2,4,8,16,32}` on the first half of a 128 KiB feature window taken from
the middle of the input. The second half (with APM replayed, so the estimate
sees the probability reaching the range coder) selects among five shapes and
the rate. Rate 0 is a candidate, so the encoder can always fall back to the
frozen mixer.

## Format additions

`docs/format.md` §2.3 and §5.2 define profile id **8 `PAQ_MIX`** (quant
`NONE`):

- 12-byte model head: `lines_log2`, `hist_log2`, `match_log2`, `sel1`,
  `hidden`, `sel2`, `q1` (F16/Q8), `q2` (F16/Q8), `ushift` `U` (4..31),
  three reserved bytes.
- Payload: frozen `W1` rows, initial `W2` rows (both in profile 1's row
  format, so all row quantizers apply), then `E = H+1` (or 26 when `H = 0`)
  rate bytes.
- Mixer: `base` = profile 1's output layer; `delta =
  trunc(sum_i dw[c][i] * e[i] / 2^12)`; `t = clamp(base + delta, -2047,
  2047)`; `p_mix = squash(t)`; APM exactly as profile 1.
- Update: with `err = (bit ? 65536 : 0) - p_mix`,
  `g = trunc_toward_zero(err * e[i] * rate[i] / 2^U)`,
  `dw[c][i] = clamp_s16(dw[c][i] + g)`. Truncation toward zero (not floor)
  keeps positive and negative steps symmetric; floor would add a systematic
  -1 unit drift. Products are int64, deltas saturate to int16.

The arithmetic lives in `src/p8.{h,c}` and is shared by encoder, decoder and
the fitter's replay. The per-bit state is in `struct taw_p1` (`src/p1.h`), so
upstream's fused AVX2 loop is explicitly bypassed for profile 8.

## Tests

`tests/test_p1.c` gains model validation (head ranges, reserved bytes, size
mismatch, non-finite F16/scale, Q8 `-128`, bad `quant`), two golden vectors
(H=8, Q8 output weights, nonzero rates: len 18931, CRC `0xCC99DFE4`; H=0,
F16 output weights: len 17746, CRC `0xE9855CB2`), and whole-archive
encode/decode parity across scalar/AVX2/AVX-512. Full `ctest`: **12/12 pass**
after the rebase. Profile 1 goldens are unchanged.

## Benchmarks

`text1m`, `libc1m`; `.aw`, levels 1 and 5; archive bytes (smaller is
better). Baseline is profile 1 from the rebased build and matches the new
reference exactly.

| corpus | level | profile 1 (B) | profile 8 (B) | delta (B) | delta |
|--------|------:|--------------:|--------------:|----------:|------:|
| text1m | 1 | 181557 | 183075 | +1518 | +0.84% |
| text1m | 5 | 179166 | 179564 | +398 | +0.22% |
| libc1m | 1 | 379197 | 375970 | **-3227** | **-0.85%** |
| libc1m | 5 | 375291 | 374616 | **-675** | **-0.18%** |
| **total** | | 1115211 | 1113225 | **-1986** | **-0.18%** |

Chosen shapes and rates: text L1/L5 `H=16, sel1=NONE, sel2=BITPOS`, rate 0
(model 629 B); libc L1 `H=8, sel1=NONE, sel2=C0`, rate 4 (model 3061 B);
libc L5 `H=4, sel1=BITPOS, sel2=C0`, rate 4 (model 2705 B). Unlike the
pre-rebase run, **online adaptation is now selected at both libc points**.
The bit-position sampler changed the fitted weights enough that a nonzero
rate helps there. On text, whose profile 1 baseline is the strongest,
rate 0 remains the encoder's choice and profile 8 loses only from its larger
model section (629 B vs profile 1's 608 B) and from a slightly different
fit of the same shape.

## Verdict

**Net win, but uneven.** Total archive bytes improve by 1986 B (-0.18%),
driven by libc L1 (-3227 B) and libc L5 (-675 B); text regresses by 1518 B
at L1 and 398 B at L5. Two of four points win, and the wins are larger than
the losses. The result is still within the noise of the shape-selection
heuristic, so this is evidence that the mechanism works rather than a
conclusive victory.

The rebase changed the picture materially: upstream's bit-position sampler
made online weight adaptation useful on libc (rate 4 selected at both
levels), whereas the old sampler always chose rate 0 because offline-fitted
weights were already near-optimal. Keeping the frozen base exact also means
a rate-0 profile 8 archive can never be worse than the equivalent profile 1
model except for its 21-byte header/rate overhead.

## Caveats

- Profile 8 is opt-in (`-p paq`); `auto` is unchanged, so no existing
  archive or test vector changes meaning. Decoders still refuse unsupported
  profiles/quants.
- Shape/rate selection uses a 128 KiB window from the middle of the input
  with replayed APM. On small files (< ~40 KiB) no shape fits the `n/32`
  model budget and `-p paq` returns `TAW_E_ARG`.
- Fitting is slower than profile 1: five shapes are trained and each shape
  replays seven rates over a dense feature window with APM (roughly 30-50 s
  per 1 MiB compression here, versus a few seconds for profile 1 — timings
  are noisy on the shared 12-core box). Decoding is essentially the same
  cost as profile 1; the adaptive mixer is scalar even on AVX2/AVX-512
  machines (context-feature parity across ISAs is still tested).
- The benchmark corpora and pre-existing archives were not modified.
  Archive byte counts, not wall-clock time, are the signal used here.
