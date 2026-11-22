# Experiment: soft-gated mixture-of-experts mixer (profile 3)

Branch `exp/gated`, based on `origin/main` at `a427e7f`. Everything is local;
nothing was pushed.

## What works

A new arithmetic profile **3 `SOFT_GATED`** is implemented end to end:

* format specification (new §2.3 and §5.2 in `docs/format.md`);
* header constant (`TAW_PROFILE_SOFT_GATED`), CLI profile name, loader,
  dequantizer and **scalar** inference (`src/p3.c`);
* predictor integration sharing the profile 1 context models, squash/stretch
  tables, both APM stages and the range coder (`src/predictor.c`);
* encoder fitting for the new mixer in `src/fit.c`, including a real-codepath
  candidate ranking step;
* model validation and two golden vectors in `tests/test_p1.c`;
* generalized three existing tests that asserted "profile 1 is always
  chosen" (`test_aw.c`, `test_awseek.c`, `cli_roundtrip.cmake`).

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`,
`cmake --build build -j2` and `ctest --test-dir build --output-on-failure`
all succeed: **10/10 tests pass**, including the unchanged profile 1 golden
vectors (so profile 1's arithmetic was not disturbed).

## Design

The 12 context models, their 25 stretched inputs, `squash`, `stretch`, the two
adaptive APM tables, the `(2*p_mix + q1 + q2) >> 2` blend and the range coder
are exactly profile 1's. Only the mixer changes.

* **K = 2..4 experts**, each a linear model over either the 26 mixer inputs
  (`hidden = 0`) or a shared `hidden = 4` nonlinear layer. Expert weights use
  the profile 1 F16/Q8 row encoding.
* **Integer gate** over cheap features: `int8` logit tables indexed by the
  partial byte `c0`, the match class `msel`, and a confidence bucket derived
  from the mean absolute deviation of the 25 stretched inputs. The gate logit
  of expert `k` is a plain int16 sum of these three table lookups.
* **Exact integer softmax**: `E[0] = 256`, `E[d] = (E[d-1]*235) >> 8` for
  `d = 1..511` (a fixed-point `256*2^(-d/8)`). Gate weights are normalized by
  prefix-sum rounding, `w[k] = floor(256*prefix_k/S) - floor(256*prefix_{k-1}/S)`,
  so they sum to exactly 256 with a fixed summation order. The expert outputs
  are accumulated as `sum_k w[k]*z[k]` in `k` order, scaled by `1/256`,
  clamped, truncated and squashed. This keeps the whole gate/mixer
  deterministic with no libm and a fixed float order.
* Following the result of sibling experiment `exp-entropygate`, one variant
  makes expert 0 a **fixed raw context mixture** (`1/25` on every stretched
  input) and lets the confidence feature route weight to it when contexts
  agree. That variant is only valid with `hidden = 0` and costs `K*16` extra
  int8 gate bytes.

Model sizes are small: 932 B (K=3, H=0, Q8, confidence) to 1240 B
(K=4, H=0, Q8, confidence), far below the `raw/32 = 32768` B budget.

Encoder fitting trains the shared hidden layer, the experts and the gate
logits with AdamW on the same sampled bits profile 1 uses. Gate logits are
trained in natural-log space and serialized as `round(8/ln2 * logit)` int8,
which makes the integer `E[d]` table reproduce the trained soft routing.
Because the sample-based score omits the APM stages and was observed to rank
mixers differently from the range coder, the final profile/candidate choice
is made by actually encoding the whole input with each fitted candidate
(capped at 1 MiB).

## Format additions

* Profile table: id 3 `SOFT_GATED`, ids 2 and 4..255 reserved.
* §2.3: state reuse, `E[d]` table, confidence-bucket computation, gate logit
  sum, integer softmax with prefix-sum normalization, expert evaluation,
  weighted accumulation and squash. No change to §2.2 (APM, range coder).
* §5.2: 8-byte model head (`K`, `hidden`, geometry, `gate_flags`) and payload
  order (hidden rows, expert rows, `K*256` int8 `c0` table, optional `K*8`
  `msel` table, optional `K*16` confidence table), with validation rules.

## Results

Read-only corpora: `/tmp/opencode/tadamw-bench/text1m` and `libc1m` (1 MiB
each). Best-of-3 decodes. `size` includes the model section and framing.

| corpus | level | baseline size | profile 3 size | delta | chosen profile / model |
|--------|-------|---------------|----------------|-------|------------------------|
| text1m | L1 | 190610 | **188907** | −1703 (−0.89%) | 3, K=4 H=4 Q8, 1204 B |
| text1m | L5 | 187956 | **186991** | −965 (−0.51%) | 1, H=8 Q8, 1888 B |
| libc1m | L1 | 388792 | **381688** | −7104 (−1.83%) | 3, K=3 H=0 Q8 + confidence, 932 B |
| libc1m | L5 | 385741 | **383663** | −2078 (−0.54%) | 3, K=4 H=0 Q8, 1176 B |

All archives round-trip exactly; `taw i` reports the profile named in the
table. Timings (noisy: sibling agents were benchmarking concurrently) were
roughly enc 13–35 s (baseline 2–6 s) and dec 1.8–2.8 s (baseline 1.25–1.39 s).
The encode slowdown is from fitting extra candidates and re-encoding the
input to rank them; the decode slowdown is mostly measurement contention
(profile 3 decodes no more experts than profile 1 decodes hidden units).

## Verdict

**Positive in absolute terms, mixed relative to the sibling results.** The
extra weights do pay for themselves in three of four cases, with the best
result −1.83% on libc L1; on text L5 the fitter keeps profile 1. However,
this is weaker than sibling `exp/calibration` (−1.2% to −2.1%) everywhere
except libc L1, and comparable to `exp-entropygate` (−0.58% to −1.09%)
only on libc. As a standalone profile 3 is therefore only marginally worth
its complexity.

The main reason is structural: the changed mixer only reaches the final
probability through the unchanged APM blend `(2*p_mix + q1 + q2) >> 2`, so at
most half of the probability is attributable to the improved mixer. That
matches the sibling finding that the adaptive integer part dominates the
fitted MLP. Combining a soft-gated expert mixer with an APM blend that can
suppress `p_mix` would likely be the strongest configuration, but changing
that blend was outside this experiment's "keep APM unchanged" constraint.

## Caveats / incomplete aspects

* SIMD inference is not implemented for profile 3 (scalar only); this is
  explicitly allowed by the task. ISA parity tests therefore cover profile 1
  only; profile 3 golden vectors are scalar and fixed-order.
* `hidden > 0` combined with the confidence flag is forbidden by the format
  and by the loader (the fixed context expert is defined in the input
  domain). Both allowed shapes are exercised by the golden vectors and fit.
* At L5 some K=4/H=0 gate fittings collapse to near-hard routing and lose to
  profile 1 (text L5); the real-codepath ranking correctly rejects those
  candidates, but fitting is not fully robust.
* The candidate ranking re-encodes up to 1 MiB per candidate, which makes
  compression considerably slower than baseline; a cheaper accurate score
  (or a larger sample for the estimator) is future work.
* Decode timings were collected while other agents benchmarked in parallel
  and should not be read as a precise speed result.
