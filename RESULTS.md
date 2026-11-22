# Experiment: soft-gated mixture-of-experts + learned APM blend

Branch `exp/gated`, rebased onto `origin/main` at `e64cc83` (9 upstream
commits: GPU fitting, Q4/Q2/FP8/... quant formats, bit-position sampling,
fused AVX2 profile-1 decoder, parallel awseek decode). Two local commits:
`435b2dd` (phase 1, profiles 3) and `2a5cd5f` (phase 2 WIP, profile 7).
Final commit adds the spec, goldens and this report. Nothing was pushed.

## What works

* **Profile 3 `SOFT_GATED`**: K=2..4 experts, exact integer-softmax gate over
  `c0`, match class and a 16-level confidence bucket (mean absolute deviation
  of the 25 stretched inputs), with an optional fixed raw-context-mixture
  expert. Contexts, squash/stretch, APM and range coder are profile 1's.
* **Profile 7 `SOFT_GATED_CAL`** (phase 2): profile 3 plus 8 bytes of fitted
  4-bit APM blend weights `w0,w1,w2`, replacing
  `(2*p_mix + q1 + q2) >> 2` with
  `p = (w0*p_mix + w1*q1 + w2*q2) / (w0+w1+w2)`. All integer, fixed order;
  identity `(2,1,1)` reproduces profile 3 exactly.
* Loader/dequantizer/scalar inference (`src/p3.c`), predictor and CLI support,
  encoder fitting (AdamW experts/gate; blend fitted by coding cost on a 16 KiB
  held-out probe), format spec (`docs/format.md` §2.3, §2.4, §5.2, §5.3),
  golden vectors and validation in `tests/test_p1.c`, and test generalization
  for the new profiles.
* Final profile/candidate choice is made by encoding the whole input
  (capped at 1 MiB) with each fitted candidate: the sample-based score omits
  the APM stages and mis-ranks mixers.
* Full `ctest` passes **12/12** on the rebased tree: tests 1-9 completed in a
  run interrupted by a server restart, tests 10-12 were re-run afterwards and
  all passed (including the new profile 7 golden vectors). All profile 1
  golden vectors are unchanged.

## Format additions

* Profile table: 3 `SOFT_GATED`, 7 `SOFT_GATED_CAL` (2 and 4..6, 8..255
  reserved). Upstream's extended quant table for profile 1 is untouched;
  profiles 3/7 accept `F16`/`Q8`.
* §2.3/§5.2: gate logits, `E[d]=(E[d-1]*235)>>8`, prefix-sum normalization to
  weights summing to 256, expert evaluation, weighted accumulation, model
  head (8 bytes) and payload layout.
* §2.4/§5.3: learned APM blend arithmetic and 16-byte head (8-byte profile 3
  head + `w0,w1,w2` + reserved), validation rules and identity values.

## Benchmarks (post-rebase, new clean baseline)

Archives made with `-p mix` on `/tmp/opencode/tadamw-bench/text1m` and
`libc1m` (1 MiB each). Archive bytes, including model and framing.
The machine was heavily loaded (load average ~27-35) during these runs, so
timings are not meaningful; sizes are deterministic.

| corpus | L | new baseline (p1) | p3/p7 result | delta | chosen profile / model |
|--------|---|-------------------|--------------|-------|------------------------|
| text1m | 1 | 181557 | **179820** | −1737 (−0.96%) | 7 (K=4 H=0 Q8 + confidence), 1184 B |
| text1m | 5 | 179166 | 179166 | 0 (0.00%) | 1 (upstream's 608 B model wins) |
| libc1m | 1 | 379197 | **372999** | −6198 (−1.63%) | 7 (K=4 H=0 Q8), 1248 B |
| libc1m | 5 | 375291 | **372007** | −3284 (−0.88%) | 7 (K=4 H=0 Q8), 1248 B |

All four archives round-trip; `taw i` reports the profile shown.

### Comparison against sibling profiles

| corpus | L | this work | profile 6 (new, rebased) | profile 9 ensemble (old base) |
|--------|---|-----------|--------------------------|-------------------------------|
| text1m | 1 | **179820** | 180874 | 184060 |
| text1m | 5 | 179166 | **178997** | 182655 |
| libc1m | 1 | **372999** | 375305 | 383060 |
| libc1m | 5 | **372007** | 373312 | 380717 |

Profile 6 numbers are from sibling branch `exp/calibration` commit `50b7f89`
("Rebase profile 6 calibration onto new main and refresh benchmarks"), i.e.
measured on the same new baseline. Profile 9's numbers were supplied by the
coordinator but are **stale** (old baseline 190610/187956/388792/385741);
its branch had not been rebased at the time of writing, so that column is
only indicative — this work's absolute sizes are lower than all of them.

### Stale numbers from before the rebase (do not use)

Phase 1 on old `main`: text L1 188907, L5 186991; libc L1 381688, L5 383663
against the old baseline 190610/187956/388792/385741. Old profile 6 numbers
186508/186098/384244/382830 and old profile 9 numbers
184060/182655/383060/380717 are likewise superseded.

## Verdict

The combination beats profile 6 in three of four cases (best: −2306 bytes,
−0.61%, on libc L1) and is within 169 bytes of it on text L5, where the
fitter keeps upstream's profile 1 model. It improves the new profile-1
baseline by 0.88–1.63% on libc and 0.96% on text L1. The learned blend is the
larger part of the gain: fitted weights are APM-dominant (`(0,2,1)` or
`(0,1,0)` most often), confirming that the fitted mixer only reaches the
coded probability through the APM stages — an improvement to the mixer alone
has at most half the effect. The extra 8 bytes pay for themselves everywhere
profile 7 is selected.

Against profile 9 the comparison is inconclusive because its published
numbers predate the upstream baseline change; on absolute archive bytes this
work is smaller on all four cases.

## Caveats / incomplete aspects

* Profile 3/7 inference is scalar (SIMD optional); profile 1 keeps upstream's
  fused AVX2 path. ISA parity golden vectors cover profile 1 only; profiles
  3/7 have scalar golden vectors with fixed order.
* `hidden > 0` with the confidence flag is forbidden by the format (the fixed
  context expert lives in the input domain).
* Some K=4/H=0 gate fittings collapse toward near-hard routing at higher
  effort; the real-codepath ranking rejects them (text L5 keeps profile 1).
* Candidate ranking re-encodes up to 1 MiB per fitted candidate, so encoding
  is much slower than baseline; a cheaper accurate score is future work.
* Timings in this report are unreliable because sibling agents were running
  large benchmarks concurrently; only archive bytes should be compared.
