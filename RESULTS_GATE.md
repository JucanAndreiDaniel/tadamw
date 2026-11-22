# Experiment: entropy-adaptive confidence gating (arithmetic profile 11)

Branch `exp/entropygate`, rebased onto `origin/main` at `e64cc83` (previous
base `a427e7f`; old numbers below are explicitly stale). Reference:
StateSMix (arXiv:2605.02904), whose *entropy-adaptive scaling* modulates the
contribution of n-gram statistics by the neural model's predictive confidence
so that an already-calibrated model is not over-corrected.

> **Stale-numbers note.** The earlier results on the `a427e7f` base (text L1
> 190610 -> 188538, L5 187956 -> 186820; libc L1 388792 -> 386533, L5
> 385741 -> 382941) are invalid. Upstream then landed bit-position sampling,
> new weight formats, a fused AVX2 decoder, parallel awseek decode and GPU
> fitting, which changed both the baseline and the fitted gate. Only the
> table below, measured after the rebase, is current.

## Design

Profile 11 (`ENTROPY_GATE`) is a strict extension of profile 1. The context
models, frozen MLP, integer squash, both APM stages and the range coder are
unchanged. Between the MLP and the APM stages we insert a deterministic
integer gate:

1. From the 25 stretched context estimates `st[0..24]` (the exact integers
   behind the mixer inputs `x[i] = st[i]/256`) compute
   `mean = round(sum(st)/25)` and `mad = round(mean |st[i] - mean|)`.
   `mad` is a cheap integer proxy for the models' disagreement/confidence.
2. Compute the raw context mixture `p_ctx = squash(mean)`.
3. With two signed 16-bit parameters `a, b` stored in the archive, compute
   `w = clamp(a + floor(b*mad/64), 0, 256)` and

   `p_gate = ((256 - w) * p_mlp + w * p_ctx + 128) >> 8`.

4. Feed `p_gate` to the same APM stages as before.

`a = b = 0` gives `w = 0` and bit-identical profile-1 output; a test asserts
this. All post-rebase fits produced **negative `b`**: the raw context mixture
is favoured when the context estimates agree (low `mad`, i.e. the neural
prediction is already well calibrated), and the frozen MLP is trusted when
they disagree. That is the intended StateSMix analogue.

## Format additions (`docs/format.md` §2.3, §5.2)

- Header `arith_profile = 11`; all quant formats profile 1 accepts are
  accepted (the upstream Q4/Q2/FP8/NVFP4/IQ8/IQ4/IQ2 additions work unchanged
  because the gate tail is orthogonal to weight quantization).
- Model section = the §5.1 head and weight payload plus a 4-byte tail:
  two little-endian signed int16 values `gate_a` (`-256..512`) and
  `gate_b` (`-2048..2048`). `model_size = 8 + payload + 4`.
- Decoder refuses an invalid tail, size mismatch or unsupported quant.
- Loader, scalar inference, fitter and golden vectors share the same integer
  helper functions (`p11_features`, `p11_gate_p`).
- The upstream fused AVX2 per-bit loop now computes the same gate from the
  integer stretch vectors already in registers, so profile 11 keeps the
  upstream decoder speed instead of falling back to the generic loop.
  Golden vectors pin scalar and AVX2 gated output to identical bits.

Encoder: after fitting the network as for profile 1, `src/fit.c` collects at
most 65,536 evenly spread training bits, caches `(p_mlp, p_ctx, mad, bit)`
and finds `a, b` by a coarse grid plus local refinement on the exact integer
cost. `-p gate` (`TAW_PREDICTOR_GATE`) forces profile 11 for benchmarking;
`auto` and `mix` are deliberately unchanged so the baseline is untouched.

## Results (post-rebase)

Archives were rebuilt with the same binary; corpora in
`/tmp/opencode/tadamw-bench` were read only. Archive size is the primary
signal. Contemporaneous `mix` runs matched the supplied new baseline exactly.

| corpus | level | new baseline `mix` (B) | profile 11 `gate` (B) | delta (B) | delta (%) | model bytes (mix → gate) | fitted a, b |
|--------|-------|-----------------------:|----------------------:|----------:|----------:|-------------------------:|-------------|
| text1m | 1 | 181557 | 181473 | -84   | -0.046% | 608 → 612 | 286, -64 |
| text1m | 5 | 179166 | 179098 | -68   | -0.038% | 608 → 612 | 238, -64 |
| libc1m | 1 | 379197 | 378070 | -1127 | -0.297% | 608 → 612 | 224, -56 |
| libc1m | 5 | 375291 | 374705 | -586  | -0.156% | 608 → 612 | 258, -64 |

All four gated archives were decompressed and byte-compared with their
sources. Full `ctest` passes (12/12), including profile-11 golden vectors for
F16 and Q8 over scalar and AVX2 ISAs, gate validation, zero-gate identity,
an end-to-end encoder/header/decoder test, an awseek roundtrip, and all new
upstream tests (quant, GPU, parallel awseek).

Timings (single noisy run, many agents sharing 12 cores, best of 3 decodes;
treat as indicative only):

| corpus | level | mix enc / dec (s) | gate enc / dec (s) |
|--------|-------|-------------------|--------------------|
| text1m | 1 | 17.72 / 8.504 | 21.39 / 8.290 |
| text1m | 5 | 40.16 / 6.546 | 45.22 / 6.601 |
| libc1m | 1 | 13.18 / 6.815 | 19.26 / 7.087 |
| libc1m | 5 | 30.05 / 4.925 | 32.51 / 4.736 |

Decode cost is now essentially equal to `mix` (the gated fused AVX2 loop);
encoding costs the extra one-off gate fit and is somewhat slower.

## Verdict

**Still positive, but much smaller than before the rebase.** The gate improves
archive size on all four measured corpus/level combinations (0.04%–0.30%)
for four extra model bytes, and the fitted parameters agree in sign and
magnitude. Upstream's improvements (notably better bit sampling and more
weight formats) already captured much of the headroom the gate previously
exploited: gains on text shrank from ~0.6–1.1% to ~0.04%, and on libc from
~0.6–0.7% to ~0.16–0.30%.

## Caveats / follow-ups

- The fitter optimizes pre-APM probability; the APM stages afterwards are
  adaptive and stateful, so the chosen `(a, b)` is only an approximation of
  the optimum of the final coded cost. Fitting through a simulated APM pass
  could improve results further.
- Gate fitting uses a subsample (≤65,536 bits) of the collected samples and a
  discrete grid; it is deterministic but not a globally optimized continuous
  fit.
- Evidence is two 1 MB corpora at levels 1 and 5. `-p gate` forces profile 11
  even if the fit finds `a=b=0`, in which case the archive is 4 bytes larger
  than profile 1. The observed gains are small enough to be within the noise
  of a different fitting seed or corpus.
- `auto`/`mix` were intentionally left unchanged to keep the baseline
  comparable. Given the consistent (if small) gains, a follow-up could fit
  the gate in the default path and let `auto` choose between profiles 1 and
  11 on estimated cost.
- The new predictor enum value and `-p gate` CLI option extend the public API;
  archives produced with profile 11 are refused by decoders that only know
  profiles 0 and 1, as required by the format's strict "decode or refuse"
  rule.
