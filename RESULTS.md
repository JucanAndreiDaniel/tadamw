# Profile 12 `MDL_TREE` — MDL-pruned input-specific context tree

## Design

Adaptation of the context-tree idea in Bai et al., *Practical Lossless
Volumetric Medical Image Compression via Tri-plane Context Tree Learning*
(arXiv:2608.13897) to tadamw's bit stream. The paper learns an
input-specific adaptive tree on a sample of the input by minimizing MDL,
choosing among predictors at each node. Here:

- The 12 adaptive integer context models, the 25 stretched inputs, the
  squash/stretch tables, the two APM stages and the range coder are exactly
  profile 1's (no changes to coder state or rounding).
- A **shared hidden layer** (H units, selector `sel1`) produces profile-1
  hidden activations `h[j] = clamp(dot4(W1[a][j], x, 26), -8, 8)`.
- A small **binary tree** over ten cheap, decoder-visible integer features
  (bitpos, msel, c0/c1 nibbles, match flag, sign of three context inputs)
  selects one **linear output row** per bit. The row computes
  `z = dot4(w, h, H) + bias`, `t = trunc(z*256)`, then `p = squash(t)` and
  the unchanged APM path.
- The encoder fits on the collected bit samples. The first three levels are
  a fixed balanced `bitpos` tree (eight base experts, matching profile 1's
  per-bitpos output layer). Deeper splits are chosen greedily with a
  diagonal-Newton gain over all ten features, then **kept only if the exact
  integer coding gain of the quantised child models exceeds the MDL charge**
  for one node and one output row (8 bits per stored byte). Weights are
  fitted with AdamW and pass through the real row quantiser before scoring,
  so the MDL decision sees the decoder's probabilities.
- AUTO compares the tree's MDL estimate with the best fitted mixer and
  keeps the smaller. `-p mdl` forces the tree; `-p mix` forces profile 1.

## Format and code additions

- `docs/format.md`: profile id **12** in the profile table, §2.3
  (`MDL_TREE` inference and the ten split features), §5.2 (8-byte head,
  first-layer rows, four-byte nodes, per-leaf output rows; all validation
  rules). All row quant formats introduced upstream (F16, Q8, Q4, Q2, FP8,
  NVFP4, IQ8, IQ4, IQ2) are accepted.
- `src/p12.{h,c}`: model-section load/validate/serialize and scalar
  inference. `src/p1.h`: `taw_p1` gains a tree pointer and a `is_tree` flag.
  `src/predictor.c`: dispatch, APM, reset and free; the fused AVX2 bit loop
  is disabled for profile 12 (the tree walk is scalar).
- `src/fit.c`: tree fitting, quantisation-aware MDL pruning, and profile
  selection in `taw_model_fit_source` (kept alongside upstream's GPU,
  quant-format and bit-position-sampling code).
- `tools/taw.c`, `include/tadamw/tadamw.h`, `src/header.c`: `-p mdl` /
  `TAW_PREDICTOR_MDL` and validation.
- `tests/test_p1.c`: model validation (bad head, cycles, dangling leaves,
  bad feature/threshold, size mismatch), a roundtrip over **every** supported
  row quant, and golden vectors for profile 12 (Q8 `len 44239 / crc
  0x211B478A`, F16 `len 48918 / crc 0x51D83A69`).

## Results

New baseline (origin/main `e64cc83`, 1 MiB corpora, `-p mix`): text L1
181557, L5 179166; libc L1 379197, L5 375291 bytes. Archive bytes are the
primary signal; timings are noisy on a shared 12-core machine.

| corpus | level | baseline (`mix`) | forced `mdl` | AUTO | AUTO profile |
|--------|-------|------------------|--------------|------|--------------|
| text1m | 1 | 181557 | **180185** | **180185** | 12 |
| text1m | 5 | 179166 | 179737 | 179166 | 1 |
| libc1m | 1 | 379197 | **377850** | **377850** | 12 |
| libc1m | 5 | 375291 | 378140 | 375291 | 1 |

Forced tree at L1: −1372 B (−0.76 %) on text, −1347 B (−0.36 %) on libc.
At L5 the tree is 571 B (text) / 2849 B (libc) worse than the new mixer,
and AUTO correctly falls back to profile 1, so AUTO never regresses.

Model sections are 547–1200 bytes for the tree (vs 608 for the mixer).
Encoding with the tree is ~17–37 s vs ~9–46 s for the mixer; decoding is
~10–18 s vs ~3–9 s, because profile 12 cannot use the fused AVX2 bit loop.
These timings are indicative only.

**Stale numbers:** all measurements taken before `e64cc83` (e.g. text L1
190610 / L5 187956, libc L1 388792 / L5 385741 for the old baseline, and
the pre-rebase profile-12 numbers 189496 / 188849 / 387208 / 386536) are
**invalid**; upstream's bit-position sampler, new row quant formats and
fused decoder changed the baseline substantially. Only the table above is
current.

## Verdict

Partially successful. The MDL tree gives a small but consistent **L1 win on
both corpora** and, through AUTO, no regression at L5. It does not beat the
new, much stronger mixer at L5 when forced, and its scalar decode path is
slower. The experiment confirms that an MDL-pruned input-specific tree can
complement the frozen MLP at low model cost, but the gain is modest compared
with upstream's other improvements.

## Caveats

- All benchmark figures come from single runs on a shared machine; archive
  bytes are exact, times are not.
- The MDL estimate is computed on a sample with the decoder's probabilities
  but without APM adaptation, so it systematically under- or over-estimates
  the final archive; AUTO occasionally prefers the mixer even where the tree
  would be slightly better (and vice versa).
- Only a fixed H=12, `sel1=NONE`, Q8 tree configuration is used by default;
  environment overrides (`TAW_P12_H`, `TAW_P12_SEL1`, `TAW_P12_MIXEPOCHS`,
  `TAW_P12_LEAVES`, `TAW_P12_PENALTY`) exist for experiments but are not part
  of the format.
- Splits beyond the eight base experts are rare: the per-leaf model is
  already close to optimal after the bitpos experts, so the MDL charge
  rejects most deeper splits. The tree is therefore close to a specialised
  per-bitpos MLP with a few adaptive refinements.
- The paper's tri-plane volumetric context representation does not apply to
  this 1-D byte stream; only its MDL tree-learning principle is transferred.
