# tar_aw design

## Goal

A general-purpose compressor in the spirit of zstd/xz, with a different
predictor. The encoder fits a small model to the input it is compressing,
using AdamW on the bitwise cross-entropy, and embeds the frozen weights
in the archive. The decoder only runs inference. The bit probabilities
drive a binary range coder (format.md §4).

## Decisions so far

| topic | decision |
|-------|----------|
| fitting | offline, encoder only; the decoder uses frozen weights |
| predictor | bitwise mixer: hashed order-N context features → small MLP → P(bit=1) |
| model budget | `model_size <= raw_size / 32`; a heuristic picks the size |
| level | effort only (epochs, number of candidate sizes), not a size knob |
| weights | stored in a tagged quant format; F16 is standard |
| arithmetic | tagged arithmetic profile; decoders refuse unknown profiles/quants |
| variants | `.aw` stream, `.awseek` seekable with one global model |
| language | C23 inside, CMake + CTest; libtadamw needs libc and pthreads; `taw-nx` also needs OpenSSL 3 and LZ4 |
| library | `libtadamw` (static or shared), `<tadamw/tadamw.h>` usable from C99+/C++, encoder and decoder halves selectable at build time |
| portability | every SIMD path has a generic one with identical results; byte-wise little-endian I/O, so big-endian hosts (ppc64) work too (not yet run on one) |
| license | GPL-3.0-or-later, since `taw-nx` borrows ScarletNX's hactool code |
| Switch dumps | an ordinary `.awseek` with the dump (NCAs and gamecard info decrypted) and `/.meta`; decoding encrypts again and checks a SHA-256 of the original |
| awseek extras | file directory (format.md §7.1), stored blocks/chunks, parallel streaming encoder (`taw_seek_encode`) whose output does not depend on the thread count |
| libm | never used, in the encoder or the decoder; squash/exp go through SIMD intrinsics or tables |
| model shape | table/MLP split chosen heuristically per input, not fixed |
| weight formats (profile 1) | F16 and INT8 (Q8) from the start; then Q4, Q2, FP8 (E4M3), NVFP4, IQ8, IQ4, IQ2 (format.md §3, §5.1) |
| SIMD | scalar reference + AVX2/F16C and AVX-512 paths (ideas: ~/ucas, ~/q2_kernel kernels) |
| GPU | the encoder may use a GPU for fitting, as a hybrid. Decoders are CPU by default and may opt in to a GPU (hybrid or fully). A CPU decoder is always available and is never dropped |
| profile 1 inputs | 12 adaptive integer context models (o0-o4, o6, word, 5 sparse), each giving a direct and an indirect (bit history) estimate, plus a match model: 25 inputs + bias |
| profile 1 mixer | frozen MLP: one optional hidden layer (clamp ±8), weight sets picked by a selector context (the "table" part), fixed `dot4` summation order |
| profile 1 output | integer squash, then two adaptive integer APM stages (c0, c1:c0) |

## Profile 1 in short

format.md §2.2 is normative; this is the reasoning.

- **Adaptive features, frozen mixer.** The decoder has to adapt to local
  statistics somehow. Profile 1 keeps all adaptation in deterministic
  integer state (counters, bit histories, the match model, APM) and
  freezes only the mixer. Two things follow: the features depend only on
  the data, never on the weights, so the encoder computes them once and
  fits on stored samples; and all float work is one small MLP.
- **Two estimates per context.** A slot holds an adaptive-rate
  probability and the slot's last 7 bits. The bits go through a per-model
  adaptive map (history to probability). Measured on the bench corpus, a
  single counter wanted a different adaptation rate per file (limit 2 to
  30); the pair lets the fitted mixer choose. This saved about 3% over
  the best single-rate counter.
- **Tagged lines.** Hashed models use 64-byte lines (one nibble's bit
  tree per line, so one cache miss per model per nibble) with a 16-bit
  tag. A mismatched line is reset instead of inheriting another context's
  statistics: 2.5% smaller. Two-way lines were not consistently better.
- **Sparse contexts** (bytes 2+3, 3+4, 2+4, 1+4, 1+2+7+8 back) are for
  binaries: libc went from 327 KB to 307 KB, with text unchanged.
- **APM** adds about 2%.
- **Determinism.** The only float code is `dot4` products, clamps and one
  truncation, in an order the spec fixes. Each SIMD lane computes one
  hidden unit in exactly the scalar order. The output layer's `dot4` maps
  onto SIMD exactly too: lane r of each 4-wide half belongs to partial sum
  `s_r`. So scalar, AVX2 and AVX-512 agree by construction, and the tests
  check it. Four partial sums (rather than one running sum) shorten the
  add dependency chain without giving up exactness.

## Decoder performance

Measured on an i5-9400 (AVX2, no AVX-512), 1 MB inputs at level 1:

| path | words1m | libc1m |
|------|---------|--------|
| scalar | 2.8 s | 2.9 s |
| AVX2 before the context-model kernels | 1.58 s | 1.68 s |
| AVX2 now | 1.20 s (0.83 MB/s) | 1.35 s (0.74 MB/s) |

Per bit, profile 1 runs one arithmetic step per context model, 12 times
over. That shapes the code:

- **Data layout.** All context tables are one arena of u32 slots
  (`[O0][O1][T_2..T_11]`) addressed by u32 indices, not pointers. The
  per-bit hot state is a set of small parallel arrays padded to 16 lanes:
  line bases, slot indices, slot values, history-map indices and values.
  The AVX2 kernels (`p1_avx2.c`) work on those directly: slot indices are
  `line base + nib`, slot values come in by gather, and both counters of
  all 12 models update in two 8-lane halves.
- **Store forwarding mattered most.** Vector loads of arrays just filled
  by scalar stores cannot forward. That cost about 6 blocked loads per bit
  (48.8M on a 1 MB input) until values were built in registers: gather
  slots straight from the arena, assemble history-map entries with
  `set_epi32`, build hash lanes from scalars, and keep the hidden layer in
  registers through the whole mixer. Blocks are now about 0.15 per bit.
- **Gathers vs scalar loads.** Slots are gathered: within a nibble every
  bit uses a different slot, so no pending store aliases them.
  History-map entries are often the ones the previous bit just wrote, and
  gathers cannot forward from pending stores, so those are scalar loads.
- **Exact SIMD shortcuts.** The bit is the same in every lane, so each
  counter needs one product (`bit ? max - p : p`). `R[n]` is
  `cvtt(2^17 / (2n + 3))` in binary32: the quotient's rounding error,
  at most 2^-7/d, is below the 1/d gap to the next integer, so it
  truncates correctly for every `n`. That replaces two dependent gathers.
- **Tried, not kept:** computing the next nibble's or byte's hashes a bit
  early for both outcomes and prefetching their lines was about 3% slower
  than computing them once at the boundary. Scalar table lookups instead
  of the stretch gathers were 7% slower. `-O3` gains 1-2%.

- **Evaluating the next bit's outcomes ahead** (branch
  `exp/two-outcome-decode`). While bit i is decoded, the next bit's
  prediction is prepared from the state before bit i's update. That is
  exact: within a nibble bit i+1 reads other slots and APM rows, and the
  history-map entries and match counter it may share with bit i come from
  bit i's update precomputed for that outcome. Measured (words1m /
  libc1m, level 1): main 1.23 / 1.42 s; both outcomes 1.81 / 2.07 s; only
  the likely outcome (redo on a miss) 1.29 / 1.48 s. The per-candidate
  records cost the sequential path about 5% on their own.

Before the second round below, the decoder ran about 675 cycles and
1060 instructions per bit at IPC 1.6. Evaluating more than one outcome
multiplies the work, so it loses.

### Second round (fused loop)

pmcstat on libc1m (8.4M bits), the same archive throughout:

| step | cycles | instructions |
|------|--------|--------------|
| before | 5.66 G | 8.89 G |
| both-outcome hashes + prefetch at the boundaries | 5.51 G | 9.58 G |
| mixer: partial sums in registers | 5.12 G | 8.31 G |
| counter update on packed fields (identities below) | 5.0 G | 7.96 G |
| one fused AVX2 per-bit loop | 4.27 G | 6.93 G |
| vector line selection, A2 row prefetch, inline tail | 4.0 G | 6.47 G |

Wall clock, `taw d` per 1 MiB file at level 1: 1.24-1.37 s before,
0.93-1.08 s now (20-25% less). Encoding uses the same loop and got about
10% faster.

- **Fused loop** (`p1_run_avx2.c`). One function per mode (encode,
  decode, skip) runs inputs, mixer, squash, APM, coder and counter
  updates. Slot values and history-map entries stay in registers from
  the gathers to the update; the mixer, squash and selectors are inline
  (no function pointer). Only nibble and byte boundaries call out.
- **The mixer kept its partial sums in memory**: `s[i & 3]` with a loop
  index made the compiler keep the four sums in a stack array, so every
  product went through a store and a reload. Four named registers keep
  dot4's exact order.
- **Boundary speculation, kept this time.** One bit before a nibble or
  byte boundary only two outcomes remain. Both outcomes' line hashes
  (and, before a byte end, byte hashes and the match-table index) are
  computed, their lines prefetched, and the boundary then picks one, so
  nothing is computed twice. L3 misses fell from 6.1M to 1.0M. Measured
  against no speculation it is 3-5% faster. (The earlier attempt
  recomputed the hashes at the boundary.)
- **The A2 row** (`c1:c0`, 4.3 MB table) is known at the start of a
  bit; prefetching it there hides its latency behind the mixer.
- **Line selection** (tag check and reset of 10 lines) is one gather and
  a compare, with a loop over the mismatches only.
- **Counter identities**, proved with ~/ucas (`equiv`): `65535 - ps =
  ps ^ 0xFFFF` and `2^22 - 1 - p = p ^ 0x3FFFFF` (exhaustive);
  `bit ? ps + d : ps - d = ps + sign(d, ±1)` (one `vpsignd`; SAT); the
  history step on the packed slot,
  `h' << 8 = ((e << 1) & 0xFE00) | (e & 0x8000) | bit << 8` (SAT). ucas
  found no cheaper scalar form for the squash, APM, rate and mixing
  formulas. For the APM interpolation,
  `(a(128 - w) + bw) >> 7 = a + (((b - a) w) >> 7)` holds by algebra
  (`128a` is a multiple of 128 and `>>` floors), but saves nothing in the
  vector code.
- **Resets.** Line tags carry an epoch (`(hash & 0xFFFF) | epoch << 16`):
  a reset starts a new epoch, and a line of an older epoch is reset when
  it is selected, exactly as if the table had been zeroed. Only O0 and
  O1 are cleared. The arena comes from `mmap` (zero, and pages a unit
  never touches are never faulted in).
- **Tried, not kept:** resetting A2 rows lazily (per-row epoch): the
  check costs about 2.5% per bit and saved nothing measurable even with
  64 KiB blocks. Smaller hashed tables (lines_log2 18 down to 12) were
  only up to 8% faster and cost up to 8% in size: the misses mostly
  overlap.

What remains is mostly the dependency chain of one bit (slot gather,
stretch gather, mixer, squash, APM, coder) plus the 24 scalar write-backs
(no scatter in AVX2). `.awseek` archives also decode their blocks in
parallel (`taw d -j N`, `taw_seek_set_threads`): 5 blocks went from
5.2 s to 1.35 s on 5 threads.

## Encoder pipeline

1. Buffer the input. Fitting is offline, so the encoder sees all of it.
2. Table geometry from the coding-unit size `u` (n for aw, the block size
   for awseek): `lines_log2 = clamp(log2 u - 2, 8, 18)`,
   `hist_log2 = clamp(log2 u, 12, 24)`, `match_log2 = clamp(log2 u - 2, 10, 22)`.
3. One pass of the context models over the input, with the decoder's
   per-unit resets, keeping an evenly spread sample of bits (features,
   selector context, bit). The selection runs over bits in a per-byte
   rotated order (a hash of the byte position), so every bit position is
   sampled: a plain stride kept only bit 7 whenever 8n / samples was a
   multiple of 8, which cost 3-11% on power-of-two inputs.
4. Candidate shapes: hidden ∈ {0, 4, 8, 16} × selectors × {F16, Q8}, each
   kept only if `8 + payload <= n/32`. They are ranked by a prior from
   full sweeps on the bench corpus: the best models cost n/1100 to n/220
   bytes, and the top shapes are within ~0.5% of each other. So the prior
   favours models near n/500 bytes, with a hidden layer, Q8 weights and a
   bit-position selector. The target is capped at 128 KiB: on a 1.3 GB XCI
   the uncapped prior chose a 1.8 MB model with a 4096-set selector, whose
   weights fall out of cache, and decoding ran about 3x slower.
5. Fit the top candidates with AdamW (batch 256, lr 0.02 → 0.001
   linearly, β = 0.9/0.999, weight decay 1e-5, lazy updates of only the
   weight sets a batch touches). The first `max(1, epochs/5)` epochs train
   one shared set, which then seeds every set: without this warm-up the
   result depended on the seed by up to 2%. Fitting is plain scalar C and
   need not be reproducible.
6. Quantize (F16 round to nearest even; Q8 per-row absmax/127 scale),
   load the bytes through the decoder's own loader, and estimate
   `bits/bit × 8n + 8 × model bytes` on the samples with the real
   inference code. Keep the best.
7. Code the input with profile 1. For `.aw` with `predictor = auto`, also
   code it with profile 0 and keep the smaller archive. `.awseek` streams
   through `taw_seek_encode` instead. It codes each block once with the chosen
   model and stores the blocks that do not shrink. Collection, candidate
   fitting and block coding run on `threads` workers. The samples are picked
   by position, so the archive does not depend on the thread count.

Level sets the effort only:

| level | samples (bits) | candidates | epochs |
|-------|----------------|------------|--------|
| 1 | 256 Ki | 1 | 3 |
| 2 | 512 Ki | 1 | 4 |
| 3 | 512 Ki | 2 | 6 |
| 4 | 1 Mi | 2 | 6 |
| 5 | 1 Mi | 3 | 6 |
| 6 | 2 Mi | 3 | 8 |
| 7 | 2 Mi | 4 | 8 |
| 8 | 4 Mi | 6 | 10 |
| 9 | 4 Mi | 8 | 12 |

`TAW_FIT_DEBUG=1` in the environment prints every fitted candidate to
stderr; `TAW_FIT_DEBUG=a` fits all candidates (slow; that is how the
prior was calibrated).

## Weight formats

All formats are dequantized to binary32 once, when the model loads, so
they cost nothing per bit: they only trade model bytes for accuracy.
Mixer rows are short (26 or H + 1 weights), so every format keeps its
scales per row rather than ggml's 32- or 256-weight blocks, and the
"IQ" formats are plain tables (no E8 lattice or super-blocks):

- `Q4` is ggml's Q4_0 per row; `Q2` is affine (scale and offset per row,
  like Q2_K's min) because two symmetric bits have no zero.
- `FP8` is E4M3 with an F16 row scale. `NVFP4` is NVIDIA's layout
  (E2M1, an E4M3 scale per 16 weights) with an F16 row scale in place of
  the per-tensor FP32 one.
- `IQ4` uses ggml's IQ4_NL values. `IQ2` is the 4-level Lloyd-Max
  quantizer of a Gaussian. `IQ8` is `k (|k| + 128)`: steps of 129 near
  zero widen to 255 at the ends, which suits weights peaked at zero, and
  all 256 codes are valid (Q8 refuses -128).
- The quantizer (encoder only) searches the row scale on a grid around
  `amax / max level` and refines it by least squares on the assignment;
  `Q2` searches the clipped range and solves the 2x2 least squares;
  `NVFP4` tries a few row scales and, per block, the E4M3 codes around the
  ideal one. Every choice is scored with the decoder's own dequantized
  values.

On the bench corpus at level 1 (every candidate fitted, archive estimate
relative to the best), the 4-bit formats win most often: IQ4/Q4 on libc,
roff and C source, FP8 on text and the word list. F16 and IQ2 never win;
IQ2 loses 2-15% (post-training quantization to 2 bits is too coarse for
the hidden layer).

## Decoder

1. Validate the header and refuse unsupported `(profile, quant)` pairs.
2. Load and dequantize the model once (F16C on x86 when available), and
   pick the layer-1 kernel by CPUID: AVX-512F, AVX2, or scalar.
3. Decode chunks in order (aw). For awseek, decode only the blocks that
   overlap the requested range, after resetting the history at each block.

## Determinism

Fitting may be sloppy, but inference may not. Every probability must be
bit-identical between the encoder and every decoder. The rules:

- Each arithmetic profile fixes its operation order, accumulator widths,
  rounding and squash function.
- The build disables FP contraction and fast-math
  (`-ffp-contract=off -fno-fast-math`).
- Nothing links libm: it is slow, and its results differ between libcs.
  Squash and stretch use SIMD-intrinsic or table formulas defined in the
  spec. Every SIMD path (scalar, AVX2/F16C, AVX-512) must give the scalar
  reference's exact bits.
- The probability handed to the coder is a 16-bit integer, and the
  float-to-integer conversion is defined by the profile.

## Seeking

`.awseek` resets the predictor history at every block but shares the one
model. A seek costs reading the header, model and index, plus decoding
the overlapping blocks. Blocks are independent: the encoder codes them
on `threads` workers, and a reader may decode the blocks of one request
in parallel (`taw_seek_set_threads`, `taw_decompress_mt`, `taw d -j N`).
Each decoding thread holds its own tables (about 175 MB for 1 MiB
blocks), whole blocks are decoded straight into the caller's buffer and
the partial ones at the edges through a per-thread buffer. The source's
`read_at` must then be thread safe (`pread` is).

## Source layout

| path | role |
|------|------|
| `include/tadamw/tadamw.h` | public API (`cmake/config.h.in` gives `<tadamw/config.h>`) |
| `src/api.c` | public entry points, variant dispatch |
| `src/buf.c` | growable output buffer |
| `src/crc32c.c`, `src/crc32c_sse42.c` | CRC-32C: slicing-by-8, SSE4.2 |
| `src/pool.c` | worker threads (`taw_parallel`) |
| `src/rc.c` | binary range coder (format.md §4) |
| `src/predictor.c` | arithmetic profiles: dispatch, profile 0, profile 1 APM and per-bit loop |
| `src/p1.h` | profile 1 internal interface and constants |
| `src/p1_ctx.c` | profile 1 context models, squash/stretch tables |
| `src/p1_net.c` | profile 1 model parsing, dequantization, scalar mixer, ISA choice |
| `src/p1_avx2.c` | AVX2/F16C kernels: context-model inputs/update, hashes, mixer, F16 load |
| `src/p1_run_avx2.c` | the fused AVX2 per-bit loop (encode, decode, skip) |
| `src/p1_avx512.c` | AVX-512F mixer (used by `p1_net_eval`; the per-bit loop runs the fused AVX2 code) |
| `src/p1_simd_out.h` | mixer pieces shared by the AVX2 and AVX-512 files |
| `src/half.h` | binary16 conversions |
| `src/quant.c` | weight row formats: sizes, validated decoding, the encoder's quantizer |
| `src/fit.c` | encoder: feature pass, candidates, AdamW, quantization, choice |
| `src/header.c` | header (de)serialization and validation |
| `src/aw.c` | `.aw` stream encode/decode |
| `src/awseek.c` | `.awseek` encode/decode/random access |
| `tools/taw.c` | CLI |
| `tools/taw_nx.c` | XCI/NSP transcoder (docs/nx.md) |
| `tools/hactool/` | hactool as ScarletNX carries it, plus `snx_hac_encrypt` (tadamw) |
| `tests/` | CTest executables + CLI roundtrip script |

## Open items

- Decoder speed: about 1 MB/s per thread with AVX2 (see "Decoder
  performance"), and `.awseek` scales with threads across blocks. Ideas
  left: fewer context models where some never earn weight, AVX-512
  scatter for the write-back, shortening the slot → stretch → mixer
  chain. Evaluating several outcomes of the next bit was tried and is
  slower.
- Encoder speed: about 2.8 s/MB at level 1, 7.5 s/MB at level 5, and
  around 2 minutes/MB at level 9. That is two context passes
  (features, then coding) plus fitting. Fitting could run on a GPU
  (hybrid, see the decisions table); its results don't need to be
  reproducible.
- Small awseek blocks are expensive under profile 1: every block resets
  about 5 MB of tables (the APM alone is 4.3 MB). The encoder could size
  tables down for small blocks, but the APM is fixed by the spec.
- Optional GPU decoding path: opt-in only, and it must reproduce the CPU
  reference bit for bit (no contraction, e.g. SPIR-V `NoContraction`, and
  the same `dot4` order). Most of the work is integer context modelling,
  which is serial per coding unit, so GPU decoding mainly makes sense
  across many awseek blocks at once.
- AVX-512 mixer: compiled and dispatched by CPUID, but the development
  machine has no AVX-512, so it has never actually run. The parity tests
  cover it automatically on a machine that has it. Its context models use
  the AVX2 kernels.
- `.aw` streams are still encoded from memory; `.awseek` has the streaming
  encoder (`taw_seek_encode`), which fits on an even sample of the whole input.
- Big-endian: the code is written for it (byte-wise I/O, CRC table path
  byte-swaps), but it has not run on a big-endian machine yet.
- taw-nx: the XCI's update, normal and logo partitions stay encrypted
  (small); personalized tickets' title keys are not decrypted, so those NCAs
  stay encrypted. The encoder writes a temporary plain image the size of the
  dump next to the output; a source that decrypts on the fly would avoid it.
  Incompressible blocks (compressed game assets) pay a full coding pass before
  being stored; an entropy pre-check could skip them.