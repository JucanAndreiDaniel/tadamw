# Dynamic, content-aligned `.awseek` blocks (H-Net-style chunking)

Branch `exp/hnet-blocks`, base `origin/main` `e64cc83`. All new behavior is
opt-in; the default `.aw`/`.awseek` bytes are unchanged.

## Verdict

**Format mechanism: success. Size win at equal granularity: marginal and
corpus-dependent.** The flag/index extension is small, deterministic,
thread-count independent, and leaves random access, named-file extraction and
profile 1 arithmetic intact. At an equal 64 KiB nominal target, dynamic blocks
are 0.37 % smaller on `text1m` and 0.04 % *larger* on `libc1m` (L5). The large
gains (4–8 %) come from raising the target (fewer predictor resets), which is
a granularity trade-off rather than a boundary-placement win. On mixed media
the boundaries do align tightly with content changes. Use the extension when
structure-aligned resets or fewer index entries matter; do not expect a free
compression win on homogeneous data.

## Format additions (`docs/format.md` §7.2)

* Header flag bit 1 (`TAW_FLAG_DYN_BLOCKS`, value 2), awseek only. Old decoders
  reject it with `TAW_E_UNSUPPORTED` instead of misreading the index.
* With the flag, each index entry grows from 16 to 20 bytes:
  `u64 coded_offset, u32 coded_len, u32 raw_crc, u32 raw_len`. `block_count` is
  no longer `ceil(total/block_size)`; decoded block `i` is
  `[Σ raw_len[<i], Σ raw_len[≤i])`. The same prefix sums drive seeks.
* Decoder validation: `Σ raw_len == total_raw_len`, `1 <= raw_len <= 2^26`,
  `coded_len <= raw_len`, offsets in range. `coded_len == raw_len` still means
  stored. `index_crc` covers the widened entries; the directory (§7.1) follows
  them unchanged. Footer is unchanged.
* `block_size` stays a power of two and is the *nominal* target used for model
  fitting; `.aw` and profiles 0/1 are not touched.
* Public API: `taw_params.dyn_blocks`, `.chunk_target`, `.change_threshold`;
  `taw_info.has_dyn_blocks`. CLI: `taw c -D [-T N] [-C N]` and
  `taw-nx enc -D [-T N] [-C N]`; `taw i` prints `dynamic: yes|no`.

## Boundary algorithm

Two sequential passes over the source (constant memory, 64 KiB scratch for
non-memory sources), so the result does not depend on `threads`:

1. **Change score.** For byte `x` with order-1 predecessor `c`, a saturating
   `uint16_t` histogram gives
   `est = ((count[c][x]+1)<<16)/(total[c]+256); s = 256 - (est>>8)`.
   The deviation `|s - EWMA(s)|` (1/256 step) is zeroed below 8 and is the
   *change* signal. Uniform noise has high `s` but low deviation; only real
   context shifts produce large values.
2. **Placement.** With `target = chunk_target ?: block_size`,
   `min_len = target/4`, `max_len = min(4*target, 2^26)`. In ratio mode
   (default) pass 1 totals the change signal and sets
   `share = total_change / ceil(total/target)`; pass 2 cuts when accumulated
   change reaches `share`, between the length bounds. A nonzero
   `change_threshold` instead cuts at a single byte with `s >= threshold`
   (a spike detector; diagnostic only — see caveats).

Example (`mixed1m` = 256 KiB text, 256 KiB random, 256 KiB zeros, 256 KiB
text; `-B 65536 -T 65536`): boundaries land at raw offsets 261575, 523719 and
785863 — within 600 bytes of all three true transitions — and the zero run
becomes two maximum-size (262144 B) blocks, while text uses 29–49 KiB blocks.

## Benchmarks

Corpora: `/tmp/opencode/tadamw-bench/{text1m,libc1m}` (1 048 576 B each,
read-only). Command shape:
`taw c -s -l L -p mix -B 65536 [-D -T T] corpus arc.awseek`.
`-p mix` keeps the model identical across rows (fitting uses `block_size`).
Decode times are best of 3 serial runs; enc times are single runs on a shared
machine and are noisy. **Bytes are primary.**

| corpus | L | mode / target | bytes | ratio | blocks | raw min–max | index B | enc s | dec s |
|---|---|---|---|---|---|---|---|---|---|
| text1m | 1 | fixed 64K | 229962 | 0.2193 | 16 | 65536–65536 | 272 | 3.13 | 1.514 |
| text1m | 1 | dyn 32K | 244520 | 0.2332 | 32 | 17880–63469 | 656 | 2.80 | 1.350 |
| text1m | 1 | **dyn 64K** | **229102** | **0.2185** | 16 | 43729–94385 | 336 | 2.71 | 1.331 |
| text1m | 1 | dyn 128K | 218486 | 0.2084 | 8 | 96176–180582 | 176 | 2.73 | 1.378 |
| text1m | 1 | dyn 256K | 211275 | 0.2015 | 4 | 204413–339462 | 96 | 2.92 | 1.231 |
| text1m | 5 | fixed 64K | 228902 | 0.2183 | 16 | 65536–65536 | 272 | 10.17 | 1.271 |
| text1m | 5 | dyn 32K | 243524 | 0.2322 | 32 | 17880–63469 | 656 | 9.48 | 1.014 |
| text1m | 5 | **dyn 64K** | **228040** | **0.2175** | 16 | 43729–94385 | 336 | 10.01 | 1.209 |
| text1m | 5 | dyn 128K | 217411 | 0.2073 | 8 | 96176–180582 | 176 | 10.10 | 1.204 |
| text1m | 5 | dyn 256K | 210198 | 0.2005 | 4 | 204413–339462 | 96 | 9.81 | 1.171 |
| libc1m | 1 | fixed 64K | 422664 | 0.4031 | 16 | 65536–65536 | 272 | 2.42 | 1.256 |
| libc1m | 1 | dyn 32K | 436764 | 0.4166 | 32 | 19728–61515 | 656 | 2.45 | 1.309 |
| libc1m | 1 | **dyn 64K** | **422868** | **0.4033** | 16 | 39689–106404 | 336 | 2.70 | 1.239 |
| libc1m | 1 | dyn 128K | 411065 | 0.3920 | 8 | 98485–187942 | 176 | 2.51 | 1.237 |
| libc1m | 1 | dyn 256K | 405797 | 0.3870 | 4 | 241278–286427 | 96 | 2.43 | 1.390 |
| libc1m | 5 | fixed 64K | 419752 | 0.4003 | 16 | 65536–65536 | 272 | 10.91 | 1.480 |
| libc1m | 5 | dyn 32K | 433511 | 0.4135 | 32 | 19728–61515 | 656 | 10.24 | 1.252 |
| libc1m | 5 | **dyn 64K** | **419926** | **0.4004** | 16 | 39689–106404 | 336 | 12.48 | 1.551 |
| libc1m | 5 | dyn 128K | 408465 | 0.3896 | 8 | 98485–187942 | 176 | 11.77 | 1.288 |
| libc1m | 5 | dyn 256K | 403408 | 0.3847 | 4 | 241278–286427 | 96 | 11.03 | 1.594 |

Equal-target deltas (dynamic 64K vs fixed 64K): `text1m` −860 B at L1 and
−862 B at L5; `libc1m` +204 B and +174 B. The 20-byte entries add 4 B per
block (+64 B for 16 blocks) but the variable lengths are what change the coded
payload. Raising the target to 256K saves 8.2 % (`text1m` L5) / 3.9 %
(`libc1m` L5) and shrinks the index to 96 B, at the cost of 4 seek units
instead of 16. Lowering the target to 32K costs 6–7 % because of extra resets
and index entries.

`-C` spike mode (200…256, `-T 65536`) collapsed to 64 `min_len` blocks and
produced 266.5 KiB (`text1m`) / 460.2 KiB (`libc1m`) at L1 — worse than fixed.
Ratio mode is the useful configuration.

## Correctness

* **Default is byte-identical:** golden CRC32C/size of four deterministic
  `.awseek` archives (zero/random/text/skewed, 100 kB, order-0, 4096 B blocks)
  are pinned in `tests/test_awseek.c` and match `origin/main` exactly:
  `914/2f4c13cc`, `100464/3f6480cc`, `54968/525c32a5`, `26977/a35c2865`.
* **Random access:** 5000-reading stress helper (reduced to 300 after a
  timeout on the shared host) at 1/3/8 threads on dynamic L5 archives —
  0 mismatches; in-test 120 mixed serial/parallel reads on profile 1; range,
  bounds, corruption-isolation and index-CRC checks.
* **CLI:** `taw i` reports the flag; `taw x` range and named-file extraction
  work on dynamic archives produced by `taw-nx enc -D` (with a directory),
  and `taw-nx dec` roundtrips them.
* **Suite:** `ctest` 12/12 (gpu skipped by design). ASan/UBSan build of
  `test_awseek` is clean.

## Caveats / follow-ups

* The encoder reads the input twice for boundary selection (no extra decoded
  data is stored); for a non-memory source that is two sequential scans.
* The change score's constants (EWMA 1/256, deviation floor 8, `target/4` …
  `4*target`) are heuristic. They were chosen by the sweeps above, not derived
  from the H-Net ratio loss; a real representation model would be stronger but
  much costlier.
* The large-target gain is mostly "fewer predictor resets", available to fixed
  blocks too. A fair comparison at equal block count stays near parity.
* A single huge homogeneous region is capped at `max_len`; very small targets
  can create many index entries and more resets.
* Flag bit 1 is a format change: archives produced with `-D` need this build
  (or a decoder that understands §7.2).
