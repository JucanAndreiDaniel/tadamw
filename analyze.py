#!/usr/bin/env python3
"""Build RESULTS.md from results.tsv. Pure reporting; no new experiments."""
import csv
import statistics
from collections import Counter

CFG = ['base', 'adamw', 'muon', 'faro', 'adago', 'namo', 'namod',
       'malter', 'c3elr', 'hnet', 'gate']
LABEL = {
    'base': 'Shipped base', 'adamw': 'AdamW tuned', 'muon': 'Muon tuned',
    'faro': 'FARO', 'adago': 'AdaGO', 'namo': 'NAMO', 'namod': 'NAMOD',
    'malter': 'MALTER', 'c3elr': 'C3+ELR', 'hnet': 'H-Net (p14)',
    'gate': 'Entropy gate (p11)',
}
CORP = ['text1m', 'libc1m', 'code1m', 'json1m', 'html1m', 'log1m',
        'dna1m', 'rand1m', 'mixed1m', 'text4m', 'libc4m']
CORE = ['text4m', 'libc4m', 'code1m', 'json1m']
INF = [c for c in CORP if c != 'rand1m']


def load():
    rows = list(csv.DictReader(open('results.tsv'), delimiter='\t'))
    assert len(rows) == 278, len(rows)
    assert all(r['decode_ok'] == '1' and r['enc_rc'] == '0' and r['dec_rc'] == '0'
               for r in rows), 'verification failure in raw data'
    B = {(r['config'], r['corpus'], int(r['level'])): int(r['archive_bytes'])
         for r in rows}
    T = {(r['config'], r['corpus'], int(r['level'])): (float(r['enc_sec']),
                                                      float(r['dec_sec']))
         for r in rows}
    S = {(r['corpus']): int(r['src_bytes']) for r in rows}
    return B, T, S


def fmt(n):
    return f'{n:,}'


def delta(n):
    return f'{n:+,}'


def md_table(header, rows, align_first='l'):
    a = ['---:' if x.startswith(':') else '---' for x in header]
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(a) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def wtl(deltas):
    return (sum(x < 0 for x in deltas), sum(x == 0 for x in deltas),
            sum(x > 0 for x in deltas))


def statline(ds):
    return f'{statistics.mean(ds):+,.1f} ± {statistics.stdev(ds):,.1f}'


def main():
    B, T, S = load()
    o = []
    w = o.append

    w('# Definitive tadamw benchmark matrix — final numbers report')
    w('')
    w('Branch `exp/final-matrix` (fresh at `origin/main` `e64cc83`). Worktree:')
    w('`/home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix`.')
    w('')
    w('**278 encodes, 278 decodes, 278/278 byte-identical to source, 0 failures.** '
      'Every archive was decoded with the **same binary** that produced it and '
      'checked with `cmp`. Raw rows: [`results.tsv`](results.tsv). This is a '
      'measurement-only consolidation; no new algorithmic experiments were run.')
    w('')
    w('All encodes ran with **12 parallel workers on a 12-core machine**, '
      'including the 36 L9 encodes. Archive bytes are deterministic and '
      'load-independent, so they are the primary metric. **Encode/decode times '
      'were recorded under full CPU contention (`run_all.sh`, `-P 12`); they '
      'are noisy, load-contended observations and are NOT used for any speed '
      'claim.**')
    w('')

    w('## Setup')
    w('')
    w('### Corpora')
    w('')
    w('Eleven deterministic corpora. The brief names `text1m`/`libc1m` as 1 MiB '
      'files inside `corpora2/`; in the read-only bench tree those two actually '
      'live in the parent directory `/tmp/opencode/tadamw-bench/`, while the '
      'other nine live in `/tmp/opencode/tadamw-bench/corpora2/`. Both locations '
      'are READ-ONLY. All eleven were used at L1/L5. L9 uses the four core '
      'corpora `text4m`, `libc4m`, `code1m`, `json1m`.')
    w('')
    w('| corpus | location | source bytes |')
    w('|---|---|---:|')
    loc = {c: ('parent dir' if c in ('text1m', 'libc1m') else 'corpora2/') for c in CORP}
    for c in CORP:
        w(f'| `{c}` | {loc[c]} | {fmt(S[c])} |')
    w('')
    w('### Configurations (exact bytes, exact env, verified against each branch\'s `RESULTS.md`)')
    w('')
    w('The `hnet` binary was **not present** in its sibling worktree '
      '(`exp-hnet-feature` has no `build/` directory). It was rebuilt inside '
      'this worktree from that branch\'s exact source '
      '(`git archive exp/hnet-feature | tar -x -C hnet-build`, '
      '`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`), and '
      'validated against its published numbers: `text1m` L1 = 181,307 B and '
      '`gate` against 181,473 B, both reproduced exactly. All other binaries '
      'are the shipped read-only sibling-worktree binaries, used as-is. '
      'Every invocation scrubs inherited state with `env -i`, so no `TAW_*` '
      'variable leaks between configs. `-g off` (CPU fit) throughout.')
    w('')
    w('| id | binary (read-only unless noted) | profile flag | env override |')
    w('|---|---|---|---|')
    env = {
        'base': '*(none — shipped defaults)*',
        'adamw': '`TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0.001`',
        'muon': '`TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015`',
        'faro': 'muon four LR vars + `TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8`',
        'adago': '`TAW_ADAGO=adago TAW_ADAGO_LR=29 TAW_ADAGO_EPS=1e-3` + muon four LR vars',
        'namo': '`TAW_HYBRID=namo TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.006 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01`',
        'namod': '`TAW_HYBRID=namod TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.015 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01 TAW_NAMO_BETA=0.999 TAW_NAMO_CLAMP=0.9`',
        'malter': '`TAW_MALT=malter TAW_MALT_LR0=0.28 TAW_MALT_LR1=0.0056 TAW_MALT_B2=0.9 TAW_MALT_EPS=1e-8` + `TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015`',
        'c3elr': '`TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0.001 TAW_ELR=1 TAW_ELR_C=0.1 TAW_ELR_SCOPE=shape TAW_ELR_BETA=0.99 TAW_ELR_CLAMP=8 TAW_ELR_SCHED=linear TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2`',
        'hnet': '*(none — profile 14 default fitter)*',
        'gate': '*(none — profile 11 default fitter)*',
    }
    binary = {
        'base': '`wild-main/build/tools/taw`',
        'adamw': '`exp-lr-baseline/build/tools/taw`',
        'muon': '`exp-muon-adam/build/tools/taw`',
        'faro': '`exp-faro/build/tools/taw`',
        'adago': '`exp-adago/build/tools/taw`',
        'namo': '`exp-namo/build/tools/taw`',
        'namod': '`exp-namo/build/tools/taw`',
        'malter': '`exp-malt/build/tools/taw`',
        'c3elr': '`exp-combo-adamw/build/tools/taw`',
        'hnet': '**rebuilt here** from `exp/hnet-feature`',
        'gate': '`exp-entropygate/build/tools/taw`',
    }
    prof = {c: '`-p hnet`' if c == 'hnet' else '`-p gate`' if c == 'gate' else '`-p mix`' for c in CFG}
    for c in CFG:
        w(f'| {LABEL[c]} | {binary[c]} | {prof[c]} | {env[c]} |')
    w('')

    w('#### Timing observations — load-contended, NOT for speed claims')
    w('')
    w('Recorded because every encode was timed, but all 278 ran with `-P 12` '
      'on 12 cores. These enc-seconds are inflated by contention and include '
      'L9 cells; they are **not** comparable to prior single-worker timings.')
    w('')
    rows = []
    for c in CFG:
        ts = [T[(c, corp, lvl)][0] for corp in CORP for lvl in (1, 5)
              if (c, corp, lvl) in T]
        rows.append([LABEL[c], f'{min(ts):.1f}', f'{statistics.median(ts):.1f}',
                     f'{max(ts):.1f}'])
    w(md_table(['config', 'min enc s', 'median enc s', 'max enc s (L9)'], rows))
    w('')

    # ---- 1. baseline ----
    w('## 1. Shipped baseline — exact bytes (first-class result)')
    w('')
    w('Shipped, no-override binary. L9 was run for the baseline on the four '
      'core corpora only; `—` means out of scope. Every archive below decoded '
      'byte-identically to its source.')
    w('')
    rows = []
    for c in CORP:
        l9 = fmt(B[('base', c, 9)]) if ('base', c, 9) in B else '—'
        rows.append([f'`{c}`', fmt(S[c]), fmt(B[('base', c, 1)]),
                     fmt(B[('base', c, 5)]), l9])
    w(md_table(['corpus', 'source bytes', 'L1 bytes', 'L5 bytes', 'L9 bytes'], rows))
    w('')
    w('`rand1m` grows by 672 B (stored-table overhead); it is incompressible '
      'and is treated separately in §5.')
    w('')

    # ---- 2. L1/L5 matrix ----
    w('## 2. Full L1/L5 matrix — bytes and paired deltas')
    w('')
    w('Lower bytes is better. Δ columns are `config − reference` (negative = '
      'fewer bytes = better). `rand1m` is shown but excluded from all rankings '
      'and statistics (§5).')
    w('')
    for lvl in (1, 5):
        w(f'### Level {lvl} — archive bytes')
        w('')
        rows = [[f'`{c}`'] + [fmt(B[(cfg, c, lvl)]) for cfg in CFG] for c in CORP]
        w(md_table(['corpus'] + [LABEL[x] for x in CFG], rows))
        w('')
        for ref, refname in [('base', 'shipped base'), ('muon', 'Muon tuned')]:
            w(f'### Level {lvl} — Δ bytes vs {refname}')
            w('')
            rows = [[f'`{c}`'] + [delta(B[(cfg, c, lvl)] - B[(ref, c, lvl)])
                                  for cfg in CFG] for c in CORP]
            w(md_table(['corpus'] + [LABEL[x] for x in CFG], rows))
            w('')

    # ---- 3. L9 ----
    w('## 3. L9 table — four core corpora, configs 1–9')
    w('')
    w('L9 was run only for configs 1–9 (the fitter-based configs) on the four '
      'core corpora: 9 × 4 = 36 encodes. Profiles 11/14 are excluded here by '
      'scope; their `-p mix` fitters are the relevant comparison.')
    w('')
    l9cfg = CFG[:9]
    rows = [[f'`{c}`'] + [fmt(B[(cfg, c, 9)]) for cfg in l9cfg] for c in CORE]
    w(md_table(['corpus'] + [LABEL[x] for x in l9cfg], rows))
    w('')
    for ref, refname in [('base', 'shipped base'), ('muon', 'Muon tuned')]:
        w(f'### L9 — Δ bytes vs {refname}')
        w('')
        rows = [[f'`{c}`'] + [delta(B[(cfg, c, 9)] - B[(ref, c, 9)])
                              for cfg in l9cfg] for c in CORE]
        rows.append(['**sum Δ**'] + [delta(sum(B[(cfg, c, 9)] - B[(ref, c, 9)]
                                               for c in CORE)) for cfg in l9cfg])
        w(md_table(['corpus'] + [LABEL[x] for x in l9cfg], rows))
        w('')

    # ---- 4. W/T/L and paired deltas ----
    w('## 4. Per-config win / tie / loss and mean paired Δ (`rand1m` excluded)')
    w('')
    w('n = 10 informative corpora per level; Δ = config − reference; '
      'W = negative Δ, T = zero, L = positive Δ. Mean ± sample sd is the '
      'paired per-cell difference in bytes.')
    w('')
    scopes = [('Level 1', [(c, 1) for c in INF]),
              ('Level 5', [(c, 5) for c in INF]),
              ('L1+L5 combined', [(c, l) for c in INF for l in (1, 5)])]
    for name, cells in scopes:
        w(f'### {name} (n = {len(cells)} cells)')
        w('')
        for ref, refname in [('base', 'shipped base'), ('muon', 'Muon tuned')]:
            w(f'**vs {refname}**')
            w('')
            rows = []
            for cfg in CFG:
                ds = [B[(cfg, c, l)] - B[(ref, c, l)] for c, l in cells]
                wcount, tcount, lcount = wtl(ds)
                rows.append([LABEL[cfg], statline(ds),
                             f'{wcount} / {tcount} / {lcount}'])
            w(md_table(['config', 'mean Δ ± sd (B)', 'W / T / L'], rows))
            w('')

    # ---- 5. rand1m ----
    w('## 5. `rand1m` — reported separately')
    w('')
    g1 = {cfg: B[(cfg, 'rand1m', 1)] for cfg in CFG}
    g5 = {cfg: B[(cfg, 'rand1m', 5)] for cfg in CFG}
    assert g1 == g5, 'rand1m L1/L5 differ'
    core9 = sorted({g1[c] for c in CFG[:9]})
    w(f'Configs 1–9 produce exactly {", ".join(fmt(v) for v in core9)} bytes '
      f'at both L1 and L5 (source {fmt(S["rand1m"])} B, +672 B stored-table '
      f'overhead). The two profile-specific fitters differ only in that fixed '
      f'overhead: entropy gate = {fmt(g1["gate"])} B (+676) and H-Net = '
      f'{fmt(g1["hnet"])} B (+688), reflecting profile 11\'s 4-byte gate tail '
      f'and profile 14\'s wider mixer rows. Within each config the value is '
      f'identical at L1 and L5 and independent of the fit — every pairwise '
      f'delta among configs 1–9 is 0, and gate/hnet differ from those by a '
      f'constant header/model-size offset only. All 22 `rand1m` archives '
      f'passed decode + `cmp`. `rand1m` carries no information about fitter '
      f'quality and was excluded from every W/T/L count and mean Δ in §4 and '
      f'from the ship decision.')
    w('')

    # ---- 6. ship decision ----
    w('## 6. Which config to ship')
    w('')
    w('**Ship MALTER** (`exp-malt`, `TAW_MALT=malter` + Muon vec-LR vars). '
      'Rationale, using the deterministic byte metric:')
    w('')
    namod_sum = sum(B[('namod', c, l)] - B[('muon', c, l)] for c in INF for l in (1, 5))
    malter_sum = sum(B[('malter', c, l)] - B[('muon', c, l)] for c in INF for l in (1, 5))
    namo_sum = sum(B[('namo', c, l)] - B[('muon', c, l)] for c in INF for l in (1, 5))
    adago_sum = sum(B[('adago', c, l)] - B[('muon', c, l)] for c in INF for l in (1, 5))
    w(f'- **Best-in-class L1/L5 aggregate, within noise of the leader.** Over '
      f'the 20 informative L1/L5 cells MALTER is {fmt(malter_sum)} B below '
      f'tuned Muon ({statline([B[("malter", c, l)] - B[("muon", c, l)] for c in INF for l in (1, 5)])} B per cell); '
      f'NAMOD is {fmt(namod_sum)} B, NAMO {fmt(namo_sum)} B, AdaGO '
      f'{fmt(adago_sum)} B. The MALTER–NAMOD gap is only {fmt(abs(namod_sum - malter_sum))} B '
      f'across 20 cells (~{abs(namod_sum - malter_sum)//20} B/cell), far below '
      f'the single-seed fitter noise floor — so they are statistically tied '
      f'and the tie-break must come from other evidence.')
    w('- **Broad wins vs the previous champion, with real magnitude.** MALTER '
      'wins 14 of 20 informative L1/L5 cells vs Muon (2 ties, 4 losses) and 17 '
      'of 20 vs the shipped base. FARO has more *raw* wins vs Muon (16) but '
      'they average only a few bytes each; MALTER\'s 14 wins carry a '
      f'{fmt(abs(malter_sum))} B aggregate gain.')
    w('- **Only one of the four largest L1/L5 gains that also wins at L9.** '
      'On the four core corpora at L9, MALTER is 375 B below Muon (and 25,757 B '
      'below shipped base). Of the other three configs with the largest L1/L5 '
      'gains, NAMOD is 3,126 B *worse* than Muon at L9, NAMO 2,271 B worse and '
      'AdaGO 1,672 B worse. Shipping NAMOD would trade a negligible L1/L5 edge '
      'for a real L9 regression. (FARO is 1,423 B better than Muon at L9 but '
      'its L1/L5 aggregate edge is only 347 B — roughly 17 B/cell.)')
    w('- **No pathological regression.** MALTER\'s worst single L1/L5 cell vs '
      'Muon is +1,063 B (`libc4m` L1); its largest gain is −3,941 B (`libc4m` '
      'L5) and it ties Muon exactly on both `text4m` levels. Every other '
      'informative cell is within ±700 B.')
    w('')
    w('**Runner-up: NAMOD** if L1/L5 bytes are the only thing that matters and '
      'L9 is never used — it has the single lowest L1/L5 total and the '
      'smallest worst-case regression (+332 B), but it loses to Muon at L9. '
      '**FARO** is among the most *consistent* improvements over the shipped '
      'base (L5: 10/0/0, shared with NAMO and MALTER; best of all configs at '
      'L9) but its aggregate edge over Muon at L1/L5 is only −347 B across 20 '
      'cells — statistically indistinguishable. Profiles 11/14 (`gate`/'
      '`hnet`) are not drop-in champions: they use different fitters and are '
      'behind the Muon family on aggregate (§4), with a few individual-corpus '
      'exceptions (e.g. `gate` on `text4m` L1); keep them opt-in.')
    w('')

    # ---- 7. caveats ----
    w('## 7. Caveats')
    w('')
    w('- **Single seed.** Only `exp-combo-adamw` implements `TAW_SEED`; all '
      'arms here are single-seed. The fitter picks the best of several '
      'per-cell candidates, so differences below roughly 1 kB (and several '
      'larger outliers) are within candidate-selection/quantization noise. '
      'W/T/L counts quantify direction consistency, not fit-noise error bars.')
    w('- **Load-contended timings.** All 278 encodes ran with `-P 12` on 12 '
      'cores. `enc_sec`/`dec_sec` in `results.tsv` are recorded for completeness '
      'but are noisy and systematically inflated by contention; L9 times in '
      'particular (~180–400 s uncontended) are not comparable to prior '
      'single-worker reports. **No speed conclusion is drawn from them.**')
    w('- **Different fitters for profiles 11/14.** `gate` (`-p gate`, profile '
      '11) and `hnet` (`-p hnet`, profile 14) are not merely hyperparameter '
      'variants of the profile-1/Muon fitter: each uses its own default fitter '
      'and archive format (11 adds a 4-byte gate tail; 14 widens mixer rows to '
      '27 and falls back to the scalar mixer). Their numbers are valid byte '
      'measurements but are not apples-to-apples with the `TAW_*`-driven '
      'configs, and their archives require the matching profile-aware decoder.')
    w('- **Corpus coverage.** 11 corpora at L1/L5, but L9 only 4 core corpora '
      'and only configs 1–9. `rand1m` is incompressible and excluded from '
      'ranking; `libc4m`/`text4m` dominate the aggregate deltas, as in the '
      'prior cross-corpus study. Eight of ten informative corpora show '
      'champion-to-champion differences of at most a few hundred bytes.')
    w('- **hnet binary provenance.** Rebuilt locally from `exp/hnet-feature` '
      'because its sibling worktree had no `build/`; validated against that '
      'branch\'s published `text1m` L1 = 181,307 B. Its decode used the same '
      'rebuilt binary.')
    w('- **No mutation of shared state.** Sibling worktrees, both corpora '
      'directories and `origin/main` were used strictly read-only; nothing '
      'was pushed.')
    w('')

    w('## 8. Reproduction')
    w('')
    w('```sh')
    w('cd /home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix')
    w('# rebuild the one missing binary (hnet)')
    w('git archive exp/hnet-feature | tar -x -C hnet-build')
    w('cmake -S hnet-build -B hnet-build/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang')
    w('cmake --build hnet-build/build -j4')
    w('# full 278-cell matrix, 12 workers, raw rows -> results.tsv (load-contended timings)')
    w('bash run_all.sh')
    w('python3 analyze.py > RESULTS.md')
    w('```')
    w('')
    w('`bench_one.sh` encodes one cell, times encode and decode with '
      '`date +%s.%N`, extracts archive bytes with `stat -c%s`, decodes with '
      'the same binary, `cmp`s against the source, and writes one TSV row per '
      'cell (including both return codes and `decode_ok`). `run_all.sh` fans '
      'all 278 cells out with `xargs -P 12`.')
    w('')

    open('RESULTS.md', 'w').write('\n'.join(o) + '\n')
    print(f'wrote RESULTS.md ({len(o)} lines)')


if __name__ == '__main__':
    main()
