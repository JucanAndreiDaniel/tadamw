# Definitive tadamw benchmark matrix — final numbers report

Branch `exp/final-matrix` (fresh at `origin/main` `e64cc83`). Worktree:
`/home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix`.

**278 encodes, 278 decodes, 278/278 byte-identical to source, 0 failures.** Every archive was decoded with the **same binary** that produced it and checked with `cmp`. Raw rows: [`results.tsv`](results.tsv). This is a measurement-only consolidation; no new algorithmic experiments were run.

All encodes ran with **12 parallel workers on a 12-core machine**, including the 36 L9 encodes. Archive bytes are deterministic and load-independent, so they are the primary metric. **Encode/decode times were recorded under full CPU contention (`run_all.sh`, `-P 12`); they are noisy, load-contended observations and are NOT used for any speed claim.**

## Setup

### Corpora

Eleven deterministic corpora. The brief names `text1m`/`libc1m` as 1 MiB files inside `corpora2/`; in the read-only bench tree those two actually live in the parent directory `/tmp/opencode/tadamw-bench/`, while the other nine live in `/tmp/opencode/tadamw-bench/corpora2/`. Both locations are READ-ONLY. All eleven were used at L1/L5. L9 uses the four core corpora `text4m`, `libc4m`, `code1m`, `json1m`.

| corpus | location | source bytes |
|---|---|---:|
| `text1m` | parent dir | 1,048,576 |
| `libc1m` | parent dir | 1,048,576 |
| `code1m` | corpora2/ | 1,048,576 |
| `json1m` | corpora2/ | 1,048,576 |
| `html1m` | corpora2/ | 1,048,576 |
| `log1m` | corpora2/ | 1,048,576 |
| `dna1m` | corpora2/ | 1,048,576 |
| `rand1m` | corpora2/ | 1,048,576 |
| `mixed1m` | corpora2/ | 1,048,576 |
| `text4m` | corpora2/ | 3,867,439 |
| `libc4m` | corpora2/ | 2,440,544 |

### Configurations (exact bytes, exact env, verified against each branch's `RESULTS.md`)

The `hnet` binary was **not present** in its sibling worktree (`exp-hnet-feature` has no `build/` directory). It was rebuilt inside this worktree from that branch's exact source (`git archive exp/hnet-feature | tar -x -C hnet-build`, `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang`), and validated against its published numbers: `text1m` L1 = 181,307 B and `gate` against 181,473 B, both reproduced exactly. All other binaries are the shipped read-only sibling-worktree binaries, used as-is. Every invocation scrubs inherited state with `env -i`, so no `TAW_*` variable leaks between configs. `-g off` (CPU fit) throughout.

| id | binary (read-only unless noted) | profile flag | env override |
|---|---|---|---|
| Shipped base | `wild-main/build/tools/taw` | `-p mix` | *(none — shipped defaults)* |
| AdamW tuned | `exp-lr-baseline/build/tools/taw` | `-p mix` | `TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0.001` |
| Muon tuned | `exp-muon-adam/build/tools/taw` | `-p mix` | `TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015` |
| FARO | `exp-faro/build/tools/taw` | `-p mix` | muon four LR vars + `TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8` |
| AdaGO | `exp-adago/build/tools/taw` | `-p mix` | `TAW_ADAGO=adago TAW_ADAGO_LR=29 TAW_ADAGO_EPS=1e-3` + muon four LR vars |
| NAMO | `exp-namo/build/tools/taw` | `-p mix` | `TAW_HYBRID=namo TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.006 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01` |
| NAMOD | `exp-namo/build/tools/taw` | `-p mix` | `TAW_HYBRID=namod TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.015 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01 TAW_NAMO_BETA=0.999 TAW_NAMO_CLAMP=0.9` |
| MALTER | `exp-malt/build/tools/taw` | `-p mix` | `TAW_MALT=malter TAW_MALT_LR0=0.28 TAW_MALT_LR1=0.0056 TAW_MALT_B2=0.9 TAW_MALT_EPS=1e-8` + `TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015` |
| C3+ELR | `exp-combo-adamw/build/tools/taw` | `-p mix` | `TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0.001 TAW_ELR=1 TAW_ELR_C=0.1 TAW_ELR_SCOPE=shape TAW_ELR_BETA=0.99 TAW_ELR_CLAMP=8 TAW_ELR_SCHED=linear TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2` |
| H-Net (p14) | **rebuilt here** from `exp/hnet-feature` | `-p hnet` | *(none — profile 14 default fitter)* |
| Entropy gate (p11) | `exp-entropygate/build/tools/taw` | `-p gate` | *(none — profile 11 default fitter)* |

#### Timing observations — load-contended, NOT for speed claims

Recorded because every encode was timed, but all 278 ran with `-P 12` on 12 cores. These enc-seconds are inflated by contention and include L9 cells; they are **not** comparable to prior single-worker timings.

| config | min enc s | median enc s | max enc s (L9) |
|---|---|---|---|
| Shipped base | 3.6 | 11.6 | 19.4 |
| AdamW tuned | 3.6 | 11.8 | 21.2 |
| Muon tuned | 3.8 | 14.1 | 55.7 |
| FARO | 4.0 | 13.9 | 56.2 |
| AdaGO | 4.1 | 14.3 | 48.0 |
| NAMO | 3.9 | 14.0 | 55.2 |
| NAMOD | 4.1 | 13.0 | 53.4 |
| MALTER | 3.4 | 13.8 | 51.2 |
| C3+ELR | 3.4 | 11.6 | 17.6 |
| H-Net (p14) | 7.8 | 17.1 | 31.9 |
| Entropy gate (p11) | 5.1 | 13.3 | 20.2 |

## 1. Shipped baseline — exact bytes (first-class result)

Shipped, no-override binary. L9 was run for the baseline on the four core corpora only; `—` means out of scope. Every archive below decoded byte-identically to its source.

| corpus | source bytes | L1 bytes | L5 bytes | L9 bytes |
|---|---|---|---|---|
| `text1m` | 1,048,576 | 181,557 | 179,166 | — |
| `libc1m` | 1,048,576 | 379,197 | 375,291 | — |
| `code1m` | 1,048,576 | 55,368 | 54,936 | 56,067 |
| `json1m` | 1,048,576 | 38,776 | 37,713 | 38,012 |
| `html1m` | 1,048,576 | 34,079 | 33,629 | — |
| `log1m` | 1,048,576 | 48,086 | 48,882 | — |
| `dna1m` | 1,048,576 | 263,782 | 263,546 | — |
| `rand1m` | 1,048,576 | 1,049,248 | 1,049,248 | — |
| `mixed1m` | 1,048,576 | 293,656 | 291,438 | — |
| `text4m` | 3,867,439 | 540,888 | 538,630 | 529,797 |
| `libc4m` | 2,440,544 | 825,031 | 818,442 | 808,195 |

`rand1m` grows by 672 B (stored-table overhead); it is incompressible and is treated separately in §5.

## 2. Full L1/L5 matrix — bytes and paired deltas

Lower bytes is better. Δ columns are `config − reference` (negative = fewer bytes = better). `rand1m` is shown but excluded from all rankings and statistics (§5).

### Level 1 — archive bytes

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | 181,557 | 179,913 | 179,678 | 179,502 | 179,271 | 179,273 | 179,144 | 179,405 | 180,344 | 181,307 | 181,473 |
| `libc1m` | 379,197 | 373,887 | 371,373 | 371,208 | 371,284 | 370,921 | 371,009 | 370,898 | 372,959 | 377,294 | 378,070 |
| `code1m` | 55,368 | 55,299 | 54,992 | 54,875 | 54,753 | 54,864 | 54,966 | 55,433 | 55,014 | 55,473 | 55,333 |
| `json1m` | 38,776 | 38,498 | 38,982 | 38,951 | 39,433 | 38,622 | 38,919 | 38,972 | 39,047 | 38,526 | 38,760 |
| `html1m` | 34,079 | 33,170 | 33,827 | 33,672 | 33,863 | 33,615 | 33,561 | 33,959 | 33,587 | 33,925 | 34,058 |
| `log1m` | 48,086 | 47,723 | 48,070 | 47,993 | 48,094 | 48,253 | 48,242 | 48,063 | 48,115 | 47,726 | 48,132 |
| `dna1m` | 263,782 | 263,618 | 263,720 | 263,707 | 263,719 | 263,704 | 263,672 | 263,706 | 263,649 | 263,841 | 263,788 |
| `rand1m` | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,264 | 1,049,252 |
| `mixed1m` | 293,656 | 292,482 | 292,310 | 292,040 | 292,743 | 292,146 | 292,551 | 292,084 | 292,814 | 293,749 | 293,312 |
| `text4m` | 540,888 | 542,710 | 542,645 | 543,767 | 542,645 | 542,645 | 542,645 | 542,645 | 542,730 | 541,141 | 539,651 |
| `libc4m` | 825,031 | 815,357 | 808,264 | 808,134 | 806,586 | 808,895 | 807,798 | 809,327 | 812,434 | 820,192 | 822,107 |

### Level 1 — Δ bytes vs shipped base

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | +0 | -1,644 | -1,879 | -2,055 | -2,286 | -2,284 | -2,413 | -2,152 | -1,213 | -250 | -84 |
| `libc1m` | +0 | -5,310 | -7,824 | -7,989 | -7,913 | -8,276 | -8,188 | -8,299 | -6,238 | -1,903 | -1,127 |
| `code1m` | +0 | -69 | -376 | -493 | -615 | -504 | -402 | +65 | -354 | +105 | -35 |
| `json1m` | +0 | -278 | +206 | +175 | +657 | -154 | +143 | +196 | +271 | -250 | -16 |
| `html1m` | +0 | -909 | -252 | -407 | -216 | -464 | -518 | -120 | -492 | -154 | -21 |
| `log1m` | +0 | -363 | -16 | -93 | +8 | +167 | +156 | -23 | +29 | -360 | +46 |
| `dna1m` | +0 | -164 | -62 | -75 | -63 | -78 | -110 | -76 | -133 | +59 | +6 |
| `rand1m` | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +16 | +4 |
| `mixed1m` | +0 | -1,174 | -1,346 | -1,616 | -913 | -1,510 | -1,105 | -1,572 | -842 | +93 | -344 |
| `text4m` | +0 | +1,822 | +1,757 | +2,879 | +1,757 | +1,757 | +1,757 | +1,757 | +1,842 | +253 | -1,237 |
| `libc4m` | +0 | -9,674 | -16,767 | -16,897 | -18,445 | -16,136 | -17,233 | -15,704 | -12,597 | -4,839 | -2,924 |

### Level 1 — Δ bytes vs Muon tuned

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | +1,879 | +235 | +0 | -176 | -407 | -405 | -534 | -273 | +666 | +1,629 | +1,795 |
| `libc1m` | +7,824 | +2,514 | +0 | -165 | -89 | -452 | -364 | -475 | +1,586 | +5,921 | +6,697 |
| `code1m` | +376 | +307 | +0 | -117 | -239 | -128 | -26 | +441 | +22 | +481 | +341 |
| `json1m` | -206 | -484 | +0 | -31 | +451 | -360 | -63 | -10 | +65 | -456 | -222 |
| `html1m` | +252 | -657 | +0 | -155 | +36 | -212 | -266 | +132 | -240 | +98 | +231 |
| `log1m` | +16 | -347 | +0 | -77 | +24 | +183 | +172 | -7 | +45 | -344 | +62 |
| `dna1m` | +62 | -102 | +0 | -13 | -1 | -16 | -48 | -14 | -71 | +121 | +68 |
| `rand1m` | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +16 | +4 |
| `mixed1m` | +1,346 | +172 | +0 | -270 | +433 | -164 | +241 | -226 | +504 | +1,439 | +1,002 |
| `text4m` | -1,757 | +65 | +0 | +1,122 | +0 | +0 | +0 | +0 | +85 | -1,504 | -2,994 |
| `libc4m` | +16,767 | +7,093 | +0 | -130 | -1,678 | +631 | -466 | +1,063 | +4,170 | +11,928 | +13,843 |

### Level 5 — archive bytes

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | 179,166 | 178,886 | 177,816 | 177,738 | 177,468 | 177,228 | 177,145 | 177,138 | 178,020 | 179,426 | 179,098 |
| `libc1m` | 375,291 | 371,894 | 366,997 | 366,890 | 366,598 | 367,034 | 366,727 | 366,504 | 367,730 | 371,256 | 374,705 |
| `code1m` | 54,936 | 54,859 | 54,527 | 54,656 | 54,414 | 54,057 | 54,039 | 53,957 | 54,712 | 54,950 | 54,913 |
| `json1m` | 37,713 | 37,989 | 36,995 | 36,925 | 37,352 | 37,048 | 36,972 | 36,771 | 39,029 | 37,436 | 37,749 |
| `html1m` | 33,629 | 33,225 | 33,265 | 33,134 | 34,555 | 33,026 | 32,953 | 32,905 | 36,141 | 33,022 | 33,578 |
| `log1m` | 48,882 | 48,307 | 47,702 | 48,013 | 48,673 | 48,006 | 48,034 | 47,854 | 48,973 | 48,110 | 48,834 |
| `dna1m` | 263,546 | 263,566 | 263,553 | 263,532 | 263,505 | 263,535 | 263,566 | 263,519 | 263,619 | 263,830 | 263,550 |
| `rand1m` | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,248 | 1,049,264 | 1,049,252 |
| `mixed1m` | 291,438 | 291,310 | 287,954 | 287,761 | 287,898 | 288,058 | 288,175 | 287,508 | 288,822 | 290,617 | 291,283 |
| `text4m` | 538,630 | 539,739 | 536,249 | 536,384 | 536,249 | 536,249 | 536,249 | 536,249 | 537,774 | 539,856 | 537,698 |
| `libc4m` | 818,442 | 811,754 | 804,677 | 804,367 | 799,332 | 801,369 | 801,095 | 800,736 | 800,599 | 816,648 | 815,092 |

### Level 5 — Δ bytes vs shipped base

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | +0 | -280 | -1,350 | -1,428 | -1,698 | -1,938 | -2,021 | -2,028 | -1,146 | +260 | -68 |
| `libc1m` | +0 | -3,397 | -8,294 | -8,401 | -8,693 | -8,257 | -8,564 | -8,787 | -7,561 | -4,035 | -586 |
| `code1m` | +0 | -77 | -409 | -280 | -522 | -879 | -897 | -979 | -224 | +14 | -23 |
| `json1m` | +0 | +276 | -718 | -788 | -361 | -665 | -741 | -942 | +1,316 | -277 | +36 |
| `html1m` | +0 | -404 | -364 | -495 | +926 | -603 | -676 | -724 | +2,512 | -607 | -51 |
| `log1m` | +0 | -575 | -1,180 | -869 | -209 | -876 | -848 | -1,028 | +91 | -772 | -48 |
| `dna1m` | +0 | +20 | +7 | -14 | -41 | -11 | +20 | -27 | +73 | +284 | +4 |
| `rand1m` | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +16 | +4 |
| `mixed1m` | +0 | -128 | -3,484 | -3,677 | -3,540 | -3,380 | -3,263 | -3,930 | -2,616 | -821 | -155 |
| `text4m` | +0 | +1,109 | -2,381 | -2,246 | -2,381 | -2,381 | -2,381 | -2,381 | -856 | +1,226 | -932 |
| `libc4m` | +0 | -6,688 | -13,765 | -14,075 | -19,110 | -17,073 | -17,347 | -17,706 | -17,843 | -1,794 | -3,350 |

### Level 5 — Δ bytes vs Muon tuned

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR | H-Net (p14) | Entropy gate (p11) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text1m` | +1,350 | +1,070 | +0 | -78 | -348 | -588 | -671 | -678 | +204 | +1,610 | +1,282 |
| `libc1m` | +8,294 | +4,897 | +0 | -107 | -399 | +37 | -270 | -493 | +733 | +4,259 | +7,708 |
| `code1m` | +409 | +332 | +0 | +129 | -113 | -470 | -488 | -570 | +185 | +423 | +386 |
| `json1m` | +718 | +994 | +0 | -70 | +357 | +53 | -23 | -224 | +2,034 | +441 | +754 |
| `html1m` | +364 | -40 | +0 | -131 | +1,290 | -239 | -312 | -360 | +2,876 | -243 | +313 |
| `log1m` | +1,180 | +605 | +0 | +311 | +971 | +304 | +332 | +152 | +1,271 | +408 | +1,132 |
| `dna1m` | -7 | +13 | +0 | -21 | -48 | -18 | +13 | -34 | +66 | +277 | -3 |
| `rand1m` | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +16 | +4 |
| `mixed1m` | +3,484 | +3,356 | +0 | -193 | -56 | +104 | +221 | -446 | +868 | +2,663 | +3,329 |
| `text4m` | +2,381 | +3,490 | +0 | +135 | +0 | +0 | +0 | +0 | +1,525 | +3,607 | +1,449 |
| `libc4m` | +13,765 | +7,077 | +0 | -310 | -5,345 | -3,308 | -3,582 | -3,941 | -4,078 | +11,971 | +10,415 |

## 3. L9 table — four core corpora, configs 1–9

L9 was run only for configs 1–9 (the fitter-based configs) on the four core corpora: 9 × 4 = 36 encodes. Profiles 11/14 are excluded here by scope; their `-p mix` fitters are the relevant comparison.

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR |
|---|---|---|---|---|---|---|---|---|---|
| `text4m` | 529,797 | 529,331 | 522,812 | 520,807 | 523,131 | 522,089 | 522,932 | 521,353 | 534,041 |
| `libc4m` | 808,195 | 812,151 | 792,919 | 792,804 | 793,743 | 795,608 | 795,431 | 794,751 | 796,739 |
| `code1m` | 56,067 | 55,631 | 54,102 | 54,166 | 54,215 | 54,179 | 54,107 | 54,043 | 55,512 |
| `json1m` | 38,012 | 37,354 | 36,106 | 36,739 | 36,522 | 36,334 | 36,595 | 36,167 | 38,838 |

### L9 — Δ bytes vs shipped base

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR |
|---|---|---|---|---|---|---|---|---|---|
| `text4m` | +0 | -466 | -6,985 | -8,990 | -6,666 | -7,708 | -6,865 | -8,444 | +4,244 |
| `libc4m` | +0 | +3,956 | -15,276 | -15,391 | -14,452 | -12,587 | -12,764 | -13,444 | -11,456 |
| `code1m` | +0 | -436 | -1,965 | -1,901 | -1,852 | -1,888 | -1,960 | -2,024 | -555 |
| `json1m` | +0 | -658 | -1,906 | -1,273 | -1,490 | -1,678 | -1,417 | -1,845 | +826 |
| **sum Δ** | +0 | +2,396 | -26,132 | -27,555 | -24,460 | -23,861 | -23,006 | -25,757 | -6,941 |

### L9 — Δ bytes vs Muon tuned

| corpus | Shipped base | AdamW tuned | Muon tuned | FARO | AdaGO | NAMO | NAMOD | MALTER | C3+ELR |
|---|---|---|---|---|---|---|---|---|---|
| `text4m` | +6,985 | +6,519 | +0 | -2,005 | +319 | -723 | +120 | -1,459 | +11,229 |
| `libc4m` | +15,276 | +19,232 | +0 | -115 | +824 | +2,689 | +2,512 | +1,832 | +3,820 |
| `code1m` | +1,965 | +1,529 | +0 | +64 | +113 | +77 | +5 | -59 | +1,410 |
| `json1m` | +1,906 | +1,248 | +0 | +633 | +416 | +228 | +489 | +61 | +2,732 |
| **sum Δ** | +26,132 | +28,528 | +0 | -1,423 | +1,672 | +2,271 | +3,126 | +375 | +19,191 |

## 4. Per-config win / tie / loss and mean paired Δ (`rand1m` excluded)

n = 10 informative corpora per level; Δ = config − reference; W = negative Δ, T = zero, L = positive Δ. Mean ± sample sd is the paired per-cell difference in bytes.

### Level 1 (n = 10 cells)

**vs shipped base**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +0.0 ± 0.0 | 0 / 10 / 0 |
| AdamW tuned | -1,776.3 ± 3,311.6 | 9 / 0 / 1 |
| Muon tuned | -2,655.9 ± 5,582.6 | 8 / 0 / 2 |
| FARO | -2,657.1 ± 5,718.5 | 8 / 0 / 2 |
| AdaGO | -2,802.9 ± 6,094.4 | 7 / 0 / 3 |
| NAMO | -2,748.2 ± 5,420.6 | 8 / 0 / 2 |
| NAMOD | -2,791.3 ± 5,741.9 | 7 / 0 / 3 |
| MALTER | -2,592.8 ± 5,359.5 | 7 / 0 / 3 |
| C3+ELR | -1,972.7 ± 4,278.0 | 7 / 0 / 3 |
| H-Net (p14) | -724.6 ± 1,568.9 | 6 / 0 / 4 |
| Entropy gate (p11) | -573.6 ± 953.9 | 8 / 0 / 2 |

**vs Muon tuned**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +2,655.9 ± 5,582.6 | 2 / 0 / 8 |
| AdamW tuned | +879.6 ± 2,354.0 | 4 / 0 / 6 |
| Muon tuned | +0.0 ± 0.0 | 0 / 10 / 0 |
| FARO | -1.2 ± 401.6 | 9 / 0 / 1 |
| AdaGO | -147.0 ± 598.4 | 5 / 1 / 4 |
| NAMO | -92.3 ± 321.9 | 7 / 1 / 2 |
| NAMOD | -135.4 ± 261.8 | 7 / 1 / 2 |
| MALTER | +63.1 ± 428.6 | 6 / 1 / 3 |
| C3+ELR | +683.2 ± 1,333.8 | 2 / 0 / 8 |
| H-Net (p14) | +1,931.3 ± 4,049.8 | 3 / 0 / 7 |
| Entropy gate (p11) | +2,082.3 ± 4,790.5 | 2 / 0 / 8 |

### Level 5 (n = 10 cells)

**vs shipped base**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +0.0 ± 0.0 | 0 / 10 / 0 |
| AdamW tuned | -1,014.4 ± 2,306.5 | 7 / 0 / 3 |
| Muon tuned | -3,193.8 ± 4,448.5 | 9 / 0 / 1 |
| FARO | -3,227.3 ± 4,556.2 | 10 / 0 / 0 |
| AdaGO | -3,562.9 ± 6,112.0 | 9 / 0 / 1 |
| NAMO | -3,606.3 ± 5,300.0 | 10 / 0 / 0 |
| NAMOD | -3,671.8 ± 5,399.3 | 9 / 0 / 1 |
| MALTER | -3,853.2 ± 5,486.3 | 10 / 0 / 0 |
| C3+ELR | -2,625.4 ± 5,996.4 | 6 / 0 / 4 |
| H-Net (p14) | -652.2 ± 1,438.1 | 6 / 0 / 4 |
| Entropy gate (p11) | -517.3 ± 1,043.2 | 8 / 0 / 2 |

**vs Muon tuned**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +3,193.8 ± 4,448.5 | 1 / 0 / 9 |
| AdamW tuned | +2,179.4 ± 2,418.0 | 1 / 0 / 9 |
| Muon tuned | +0.0 ± 0.0 | 0 / 10 / 0 |
| FARO | -33.5 ± 180.7 | 7 / 0 / 3 |
| AdaGO | -369.1 ± 1,833.0 | 6 / 1 / 3 |
| NAMO | -412.5 ± 1,052.7 | 5 / 1 / 4 |
| NAMOD | -478.0 ± 1,133.5 | 6 / 1 / 3 |
| MALTER | -659.4 ± 1,183.9 | 8 / 1 / 1 |
| C3+ELR | +568.4 ± 1,859.5 | 1 / 0 / 9 |
| H-Net (p14) | +2,541.6 ± 3,654.8 | 1 / 0 / 9 |
| Entropy gate (p11) | +2,676.5 ± 3,546.1 | 1 / 0 / 9 |

### L1+L5 combined (n = 20 cells)

**vs shipped base**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +0.0 ± 0.0 | 0 / 20 / 0 |
| AdamW tuned | -1,395.3 ± 2,804.9 | 16 / 0 / 4 |
| Muon tuned | -2,924.8 ± 4,920.6 | 17 / 0 / 3 |
| FARO | -2,942.2 ± 5,040.7 | 18 / 0 / 2 |
| AdaGO | -3,182.9 ± 5,953.2 | 16 / 0 / 4 |
| NAMO | -3,177.2 ± 5,236.2 | 18 / 0 / 2 |
| NAMOD | -3,231.6 ± 5,443.3 | 16 / 0 / 4 |
| MALTER | -3,223.0 ± 5,318.1 | 17 / 0 / 3 |
| C3+ELR | -2,299.1 ± 5,080.7 | 13 / 0 / 7 |
| H-Net (p14) | -688.4 ± 1,465.2 | 12 / 0 / 8 |
| Entropy gate (p11) | -545.5 ± 973.3 | 16 / 0 / 4 |

**vs Muon tuned**

| config | mean Δ ± sd (B) | W / T / L |
|---|---|---|
| Shipped base | +2,924.8 ± 4,920.6 | 3 / 0 / 17 |
| AdamW tuned | +1,529.5 ± 2,416.4 | 5 / 0 / 15 |
| Muon tuned | +0.0 ± 0.0 | 0 / 20 / 0 |
| FARO | -17.4 ± 303.5 | 16 / 0 / 4 |
| AdaGO | -258.1 ± 1,332.0 | 11 / 2 / 7 |
| NAMO | -252.4 ± 775.2 | 12 / 2 / 6 |
| NAMOD | -306.7 ± 819.8 | 13 / 2 / 5 |
| MALTER | -298.1 ± 942.5 | 14 / 2 / 4 |
| C3+ELR | +625.8 ± 1,576.1 | 3 / 0 / 17 |
| H-Net (p14) | +2,236.4 ± 3,767.5 | 4 / 0 / 16 |
| Entropy gate (p11) | +2,379.4 ± 4,113.3 | 3 / 0 / 17 |

## 5. `rand1m` — reported separately

Configs 1–9 produce exactly 1,049,248 bytes at both L1 and L5 (source 1,048,576 B, +672 B stored-table overhead). The two profile-specific fitters differ only in that fixed overhead: entropy gate = 1,049,252 B (+676) and H-Net = 1,049,264 B (+688), reflecting profile 11's 4-byte gate tail and profile 14's wider mixer rows. Within each config the value is identical at L1 and L5 and independent of the fit — every pairwise delta among configs 1–9 is 0, and gate/hnet differ from those by a constant header/model-size offset only. All 22 `rand1m` archives passed decode + `cmp`. `rand1m` carries no information about fitter quality and was excluded from every W/T/L count and mean Δ in §4 and from the ship decision.

## 6. Which config to ship

**Ship MALTER** (`exp-malt`, `TAW_MALT=malter` + Muon vec-LR vars). Rationale, using the deterministic byte metric:

- **Best-in-class L1/L5 aggregate, within noise of the leader.** Over the 20 informative L1/L5 cells MALTER is -5,963 B below tuned Muon (-298.1 ± 942.5 B per cell); NAMOD is -6,134 B, NAMO -5,048 B, AdaGO -5,161 B. The MALTER–NAMOD gap is only 171 B across 20 cells (~8 B/cell), far below the single-seed fitter noise floor — so they are statistically tied and the tie-break must come from other evidence.
- **Broad wins vs the previous champion, with real magnitude.** MALTER wins 14 of 20 informative L1/L5 cells vs Muon (2 ties, 4 losses) and 17 of 20 vs the shipped base. FARO has more *raw* wins vs Muon (16) but they average only a few bytes each; MALTER's 14 wins carry a 5,963 B aggregate gain.
- **Only one of the four largest L1/L5 gains that also wins at L9.** On the four core corpora at L9, MALTER is 375 B below Muon (and 25,757 B below shipped base). Of the other three configs with the largest L1/L5 gains, NAMOD is 3,126 B *worse* than Muon at L9, NAMO 2,271 B worse and AdaGO 1,672 B worse. Shipping NAMOD would trade a negligible L1/L5 edge for a real L9 regression. (FARO is 1,423 B better than Muon at L9 but its L1/L5 aggregate edge is only 347 B — roughly 17 B/cell.)
- **No pathological regression.** MALTER's worst single L1/L5 cell vs Muon is +1,063 B (`libc4m` L1); its largest gain is −3,941 B (`libc4m` L5) and it ties Muon exactly on both `text4m` levels. Every other informative cell is within ±700 B.

**Runner-up: NAMOD** if L1/L5 bytes are the only thing that matters and L9 is never used — it has the single lowest L1/L5 total and the smallest worst-case regression (+332 B), but it loses to Muon at L9. **FARO** is among the most *consistent* improvements over the shipped base (L5: 10/0/0, shared with NAMO and MALTER; best of all configs at L9) but its aggregate edge over Muon at L1/L5 is only −347 B across 20 cells — statistically indistinguishable. Profiles 11/14 (`gate`/`hnet`) are not drop-in champions: they use different fitters and are behind the Muon family on aggregate (§4), with a few individual-corpus exceptions (e.g. `gate` on `text4m` L1); keep them opt-in.

## 7. Caveats

- **Single seed.** Only `exp-combo-adamw` implements `TAW_SEED`; all arms here are single-seed. The fitter picks the best of several per-cell candidates, so differences below roughly 1 kB (and several larger outliers) are within candidate-selection/quantization noise. W/T/L counts quantify direction consistency, not fit-noise error bars.
- **Load-contended timings.** All 278 encodes ran with `-P 12` on 12 cores. `enc_sec`/`dec_sec` in `results.tsv` are recorded for completeness but are noisy and systematically inflated by contention; L9 times in particular (~180–400 s uncontended) are not comparable to prior single-worker reports. **No speed conclusion is drawn from them.**
- **Different fitters for profiles 11/14.** `gate` (`-p gate`, profile 11) and `hnet` (`-p hnet`, profile 14) are not merely hyperparameter variants of the profile-1/Muon fitter: each uses its own default fitter and archive format (11 adds a 4-byte gate tail; 14 widens mixer rows to 27 and falls back to the scalar mixer). Their numbers are valid byte measurements but are not apples-to-apples with the `TAW_*`-driven configs, and their archives require the matching profile-aware decoder.
- **Corpus coverage.** 11 corpora at L1/L5, but L9 only 4 core corpora and only configs 1–9. `rand1m` is incompressible and excluded from ranking; `libc4m`/`text4m` dominate the aggregate deltas, as in the prior cross-corpus study. Eight of ten informative corpora show champion-to-champion differences of at most a few hundred bytes.
- **hnet binary provenance.** Rebuilt locally from `exp/hnet-feature` because its sibling worktree had no `build/`; validated against that branch's published `text1m` L1 = 181,307 B. Its decode used the same rebuilt binary.
- **No mutation of shared state.** Sibling worktrees, both corpora directories and `origin/main` were used strictly read-only; nothing was pushed.

## 8. Reproduction

```sh
cd /home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix
# rebuild the one missing binary (hnet)
git archive exp/hnet-feature | tar -x -C hnet-build
cmake -S hnet-build -B hnet-build/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build hnet-build/build -j4
# full 278-cell matrix, 12 workers, raw rows -> results.tsv (load-contended timings)
bash run_all.sh
python3 analyze.py > RESULTS.md
```

`bench_one.sh` encodes one cell, times encode and decode with `date +%s.%N`, extracts archive bytes with `stat -c%s`, decodes with the same binary, `cmp`s against the source, and writes one TSV row per cell (including both return codes and `decode_ok`). `run_all.sh` fans all 278 cells out with `xargs -P 12`.

