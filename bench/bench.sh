#!/bin/bash
# Local A/B benchmark. Usage: MODE=off|swa|ema9|ema99 ./bench/bench.sh
set -eu
cd "$(dirname "$0")/.."
MODE=${MODE:-off}
TAW=./build/tools/taw
DATA=/tmp/opencode/tadamw-bench
OUT=bench/res_${MODE}.tsv

printf 'mode\tcorpus\tlevel\tenc_s\tdec_s_best\tsize\tmodel\n' > "$OUT"
for c in text1m libc1m; do
  for l in 1 5; do
    src=$DATA/$c
    arc=bench/${c}_${MODE}_L${l}.aw
    t0=$(date +%s.%N)
    TAW_FIT_AVG=$MODE "$TAW" c -l "$l" -p mix "$src" "$arc"
    t1=$(date +%s.%N)
    enc=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    model=$("$TAW" i "$arc" | awk '/model_size/{print $2}')
    size=$(stat -c %s "$arc")
    for run in 1 2 3; do
      t0=$(date +%s.%N)
      "$TAW" d "$arc" /dev/null
      t1=$(date +%s.%N)
      awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f\n", b-a}'
    done > bench/times_${MODE}_${c}_L${l}
    dec=$(sort -n bench/times_${MODE}_${c}_L${l} | head -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODE" "$c" "$l" "$enc" "$dec" "$size" "$model" | tee -a "$OUT"
  done
done
