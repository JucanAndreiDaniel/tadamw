#!/bin/bash
# Local benchmark: outputs go under ./bench-out, corpora are read-only.
set -eu
cd "$(dirname "$0")"
TAW=./build/tools/taw
DATA=/tmp/opencode/tadamw-bench
OUT=${OUT:-bench-out}
mkdir -p "$OUT"
LEVELS=${LEVELS:-1 5}
CORPORA=${CORPORA:-text1m libc1m}
QUANTS=${QUANTS:-q8 f16}
DBG=${DBG:-0}

printf 'corpus\tlevel\tquant\tenc_s\tsize\tmodel\tbpc_auto\n'
for c in $CORPORA; do
  for w in $QUANTS; do
    for l in $LEVELS; do
      src=$DATA/$c
      tag="${c}_L${l}_w${w}"
      arc="$OUT/${tag}.aw"
      t0=$(date +%s.%N)
      if [ "$DBG" = 1 ]; then
        TAW_FIT_DEBUG=1 "$TAW" c -l "$l" -p mix -w "$w" "$src" "$arc" 2>"$OUT/${tag}.dbg"
      else
        "$TAW" c -l "$l" -p mix -w "$w" "$src" "$arc"
      fi
      t1=$(date +%s.%N)
      enc=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
      model=$("$TAW" i "$arc" | awk '/model_size/{print $2}')
      size=$(stat -c %s "$arc")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t\n' "$c" "$l" "$w" "$enc" "$size" "$model"
    done
  done
done
