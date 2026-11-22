#!/bin/bash
# LR control: compare averaging vs a plain-AdamW LR retune.
set -eu
cd "$(dirname "$0")/.."
TAW=./build/tools/taw
DATA=/tmp/opencode/tadamw-bench
for s in 1 0.1 0.15 0.25; do
  for m in off ema999; do
    for c in text1m libc1m; do
      for l in 1 5; do
        arc=bench/ctl_${c}_L${l}_lr${s}_${m}.aw
        TAW_FIT_AVG=$m TAW_FIT_LRSCALE=$s "$TAW" c -l "$l" -p mix "$DATA/$c" "$arc"
        printf 'lrscale=%s mode=%s %s L%s size=%s model=%s\n' \
          "$s" "$m" "$c" "$l" "$(stat -c %s "$arc")" \
          $("$TAW" i "$arc" | awk '/model_size/{print $2}')
      done
    done
  done
done
