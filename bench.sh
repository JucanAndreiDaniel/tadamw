#!/bin/bash
# Benchmark one configuration. Usage: NAME=tag [env overrides...] ./bench.sh
# Reads shared corpora read-only; writes archives/decoded output into this
# worktree only. Archive bytes are the metric; decode is verified byte-exact.
set -eu
cd "$(dirname "$0")"
NAME=${NAME:?need NAME}
TAW=./build/tools/taw
DATA=/tmp/opencode/tadamw-bench
mkdir -p bench/arcs bench/dec
T=bench/results.tsv
[ -f "$T" ] || printf 'name\tmode\tconfig\tcorpus\tlevel\tbytes\tdec_s\tdecode_ok\n' > "$T"
for c in text1m libc1m; do
  for l in 1 5; do
    arc=bench/arcs/${NAME}_${c}_L${l}.aw
    $TAW c -l "$l" -p mix "$DATA/$c" "$arc"
    t0=$(date +%s.%N)
    $TAW d "$arc" bench/dec/out
    t1=$(date +%s.%N)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    if cmp -s bench/dec/out "$DATA/$c"; then ok=1; else ok=0; fi
    sz=$(stat -c %s "$arc")
    mode=${TAW_HYBRID:-adam}
    [ -z "${TAW_ADAGO:-}" ] || mode="$mode/${TAW_ADAGO}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$NAME" "$mode" "${CFG:-}" "$c" "$l" "$sz" "$dt" "$ok" | tee -a "$T"
  done
done
