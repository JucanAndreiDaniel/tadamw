#!/bin/bash
# One-off sweep helper (local to this worktree). Usage:
#   NAME=tag CFG="..." ./sweep.sh [L1|L5|both]
# Reads shared corpora read-only; writes only bench/sweep/arcs here.
set -eu
cd "$(dirname "$0")"
NAME=${NAME:?need NAME}
LEVELS=${1:-both}
TAW=./build/tools/taw
DATA=/tmp/opencode/tadamw-bench
mkdir -p bench/sweep/arcs
T=bench/sweep.tsv
[ -f "$T" ] || printf 'name\tconfig\tcorpus\tlevel\tbytes\n' > "$T"
if [ "$LEVELS" = both ]; then LS="1 5"; else LS="${LEVELS#L}"; fi
for c in text1m libc1m; do
  for l in $LS; do
    arc=bench/sweep/arcs/${NAME}_${c}_L${l}.aw
    env $CFG $TAW c -l "$l" -p mix "$DATA/$c" "$arc"
    sz=$(stat -c %s "$arc")
    printf '%s\t%s\t%s\t%s\t%s\n' "$NAME" "$CFG" "$c" "$l" "$sz" | tee -a "$T"
  done
done
