#!/bin/bash
# H-Net encoder-only sweep. Usage: ./bench/sweep.sh [configs-file]
# Each line of the config file: NAME<TAB>ENV_ASSIGNMENTS (space separated)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAW="$ROOT/build/tools/taw"
DATA=/tmp/opencode/tadamw-bench
OUT="$ROOT/bench/sweep.tsv"
ARCS="$ROOT/bench/arcs"
CFG=${1:-"$ROOT/bench/configs.tsv"}
mkdir -p "$ARCS"
if [ ! -f "$OUT" ]; then
  printf 'name\tenv\tcorpus\tlevel\tsize\tarchive\n' > "$OUT"
fi
while IFS=$'\t' read -r name envstr; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  for c in text1m libc1m; do
    for l in 1 5; do
      key="${name}_${c}_L${l}"
      if awk -F'\t' -v k="$key" '$1"_"$3"_L"$4==k{f=1} END{exit !f}' "$OUT"; then
        continue
      fi
      arc="$ARCS/${key}.aw"
      # shellcheck disable=SC2086
      env $envstr "$TAW" c -l "$l" -p mix "$DATA/$c" "$arc" >/dev/null 2>&1
      st=$?
      if [ $st -ne 0 ]; then
        printf 'ERROR %s (%s)\n' "$key" "$st" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$envstr" "$c" "$l" "ERR$st" "$arc" >> "$OUT"
        continue
      fi
      sz=$(stat -c %s "$arc")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$envstr" "$c" "$l" "$sz" "$arc" >> "$OUT"
      printf '%-26s %-8s L%s %8s\n' "$name" "$c" "$l" "$sz"
    done
  done
done < "$CFG"
