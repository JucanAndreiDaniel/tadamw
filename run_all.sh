#!/usr/bin/env bash
# Full definitive matrix. 12 parallel workers, including L9 (per updated plan).
set -euo pipefail
HERE=/home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix

CONFIGS="base adamw muon faro adago namo namod malter c3elr hnet gate"
CORPORA="text1m libc1m code1m json1m html1m log1m dna1m rand1m mixed1m text4m libc4m"
CORE4="text4m libc4m code1m json1m"
L9CFG="base adamw muon faro adago namo namod malter c3elr"

: > "$HERE/jobs.tsv"
for cfg in $CONFIGS; do
  for corp in $CORPORA; do
    for lvl in 1 5; do
      printf '%s\t%s\t%s\n' "$cfg" "$corp" "$lvl" >> "$HERE/jobs.tsv"
    done
  done
done
for cfg in $L9CFG; do
  for corp in $CORE4; do
    printf '%s\t%s\t%s\n' "$cfg" "$corp" 9 >> "$HERE/jobs.tsv"
  done
done

echo "jobs: $(wc -l < "$HERE/jobs.tsv")" >&2

# xargs receives space-separated triples; -P 12 saturates all cores.
tr '\t' ' ' < "$HERE/jobs.tsv" | \
  xargs -P 12 -n 3 bash "$HERE/bench_one.sh"

# Concatenate per-cell rows into the raw results table with a header.
{
  printf 'config\tcorpus\tlevel\tsrc_bytes\tarchive_bytes\tenc_sec\tdec_sec\tdecode_ok\tenc_rc\tdec_rc\n'
  cat "$HERE"/rows/*.tsv
} | sort -t$'\t' -k1,1 -k2,2 -k3,3n > "$HERE/results.tsv"

echo "rows: $(($(wc -l < "$HERE/results.tsv") - 1))" >&2
