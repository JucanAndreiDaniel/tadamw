#!/usr/bin/env bash
# Encode one (config, corpus, level) cell, decode with the SAME binary, cmp-verify.
# All outputs land in this worktree (never in /tmp, never in sibling worktrees).
set -u

WT=/home/andrew/.local/share/opencode/worktree/8500c0
HERE=/home/andrew/.local/share/opencode/worktree/8500c0/exp-final-matrix
CORP=/tmp/opencode/tadamw-bench/corpora2
PARENT=/tmp/opencode/tadamw-bench
OUT=$HERE/out
ROWS=$HERE/rows
mkdir -p "$OUT" "$ROWS"

config=$1; corpus=$2; level=$3

# ---- exact environment per config (scrubbed via `env -i` at invocation) ----
MUON4='TAW_HYBRID=muon TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015'
ADAMW='TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=0.001'
C3ELR="$ADAMW TAW_ELR=1 TAW_ELR_C=0.1 TAW_ELR_SCOPE=shape TAW_ELR_BETA=0.99 TAW_ELR_CLAMP=8 TAW_ELR_SCHED=linear TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2"

case "$config" in
  base)   BIN="$WT/wild-main/build/tools/taw";       PROF=mix;  EV="" ;;
  adamw)  BIN="$WT/exp-lr-baseline/build/tools/taw"; PROF=mix;  EV="$ADAMW" ;;
  muon)   BIN="$WT/exp-muon-adam/build/tools/taw";   PROF=mix;  EV="$MUON4" ;;
  faro)   BIN="$WT/exp-faro/build/tools/taw";        PROF=mix;  EV="$MUON4 TAW_FARO=1 TAW_FARO_ROUTE=muon TAW_FARO_CAP=1.8" ;;
  adago)  BIN="$WT/exp-adago/build/tools/taw";       PROF=mix;  EV="TAW_ADAGO=adago TAW_ADAGO_LR=29 TAW_ADAGO_EPS=1e-3 TAW_MUON_LR0=0.055 TAW_MUON_LR1=0.0011 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015" ;;
  namo)   BIN="$WT/exp-namo/build/tools/taw";        PROF=mix;  EV="TAW_HYBRID=namo TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.006 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01" ;;
  namod)  BIN="$WT/exp-namo/build/tools/taw";        PROF=mix;  EV="TAW_HYBRID=namod TAW_MUON_LR0=0.3 TAW_MUON_LR1=0.015 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015 TAW_NAMO_WD=0.01 TAW_NAMO_BETA=0.999 TAW_NAMO_CLAMP=0.9" ;;
  malter) BIN="$WT/exp-malt/build/tools/taw";        PROF=mix;  EV="TAW_MALT=malter TAW_MALT_LR0=0.28 TAW_MALT_LR1=0.0056 TAW_MALT_B2=0.9 TAW_MALT_EPS=1e-8 TAW_VEC_LR0=0.003 TAW_VEC_LR1=0.00015" ;;
  c3elr)  BIN="$WT/exp-combo-adamw/build/tools/taw"; PROF=mix;  EV="$C3ELR" ;;
  hnet)   BIN="$HERE/hnet-build/build/tools/taw";    PROF=hnet; EV="" ;;
  gate)   BIN="$WT/exp-entropygate/build/tools/taw"; PROF=gate; EV="" ;;
  *) echo "unknown config: $config" >&2; exit 2 ;;
esac

case "$corpus" in
  text1m|libc1m) SRC="$PARENT/$corpus" ;;
  *)             SRC="$CORP/$corpus" ;;
esac

tag="${config}_${corpus}_L${level}"
arc="$OUT/$tag.aw"
dec="$OUT/$tag.dec"
src_bytes=$(stat -c%s "$SRC")

t0=$(date +%s.%N)
# shellcheck disable=SC2086  # $EV is an intentional list of VAR=val words
env -i PATH=/usr/bin:/bin HOME=/home/andrew LANG=C.UTF-8 $EV \
    "$BIN" c -l "$level" -p "$PROF" "$SRC" "$arc" \
    >/dev/null 2>"$OUT/$tag.enc.stderr"
enc_rc=$?
t1=$(date +%s.%N)
arc_bytes=$(stat -c%s "$arc" 2>/dev/null || echo -1)

t2=$(date +%s.%N)
env -i PATH=/usr/bin:/bin HOME=/home/andrew LANG=C.UTF-8 $EV \
    "$BIN" d "$arc" "$dec" \
    >/dev/null 2>"$OUT/$tag.dec.stderr"
dec_rc=$?
t3=$(date +%s.%N)

if [ "$enc_rc" -eq 0 ] && [ "$dec_rc" -eq 0 ] && cmp -s "$SRC" "$dec"; then
  ok=1
else
  ok=0
fi

enc_sec=$(awk "BEGIN{printf \"%.3f\", $t1-$t0}")
dec_sec=$(awk "BEGIN{printf \"%.3f\", $t3-$t2}")

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$config" "$corpus" "$level" "$src_bytes" "$arc_bytes" \
  "$enc_sec" "$dec_sec" "$ok" "$enc_rc" "$dec_rc" > "$ROWS/$tag.tsv"
