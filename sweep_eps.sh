#!/bin/bash
# Reproducible sweep for the C2/C3/NC9 AdamW stabilizers.
# Usage: LEVELS="1 5" CORPORA="text1m libc1m" ./sweep_eps.sh CONFIG... 
# CONFIG is one of the named presets below; output is TSV on stdout.
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
TAW=${TAW:-$ROOT/build/tools/taw}
DATA=/tmp/opencode/tadamw-bench
OUT=${OUT:-/tmp/opencode/eps-gate-bench/arcs}
mkdir -p "$OUT"
LEVELS=${LEVELS:-5}
CORPORA=${CORPORA:-text1m}
VERIFY=${VERIFY:-0}

unset TAW_EPS_MODE TAW_EPS_ALPHA TAW_EPS_C TAW_EPS_FLOOR TAW_SIGN_BETA \
      TAW_GATE_P TAW_GATE_MODE TAW_EPS_LAYERS TAW_EPS_SCOPE TAW_EPS_GRAN \
      TAW_LR0 TAW_LR1 TAW_B1 TAW_B2 TAW_WD TAW_EPS_DEBUG \
      TAW_FORCE_SEL1 TAW_FORCE_SEL2 || true

# Strong plain-AdamW control from exp/lr-baseline (commit 9a08442).
TUNE=(TAW_LR0=0.055 TAW_LR1=0.00055 TAW_B1=0.95 TAW_B2=0.999 TAW_WD=1e-3)

for name in "$@"; do
  case $name in
    base)        E=() ;;
    c2_a05_c1e3) E=(TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3) ;;
    c2_a05_c1e2) E=(TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2) ;;
    c2_a1_c1e3)  E=(TAW_EPS_MODE=c2 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-3) ;;
    c2_a1_c1e2)  E=(TAW_EPS_MODE=c2 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-2) ;;
    c3_b09_p1m)  E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    c3_b09_p2m)  E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=2) ;;
    c3_b095_p1m) E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=1) ;;
    c3_b095_p2m) E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2) ;;
    c3_b09_p1s)  E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    c3_b09_p2s)  E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=2 TAW_GATE_MODE=shrink) ;;
    c3_b095_p1s) E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    c3_b095_p2s) E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2 TAW_GATE_MODE=shrink) ;;
    nc9_a05_c1e3_b09_p1m)  E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    nc9_a05_c1e3_b095_p2m) E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2) ;;
    nc9_a05_c1e2_b09_p1m)  E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    nc9_a1_c1e2_b09_p1m)   E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    nc9_a05_c1e3_b09_p1s)  E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    lr0p003)     E=(TAW_LR0=0.003 TAW_LR1=6e-5) ;;
    lr0p003flat) E=(TAW_LR0=0.003 TAW_LR1=0.003) ;;
    # stabilizers layered on top of the strong tuned-AdamW control
    tune_base)        E=("${TUNE[@]}") ;;
    tune_c2_a05_c1e3) E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3) ;;
    tune_c2_a05_c1e2) E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2) ;;
    tune_c2_a1_c1e3)  E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-3) ;;
    tune_c2_a1_c1e2)  E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-2) ;;
    tune_c3_b09_p1m)  E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    tune_c3_b09_p2m)  E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=2) ;;
    tune_c3_b095_p1m) E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=1) ;;
    tune_c3_b095_p2m) E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2) ;;
    tune_c3_b09_p1s)  E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    tune_c3_b09_p2s)  E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=2 TAW_GATE_MODE=shrink) ;;
    tune_c3_b095_p1s) E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    tune_c3_b095_p2s) E=("${TUNE[@]}" TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2 TAW_GATE_MODE=shrink) ;;
    tune_c2_a05_c5e4) E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=5e-4) ;;
    tune_c2_a05_c3e4) E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=3e-4) ;;
    tune_c2_a1_c1e3) E=("${TUNE[@]}" TAW_EPS_MODE=c2 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-3) ;;
    tune_nc9shrink) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink) ;;
    tune_nc9_a05_c1e3_b09_p1m) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    tune_nc9_a05_c1e3_b095_p2m) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.95 TAW_GATE_P=2) ;;
    tune_nc9_a05_c1e2_b09_p1m) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    tune_nc9_a1_c1e2_b09_p1m) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=1 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1) ;;
    tune_nc9_l12) E=("${TUNE[@]}" TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=12) ;;
    tune_cold_base) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4) ;;
    tune_cold_c2) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_EPS_LAYERS=2) ;;
    tune_cold_c3) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2) ;;
    tune_cold_nc9) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2) ;;
    tune_cold_c3_cold) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=cold) ;;
    tune_cold_c3_warm) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=warm) ;;
    tune_cold_nc9s_cold) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=cold) ;;
    tune_cold_nc9s_warm) E=("${TUNE[@]}" TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_GATE_MODE=shrink TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=warm) ;;
    # layer-1 (and both-layer) arms; use with TAW_EPS_LAYERS=1/12 set outside
    c2_l1)       E=(TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_EPS_LAYERS=1) ;;
    c3_l1)       E=(TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=1) ;;
    nc9_l1)      E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=1) ;;
    nc9_l12)     E=(TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=12) ;;
    # touch-count stratification on a cold-set-heavy shape: sel2=C0 (256 sets,
    # ~100 touches per set at L5) instead of the usual BITPOS (8 hot sets).
    cold_base)   E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4) ;;
    cold_c2)     E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c2 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_EPS_LAYERS=2) ;;
    cold_c3)     E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=c3 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2) ;;
    cold_nc9)    E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2) ;;
    cold_nc9_cold)  E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=cold) ;;
    cold_nc9_warm)  E=(TAW_FORCE_SEL1=0 TAW_FORCE_SEL2=4 TAW_EPS_MODE=nc9 TAW_EPS_ALPHA=0.5 TAW_EPS_C=1e-2 TAW_SIGN_BETA=0.9 TAW_GATE_P=1 TAW_EPS_LAYERS=2 TAW_EPS_SCOPE=warm) ;;
    *) echo "unknown config: $name" >&2; exit 2 ;;
  esac
  if [ "${DEBUG:-0}" = 1 ]; then E+=(TAW_EPS_DEBUG=1); fi
  for c in $CORPORA; do
    for l in $LEVELS; do
      arc="$OUT/${name}_${c}_L${l}.aw"
      env "${E[@]}" "$TAW" c -l "$l" -p mix "$DATA/$c" "$arc"
      if [ "$VERIFY" = 1 ]; then
        env "${E[@]}" "$TAW" d "$arc" "$OUT/${name}_${c}_L${l}.out"
        cmp "$DATA/$c" "$OUT/${name}_${c}_L${l}.out"
        rm -f "$OUT/${name}_${c}_L${l}.out"
      fi
      printf '%s\t%s\t%s\t%s\n' "$name" "$c" "$l" "$(stat -c %s "$arc")"
    done
  done
done
