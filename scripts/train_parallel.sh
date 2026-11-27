#!/usr/bin/env bash
# Full-model SFT on several benchmarks at once, sharing one large GPU (A100 80 GB: each run at
# batch 24 x accum 4 needs ~10-11 GiB) and splitting the CPUs between the runs' data loaders.
# Each run is pinned to its own cores, so video decoding of one run never stalls another.
#
#   bash scripts/train_parallel.sh                         # libero, libero_plus, vlabench in parallel
#   BENCHES="libero vlabench" bash scripts/train_parallel.sh
#   bash scripts/train_parallel.sh --resume                # continue every run from its newest checkpoint
#   bash scripts/train_parallel.sh -- run.steps=50 hub.push=false   # 50-step smoke test of all runs
#
# Outputs: outputs/<benchmark>/sft (as `benchmark.py train sft`); logs: results/plan/train_<b>.log.
# Needs the full feature cache of each benchmark (scripts/prepare_data.sh cache).
set -uo pipefail
cd "$(dirname "$0")/.."
BENCHES=${BENCHES:-"libero libero_plus vlabench"}
read -r -a BS <<< "$BENCHES"
N=${#BS[@]}
CPUS=$(nproc)
PER=$(( CPUS / N ))                     # cores per run
WORKERS=${WORKERS:-$(( PER > 1 ? PER - 1 : 1 ))}  # data-loader processes per run (one core left for the trainer)
LOG=results/plan
mkdir -p "$LOG"

RESUME=()
if [ "${1:-}" = "--resume" ]; then RESUME=(--resume); shift; fi
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

pids=()
for i in "${!BS[@]}"; do
    b=${BS[$i]}
    first=$(( i * PER )); last=$(( first + PER - 1 ))
    echo "== $(date '+%F %T') start sft $b on cores $first-$last, $WORKERS loader workers" | tee -a "$LOG/plan.log"
    OMP_NUM_THREADS=1 taskset -c "$first-$last" python3 scripts/benchmark.py train sft --benchmark "$b" "${RESUME[@]}" \
        -- run.num_workers="$WORKERS" "${EXTRA[@]}" >> "$LOG/train_$b.log" 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; code=$?
    echo "== $(date '+%F %T') end sft ${BS[$i]} (exit $code)" | tee -a "$LOG/plan.log"
    [ $code -ne 0 ] && rc=$code
done
exit $rc
