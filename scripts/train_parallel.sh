#!/usr/bin/env bash
# Full-model SFT on several benchmarks at once, sharing one large GPU (A100 80 GB: each run at
# batch 24 x accum 4 needs ~10-11 GiB) and splitting the CPUs between the runs' data loaders.
# Each run is pinned to its own cores, so video decoding of one run never stalls another.
#
#   bash scripts/train_parallel.sh                         # libero, libero_plus, vlabench in parallel
#   BENCHES="libero vlabench" bash scripts/train_parallel.sh
#   bash scripts/train_parallel.sh -- run.steps=50 hub.push=false project.save_every=25   # smoke test
#
# Always resumable: re-run the same command after a crash, kill, reboot or lost SSH session.
#   - Each run continues from its newest complete checkpoint (weights, optimizer, LR schedule, EMA,
#     RNG, data position, W&B run).
#   - A finished run is skipped.
#   - A run that crashes is restarted from its last checkpoint, up to RETRIES times.
#   - Checkpoints are written atomically (step_N.tmp, then renamed), every SAVE_EVERY steps.
#   - Only the newest KEEP_LAST are kept (~12 GB each).
# Run it detached:
#   nohup bash scripts/train_parallel.sh > results/plan/train.log 2>&1 &
# Use the same trainer overrides on every re-run (a resume must match the original config).
#
# Outputs: outputs/<benchmark>/sft (as `benchmark.py train sft`); logs: results/plan/train_<b>.log.
# Needs the full feature cache of each benchmark (scripts/prepare_data.sh cache).
# Knobs: BENCHES, WORKERS (loader processes per run), RETRIES (3), SAVE_EVERY (1000), KEEP_LAST (2).
set -uo pipefail
cd "$(dirname "$0")/.."
BENCHES=${BENCHES:-"libero libero_plus vlabench"}
RETRIES=${RETRIES:-3}
SAVE_EVERY=${SAVE_EVERY:-1000}
KEEP_LAST=${KEEP_LAST:-2}
read -r -a BS <<< "$BENCHES"
N=${#BS[@]}
CPUS=$(nproc)
PER=$(( CPUS / N ))                     # cores per run
WORKERS=${WORKERS:-$(( PER > 1 ? PER - 1 : 1 ))}  # data-loader processes per run (one core left for the trainer)
LOG=results/plan
mkdir -p "$LOG"

[ "${1:-}" = "--resume" ] && shift      # accepted for compatibility: resuming is always on
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

run_one() {  # run_one <benchmark> <first core> <last core>: train, restarting from the last checkpoint on failure
    local b=$1 first=$2 last=$3 attempt code
    for attempt in $(seq 1 "$RETRIES"); do
        echo "== $(date '+%F %T') start sft $b (attempt $attempt/$RETRIES) on cores $first-$last, $WORKERS loader workers" | tee -a "$LOG/plan.log"
        OMP_NUM_THREADS=1 taskset -c "$first-$last" python3 scripts/benchmark.py train sft --benchmark "$b" --resume \
            -- run.num_workers="$WORKERS" project.save_every="$SAVE_EVERY" run.keep_last="$KEEP_LAST" "${EXTRA[@]}" \
            >> "$LOG/train_$b.log" 2>&1
        code=$?
        echo "== $(date '+%F %T') end sft $b (exit $code)" | tee -a "$LOG/plan.log"
        [ $code -eq 0 ] && { { [ -f scripts/notify.sh ] && bash scripts/notify.sh "✅ SFT $b finished" || true; }; return 0; }
        sleep 60
    done
    [ -f scripts/notify.sh ] && bash scripts/notify.sh "❌ SFT $b failed $RETRIES times (exit $code): see $LOG/train_$b.log"
    return $code
}

pids=()
for i in "${!BS[@]}"; do
    first=$(( i * PER )); last=$(( first + PER - 1 ))
    run_one "${BS[$i]}" "$first" "$last" &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { rc=1; echo "== sft ${BS[$i]} failed after $RETRIES attempts: see $LOG/train_${BS[$i]}.log" | tee -a "$LOG/plan.log"; }
done
exit $rc
