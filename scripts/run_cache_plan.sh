#!/usr/bin/env bash
# Feature/label caches for every benchmark, one GPU, in order; every stage is safe to re-run
# (labelled episodes and scored traces are skipped). Each benchmark labels its frontal camera
# (scripts/benchmark.py CACHE).
#   1. RoboTwin subset10 (labels done: bootstrap traces)
#   2. LIBERO Spatial, Goal, 10, Object (lerobot/libero, streamed)
#   3. VLABench subset (local copy under DATA_ROOT; waits for its download)
#   4. LIBERO-plus subset (local copy under DATA_ROOT; waits for its download)
#
#   bash scripts/run_cache_plan.sh            # logs: results/plan/
set -u
cd "$(dirname "$0")/.."
LOG=results/plan
mkdir -p "$LOG"
GPU_CAP=${GPU_CAP:-14}

stage() {  # stage <name> <command...>: run, log, continue on failure
    local name=$1; shift
    echo "== $(date '+%F %T') start $name: $*" | tee -a "$LOG/plan.log"
    "$@" >> "$LOG/$name.log" 2>&1
    local rc=$?
    echo "== $(date '+%F %T') end $name (exit $rc)" | tee -a "$LOG/plan.log"
}

wait_download() {  # wait_download <repo>: until no `hf download <repo>` runs
    while pgrep -f "^\S*python\S* \S*hf download $1 " > /dev/null; do sleep 120; done  # the downloader itself, not its shell
}

# another cache writer on the same cache would corrupt it: wait for it to finish
while pgrep -f "scripts/make_pseudolabels.py" > /dev/null; do sleep 60; done

stage cache_robotwin_subset10 python3 scripts/benchmark.py cache --benchmark robotwin --suite subset10 --gpu-mem-gb "$GPU_CAP"
for s in libero_spatial libero_goal libero_10 libero_object; do
    stage "cache_$s" python3 scripts/benchmark.py cache --suite "$s" --gpu-mem-gb "$GPU_CAP"
done
wait_download lerobot/vlabench_unified
stage cache_vlabench_subset python3 scripts/benchmark.py cache --benchmark vlabench --suite subset --gpu-mem-gb "$GPU_CAP"
wait_download lerobot/libero_plus
stage cache_libero_plus_subset python3 scripts/benchmark.py cache --benchmark libero_plus --suite subset --gpu-mem-gb "$GPU_CAP"
