#!/usr/bin/env bash
# LIBERO plan, one GPU, every stage safe to re-run (finished work is skipped):
#   1. feature cache for LIBERO-Spatial
#   2. feature caches for Object, Goal, Long (same frozen pipeline settings)
#   3. Spatial ablation ladder (configs/ablations/arms.yaml: 8,000 steps, 3 seeds, then evaluation)
#   4. per-suite arms (smolvla, full, full_lora) on Object, Goal, Long
#   5. results.json, figures, LaTeX tables, docs/baselines.md tables
#
#   bash scripts/run_libero_plan.sh            # logs: results/plan/
set -u
cd "$(dirname "$0")/.."
LOG=results/plan
mkdir -p "$LOG"
GPU_CAP=${GPU_CAP:-12}

stage() {  # stage <name> <command...>: run, log, continue on failure
    local name=$1; shift
    echo "== $(date '+%F %T') start $name: $*" | tee -a "$LOG/plan.log"
    "$@" >> "$LOG/$name.log" 2>&1
    echo "== $(date '+%F %T') end $name (exit $?)" | tee -a "$LOG/plan.log"
}

# another cache writer on the same cache would corrupt it: wait for it to finish
while pgrep -f "scripts/make_pseudolabels.py" > /dev/null; do sleep 60; done

stage cache_libero_spatial python3 scripts/benchmark.py cache --suite libero_spatial --gpu-mem-gb "$GPU_CAP"
for s in libero_object libero_goal libero_10; do
    stage "cache_$s" python3 scripts/benchmark.py cache --suite "$s" --gpu-mem-gb "$GPU_CAP"
done
stage ablate_libero_spatial python3 scripts/benchmark.py ablate --suite libero_spatial
for s in libero_object libero_goal libero_10; do
    stage "ablate_$s" python3 scripts/benchmark.py ablate --suite "$s"
done
stage figures pixi run -e dev figures
