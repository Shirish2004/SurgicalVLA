#!/usr/bin/env bash
# LIBERO-Spatial evaluation of the full-architecture SFT run plus inference-time component probes.
#
# Waits for the training job to exit and its final checkpoint (step_$STEPS) to exist, then runs,
# each on 10 tasks x 50 init states x seeds 1000 2000 (same harness, same init states):
#
#   full          the model as trained: live SAM 3 objects, MoE, latent CoT (instrumented + CoT probe)
#   smolvla       released lerobot/smolvla_libero (seed 1000 already exists; only missing seeds run)
#   obj_zero      object tokens hidden            (object_token_ablation=zero)
#   moe_shared    routed experts dropped           (moe_routing_ablation=shared_only)
#   latent_drop   latent reasoning tokens removed  (latent_cot_ablation=drop)
#   moe_random    router choice replaced by random (moe_routing_ablation=random)
#   obj_shuffle   object slots permuted            (object_token_ablation=shuffle)
#
# The probes change the trained model at test time only; they measure how much the model relies
# on a component, not what training with/without it would give (that needs the ablation arms).
# Re-runnable: finished (condition, seed) pairs are skipped. Figures are rebuilt after every stage.
#
#   nohup bash scripts/run_spatial_ablation_eval.sh > results/plan/eval_spatial_ablation.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
TRAIN_PID=${TRAIN_PID:-715030}
STEPS=${STEPS:-25000}
RUN=${RUN:-outputs/libero_spatial/full_25k/seed0}
OUT=${OUT:-results/libero/spatial_full25k_eval}
SEEDS=${SEEDS:-"1000 2000"}
RES=${RES:-256}                  # the policy's training resolution (lerobot/libero frames are 256x256)
# Before starting, also wait for these processes (e.g. an Isaac Sim job) to exit and for the
# GPU to have MIN_FREE_MB free for GPU_QUIET_MIN consecutive minutes.
WAIT_PIDS=${WAIT_PIDS:-}
MIN_FREE_MB=${MIN_FREE_MB:-12000}
GPU_QUIET_MIN=${GPU_QUIET_MIN:-3}
BASELINE=results/baselines/smolvla_libero
N=~/.local/bin/notify
note() { echo "== $(date '+%F %T') $1"; [ -x "$N" ] && "$N" "$1" || true; }
PY="pixi run --manifest-path $PWD/pixi.toml -e sim-eval python"
CKPT="$RUN/checkpoints/step_$STEPS"

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
for pid in $WAIT_PIDS; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
quiet=0
while [ "$quiet" -lt "$GPU_QUIET_MIN" ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge "$MIN_FREE_MB" ]; then quiet=$((quiet + 1)); else quiet=0; fi
    [ "$quiet" -lt "$GPU_QUIET_MIN" ] && sleep 60
done
if [ ! -e "$CKPT/config.json" ]; then
    note "❌ Spatial eval: training ended but $CKPT has no policy; eval not started"
    exit 1
fi
mkdir -p "$OUT"
echo "$CKPT" > "$OUT/checkpoint.txt"

report() {
    [ -s "$OUT/full/summary.json" ] && { $PY scripts/eval_report.py --root "$OUT/full" \
        || note "⚠️ full-model report failed (see results/plan/eval_spatial_ablation.log)"; }
    $PY scripts/eval_ablation_report.py --root "$OUT" --baseline "$BASELINE" --train-run "$RUN" \
        || note "⚠️ ablation report failed (see results/plan/eval_spatial_ablation.log)"
}

# condition name, instrument (1/0), CoT probe every k chunks, policy overrides...
evaluate() {
    local name=$1 instr=$2 probe=$3; shift 3
    local args=()
    for kv in "$@"; do args+=(--policy-arg "$kv"); done
    note "🚀 Spatial eval: $name (seeds $SEEDS)"
    local t0=$SECONDS
    if ! SVLA_EVAL_INSTRUMENT=$instr SVLA_COT_PROBE_EVERY=$probe SVLA_ANNOTATE_TASKS=$([ "$name" = full ] && echo 10 || echo 0) \
        $PY scripts/eval_libero_baseline.py run --policy "$CKPT" --revision "" --suites libero_spatial \
            --resolution "$RES" --seeds $SEEDS --out "$OUT/$name" "${args[@]}"; then
        note "❌ Spatial eval $name failed: see $OUT/$name/libero_spatial/seed*/stdout.log"
        return 1
    fi
    $PY scripts/eval_libero_baseline.py aggregate --root "$OUT/$name" > "$OUT/$name/aggregate.txt"
    note "✅ $name done in $(( (SECONDS - t0) / 60 )) min: $(head -1 "$OUT/$name/aggregate.txt")"
}

# 1. smoke test: 1 task x 2 episodes, instrumented, every probe once
SMOKE=$OUT/_smoke
rm -rf "$SMOKE"
note "🧪 GPU free; smoke-testing the eval on $CKPT"
for spec in "none" "moe_routing_ablation=shared_only" "latent_cot_ablation=drop" "object_token_ablation=zero"; do
    extra=(); [ "$spec" != none ] && extra=(--policy-arg "$spec")
    if ! SVLA_EVAL_INSTRUMENT=1 $PY scripts/eval_libero_baseline.py run --policy "$CKPT" --revision "" \
            --suites libero_spatial --resolution "$RES" --seeds 1000 --n-episodes 2 --batch-size 2 --task-ids 0 \
            --out "$SMOKE/${spec//=/_}" "${extra[@]}"; then
        note "❌ Smoke test failed for '$spec' (see $SMOKE); eval not started"
        exit 1
    fi
done
note "✅ Smoke test passed"

# 2. the model as trained (instrumented: perception, latency, expert usage, CoT probe, annotated clips)
evaluate full 1 4 || exit 1
# 3. the matched seeds of the released SmolVLA baseline
note "🚀 Spatial eval: smolvla baseline"
$PY scripts/eval_libero_baseline.py run --suites libero_spatial --seeds $SEEDS --out "$BASELINE" \
    && $PY scripts/eval_libero_baseline.py aggregate --root "$BASELINE" > "$BASELINE/aggregate.txt" \
    || note "⚠️ smolvla baseline eval failed; continuing with the probes"
report
# 4. inference-time probes, most informative first
evaluate obj_zero    1 0 object_token_ablation=zero        && report
evaluate moe_shared  1 0 moe_routing_ablation=shared_only  && report
evaluate latent_drop 1 0 latent_cot_ablation=drop          && report
evaluate moe_random  1 0 moe_routing_ablation=random       && report
evaluate obj_shuffle 1 0 object_token_ablation=shuffle     && report
note "📊 Spatial eval complete: $OUT/report.md, figures in $OUT/figures"
