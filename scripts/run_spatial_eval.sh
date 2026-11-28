#!/usr/bin/env bash
# Unattended LIBERO-Spatial evaluation of the full-architecture SFT run, once training ends.
#   1. wait for the training job (TRAIN_PID) to exit and its final policy to exist
#   2. smoke test: instrumented eval, 1 task x 2 episodes, plus the report on it;
#      if it fails the full eval still runs, uninstrumented (success metrics only)
#   3. full eval: 10 tasks x 50 episodes x 3 seeds, live SAM 3 / DINOv2 / depth perception,
#      LeRobot rollout videos, instrumentation (perception, latency, MoE usage, CoT probe, annotated clips)
#   4. figures / tables / selected videos (scripts/eval_report.py)
# Telegram alert at every stage. Re-runnable: finished seeds are skipped by eval_libero_baseline.py.
#
#   nohup bash scripts/run_spatial_eval.sh > results/plan/eval_spatial_full25k.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
TRAIN_PID=${TRAIN_PID:-715030}
CKPT=${CKPT:-outputs/libero_spatial/full_25k/seed0}
OUT=${OUT:-results/libero/spatial_full25k_s0}
SMOKE=results/libero/_smoke_spatial_instrument
N=~/.local/bin/notify
note() { echo "== $(date '+%F %T') $1"; [ -x "$N" ] && "$N" "$1" || true; }
PY="pixi run --manifest-path $PWD/pixi.toml -e dev python"

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
if [ ! -e "$CKPT/final" ] && [ ! -e "$CKPT/checkpoints/last/pretrained_model" ]; then
    note "❌ Spatial eval: training ended but no final policy under $CKPT; eval not started"
    exit 1
fi
note "🧪 Training done; smoke-testing instrumented eval on $CKPT"

INSTR=1
rm -rf "$SMOKE"
if SVLA_EVAL_INSTRUMENT=1 $PY scripts/benchmark.py eval --benchmark libero --checkpoint "$CKPT" \
        --suites libero_spatial --seeds 1000 --out "$SMOKE" -- --n-episodes 2 --batch-size 2 --task-ids 0 \
   && [ -s "$SMOKE/libero_spatial/seed1000/instrument/metrics.jsonl" ] \
   && $PY scripts/eval_report.py --root "$SMOKE"; then
    note "✅ Smoke test passed: $(head -2 "$SMOKE/report.md" | tail -1)"
else
    INSTR=0
    note "⚠️ Instrumented smoke test failed (see results/plan/eval_spatial_full25k.log); running the standard eval instead"
fi

note "🚀 LIBERO-Spatial eval started (1500 episodes, instrument=$INSTR)"
if ! SVLA_EVAL_INSTRUMENT=$INSTR $PY scripts/benchmark.py eval --benchmark libero --checkpoint "$CKPT" \
        --suites libero_spatial --out "$OUT"; then
    note "❌ LIBERO-Spatial eval failed: see $OUT/libero_spatial/seed*/stdout.log"
    exit 1
fi

if $PY scripts/eval_report.py --root "$OUT"; then
    note "📊 LIBERO-Spatial done. $(sed -n 2p "$OUT/report.md") Figures: $OUT/figures, videos: $OUT/videos_selected"
else
    note "⚠️ Eval done ($(grep -o '"success_mean": [0-9.]*' "$OUT/summary.json")) but the report failed; results in $OUT"
    exit 1
fi
