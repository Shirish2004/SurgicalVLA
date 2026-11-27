#!/usr/bin/env bash
# Data preparation for the full run (LIBERO x4 suites, VLABench, LIBERO-plus; every episode) on one
# big GPU: frozen-model weights, a 3-episode pilot, feature caches (object inputs + pseudo-label
# targets + bootstrap CoT traces) with N labelling processes per benchmark, then one push of all
# caches to a private Hub dataset. Every step is resumable: re-run the same command after a stop.
#
#   bash scripts/prepare_data.sh models      # weights of every frozen model (SAM 3 is gated: accept its licence first)
#   bash scripts/prepare_data.sh pilot       # 3 episodes per benchmark -> results/pseudolabels/<b>_all_cost.json
#   bash scripts/prepare_data.sh cache       # full caches, in order libero -> libero_plus -> vlabench
#   bash scripts/prepare_data.sh push        # pseudolabels/ -> hf.co/datasets/$CACHE_REPO (resumable)
#   bash scripts/prepare_data.sh all         # models, cache, push
#
# Datasets are read from <repo>/data/{libero,libero_plus,vla_bench} (or SURGICALVLA_DATA_ROOT,
# LIBERO_ROOT, LIBERO_PLUS_ROOT, VLABENCH_ROOT). Logs: results/plan/.
# Knobs: WORKERS (labelling processes per benchmark, default 5), GPU_CAP (GiB per process, 14),
#        CACHE_REPO (Shirish24/surgical-vla-cache), BENCHES ("libero libero_plus vlabench").
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-5}
GPU_CAP=${GPU_CAP:-14}
CACHE_REPO=${CACHE_REPO:-Shirish24/surgical-vla-cache}
BENCHES=${BENCHES:-"libero libero_plus vlabench"}
export HF_XET_HIGH_PERFORMANCE=1
LOG=results/plan
mkdir -p "$LOG"
hf() { pixi run -e dev hf "$@"; }

models() {
    # perception (labelling) + phrase VLM + base policy + the SmolVLA checkpoints that score the traces;
    # revisions pinned where the launcher pins them (scripts/benchmark.py SMOLVLA_BASE, CACHE[*].ckpt)
    hf download facebook/sam3 > /dev/null || { echo "SAM 3 is gated: accept the licence at https://huggingface.co/facebook/sam3"; exit 1; }
    hf download facebook/dinov2-small > /dev/null
    hf download depth-anything/Depth-Anything-V2-Small-hf > /dev/null
    hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct > /dev/null
    hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 > /dev/null
    hf download lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d > /dev/null
    hf download lerobot/smolvla_vlabench --revision 7ef370813eb35823caa94088b50f5389a51e4c47 > /dev/null
    local ckpt=$HOME/.cache/surgicalvla/tapnextpp_ckpt.pt  # TAPNext++ point tracker (Apache-2.0), 2.5 GB
    if [ ! -s "$ckpt" ]; then
        mkdir -p "$(dirname "$ckpt")"
        curl -L --fail -o "$ckpt.part" https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt && mv "$ckpt.part" "$ckpt"
    fi
    echo "models ready"
}

pilot() {  # cost per episode on this machine, one process; the episodes stay in the cache
    for b in $BENCHES; do
        python3 scripts/benchmark.py cache --benchmark "$b" --suite all --limit 3 --gpu-mem-gb "$GPU_CAP" 2>&1 | tee "$LOG/pilot_$b.log"
    done
    grep -h '"gpu_hours"' results/pseudolabels/*_all_cost.json || true
}

cache() {  # each benchmark: N labelling processes, then N trace processes, all on one cache
    for b in $BENCHES; do
        echo "== $(date '+%F %T') start cache $b (workers $WORKERS)" | tee -a "$LOG/plan.log"
        rc=0
        python3 scripts/benchmark.py cache --benchmark "$b" --suite all --workers "$WORKERS" \
            --gpu-mem-gb "$GPU_CAP" --no-videos >> "$LOG/cache_${b}_all.log" 2>&1 || rc=$?
        echo "== $(date '+%F %T') end cache $b (exit $rc)" | tee -a "$LOG/plan.log"  # a failure does not stop the next benchmark
    done
}

push() {  # whole cache tree, same layout FeatureCache uses on the Hub (<dataset-slug>/<cache-name>/...)
    hf upload-large-folder "$CACHE_REPO" pseudolabels --repo-type dataset --private \
        --exclude "*.lock" "*.tmp" "*/.lock" --num-workers 8
}

case "${1:-all}" in
    models) models ;;
    pilot) pilot ;;
    cache) cache ;;
    push) push ;;
    all) models; cache; push ;;
    *) echo "usage: $0 {models|pilot|cache|push|all}"; exit 2 ;;
esac
