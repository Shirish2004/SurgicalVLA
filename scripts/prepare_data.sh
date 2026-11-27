#!/usr/bin/env bash
# Data preparation for the full run (LIBERO x4 suites, VLABench, LIBERO-plus; every episode) on one
# big GPU: frozen-model weights, a 3-episode pilot, feature caches (object inputs + pseudo-label
# targets + bootstrap CoT traces) with N labelling processes per benchmark, then one push of all
# caches to a private Hub dataset.
#
#   bash scripts/prepare_data.sh models      # weights of every frozen model (SAM 3 is gated: accept its licence first)
#   bash scripts/prepare_data.sh pilot       # 3 episodes per benchmark -> results/pseudolabels/<b>_all_cost.json
#   bash scripts/prepare_data.sh cache       # full caches, in order libero -> libero_plus -> vlabench
#   bash scripts/prepare_data.sh push        # pseudolabels/ -> hf.co/datasets/$CACHE_REPO
#   bash scripts/prepare_data.sh all         # models, cache, push
#   bash scripts/prepare_data.sh resume      # = all, continuing from wherever the last run stopped
#   bash scripts/prepare_data.sh status      # episodes labelled / traced per benchmark, finished steps
#
# Resuming: every step can be re-run after a crash, kill, reboot or lost SSH session, and picks up
# where it stopped:
#   - weights: hf download and curl continue partial files;
#   - labels: finished episodes are skipped; a killed worker first writes its pending episodes;
#   - traces: written every 20 episodes and on kill; scored episodes are skipped;
#   - push: upload-large-folder skips uploaded files.
# A step that finishes leaves results/plan/state/<step>.done and is skipped by `resume`/`all`
# (delete the file, or call the step by name, to run it again). A failed cache stage is retried
# RETRIES times before moving on. Run it detached so a dropped SSH session does not stop it:
#   nohup bash scripts/prepare_data.sh resume > results/plan/prepare.log 2>&1 &
#
# Datasets are read from <repo>/data/{libero,libero_plus,vla_bench} (or SURGICALVLA_DATA_ROOT,
# LIBERO_ROOT, LIBERO_PLUS_ROOT, VLABENCH_ROOT). Logs: results/plan/.
# Knobs: WORKERS (labelling processes per benchmark, default 5), GPU_CAP (GiB per process, 14),
#        RETRIES (per cache stage, 3), CACHE_REPO (Shirish24/surgical-vla-cache),
#        BENCHES ("libero libero_plus vlabench").
set -uo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-5}
GPU_CAP=${GPU_CAP:-14}
RETRIES=${RETRIES:-3}
CACHE_REPO=${CACHE_REPO:-Shirish24/surgical-vla-cache}
BENCHES=${BENCHES:-"libero libero_plus vlabench"}
export HF_XET_HIGH_PERFORMANCE=1
LOG=results/plan
STATE=$LOG/state
mkdir -p "$STATE"
hf() { pixi run -e dev hf "$@"; }
note() { echo "== $(date '+%F %T') $*" | tee -a "$LOG/plan.log"; }
alert() { [ -f scripts/notify.sh ] && bash scripts/notify.sh "$*" || true; }  # optional local phone alerts (not in git)
done_mark() { touch "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }

models() {
    # perception (labelling) + phrase VLM + base policy + the SmolVLA checkpoints that score the traces;
    # revisions pinned where the launcher pins them (scripts/benchmark.py SMOLVLA_BASE, CACHE[*].ckpt)
    hf download facebook/sam3 > /dev/null || { echo "SAM 3 is gated: accept the licence at https://huggingface.co/facebook/sam3"; return 1; }
    hf download facebook/dinov2-small > /dev/null || return 1
    hf download depth-anything/Depth-Anything-V2-Small-hf > /dev/null || return 1
    hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct > /dev/null || return 1
    hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 > /dev/null || return 1
    hf download lerobot/smolvla_libero --revision a24f1f380461b665158d52acd140152c79e8b36d > /dev/null || return 1
    hf download lerobot/smolvla_vlabench --revision 7ef370813eb35823caa94088b50f5389a51e4c47 > /dev/null || return 1
    local ckpt=$HOME/.cache/surgicalvla/tapnextpp_ckpt.pt  # TAPNext++ point tracker (Apache-2.0), 2.5 GB
    if [ ! -s "$ckpt" ]; then
        mkdir -p "$(dirname "$ckpt")"
        curl -L --fail -C - -o "$ckpt.part" https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
            && mv "$ckpt.part" "$ckpt" || return 1
    fi
    done_mark models
    note "models ready"
}

pilot() {  # cost per episode on this machine, one process; the episodes stay in the cache
    for b in $BENCHES; do
        python3 scripts/benchmark.py cache --benchmark "$b" --suite all --limit 3 --gpu-mem-gb "$GPU_CAP" 2>&1 | tee "$LOG/pilot_$b.log"
    done
    grep -h '"gpu_hours"' results/pseudolabels/*_all_cost.json || true
}

cache_one() {  # cache_one <benchmark>: labels then traces, retried until it exits cleanly
    local b=$1 attempt rc
    for attempt in $(seq 1 "$RETRIES"); do
        note "start cache $b (workers $WORKERS, attempt $attempt/$RETRIES)"
        rc=0
        python3 scripts/benchmark.py cache --benchmark "$b" --suite all --workers "$WORKERS" \
            --gpu-mem-gb "$GPU_CAP" --no-videos >> "$LOG/cache_${b}_all.log" 2>&1 || rc=$?
        note "end cache $b (exit $rc)"
        if [ "$rc" -eq 0 ]; then done_mark "cache_$b"; alert "✅ cache $b done (labels + CoT traces)"; return 0; fi
        sleep 60  # e.g. a transient OOM or Hub hiccup; finished episodes are kept
    done
    note "cache $b still failing after $RETRIES attempts: see $LOG/cache_${b}_all.log and $LOG/${b}_all_*_w*.log"
    alert "❌ cache $b failed $RETRIES times: see $LOG/cache_${b}_all.log"
    return 1
}

cache() {  # a failure does not stop the next benchmark; `resume` retries it later
    local rc=0
    for b in $BENCHES; do
        if [ "${FORCE:-0}" != 1 ] && is_done "cache_$b"; then note "skip cache $b (done)"; continue; fi
        cache_one "$b" || rc=1
    done
    return $rc
}

push() {  # whole cache tree, same layout FeatureCache uses on the Hub (<dataset-slug>/<cache-name>/...)
    hf upload-large-folder "$CACHE_REPO" pseudolabels --repo-type dataset --private \
        --exclude "*.lock" "*.tmp" "*/.lock" --num-workers 8 && done_mark push && note "pushed to $CACHE_REPO" \
        && alert "✅ caches pushed to $CACHE_REPO"
}

status() {  # what is cached, per benchmark, against the dataset's episode count
    pixi run -e dev python - "$BENCHES" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import benchmark as b
for name in sys.argv[1].split():
    repo = b.CACHE[name]["repo"]
    root = b.local_root(name)
    total = json.loads((Path(root) / "meta" / "info.json").read_text())["total_episodes"] if root else "?"
    caches = sorted(Path("pseudolabels").glob(f"{repo.replace('/', '__')}/*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not caches:
        print(f"{name:12s} no cache yet (dataset: {total} episodes, {root or 'not found locally'})"); continue
    m = json.loads(caches[-1].read_text())
    traced = {int(e) for s in m.get("cot_shards", []) for e in s["episodes"]}
    print(f"{name:12s} labelled {len(m['episodes']):>6} / {total}   traced {len(traced):>6}   cache {caches[-1].parent}")
PY
    ls "$STATE" 2>/dev/null | sed 's/^/done: /'
}

all() {
    is_done models || models || return 1
    cache || return 1
    is_done push || push
}

case "${1:-all}" in
    models) models ;;
    pilot) pilot ;;
    cache) FORCE=1 cache ;;
    push) push ;;
    all|resume) all ;;
    status) status ;;
    *) echo "usage: $0 {models|pilot|cache|push|all|resume|status}"; exit 2 ;;
esac
