#!/usr/bin/env bash
# Supervised SmolVLA baseline on RoboTwin 2.0 (unmodified SmolVLA, LeRobot's own trainer).
#
#   pixi run -e dev bash scripts/train_robotwin_baseline.sh            # full run
#   STEPS=200 BATCH=8 OUT=outputs/robotwin_smoke bash scripts/train_robotwin_baseline.sh   # smoke
#
# Data follows the RoboTwin 2.0 leaderboard protocol: 50 demo_clean demonstrations
# per task. The local lerobot/robotwin_unified copy stores each task as a contiguous
# block of 550 episodes in alphabetical task order, the first 50 clean and the next
# 500 domain-randomised (verified visually, see docs/baselines.md), so the clean
# demos are episodes 550*k + j for k < 50 tasks, j < 50.
#
# Deviation from the paper protocol (documented): one multi-task policy over all
# 50 tasks instead of 50 single-task policies (one GPU).
set -euo pipefail

ROOT=${ROBOTWIN_ROOT:-/media/protomate/Seagate/shirish/robotwin_data}
STEPS=${STEPS:-20000}   # LeRobot SmolVLA fine-tuning recipe (docs/source/smolvla.mdx)
BATCH=${BATCH:-64}
OUT=${OUT:-outputs/baselines/smolvla_robotwin_clean50}
SEED=${SEED:-1000}
TRACK=${TRACK:-true}  # Trackio, local SQLite only (no space_id -> nothing is uploaded)

EPISODES=$(python3 -c "print([550 * k + j for k in range(50) for j in range(50)])" | tr -d ' ')

exec lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.pretrained_revision=5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/robotwin_unified \
  --dataset.root="$ROOT" \
  --dataset.episodes="$EPISODES" \
  --rename_map='{"observation.images.cam_high": "observation.images.camera1", "observation.images.cam_left_wrist": "observation.images.camera2", "observation.images.cam_right_wrist": "observation.images.camera3"}' \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=100 \
  --seed="$SEED" \
  --num_workers=8 \
  --output_dir="$OUT" \
  --job_name=smolvla_robotwin_clean50 \
  --trackio.enable="$TRACK" \
  --trackio.project=surgical-vla \
  "$@"
