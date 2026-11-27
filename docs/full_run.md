# Full run on one A100 80 GB (12 CPUs)

Data preparation and SFT of the final model (object tokens, auxiliary vision heads, latent and
explicit CoT, phase head, MoE expert) on **every episode** of LIBERO (4 suites, 1,693 episodes),
VLABench (10,977) and LIBERO-plus (14,347). No subsets. One model per benchmark.

## 0. Machine setup (once)

```bash
cd SurgicalVLA
pixi install -e dev && pixi install -e sim-eval
pixi run -e dev hf auth login          # account Shirish24 (token with write access)
pixi run -e dev wandb login
nvidia-smi && nproc                    # A100 80 GB, 12 CPUs
```

Accept the SAM 3 licence at <https://huggingface.co/facebook/sam3> with the same account,
because the weights are gated.

Phone alerts (optional): `scripts/notify.sh` reports each finished or failed cache stage, Hub
push and SFT run through a Telegram bot (or WhatsApp via CallMeBot). Setup per machine:
1. In Telegram, message `@BotFather`, send `/newbot` and copy the token.
2. Run `mkdir -p ~/.config/surgicalvla-notify && echo 'TELEGRAM_BOT_TOKEN=<token>' >> ~/.config/surgicalvla-notify/env && chmod 600 ~/.config/surgicalvla-notify/env`.
3. Open your bot in Telegram and press Start.
4. Run `bash scripts/notify.sh --setup-telegram`, which saves your chat id and sends a test message.

Wrap any other command with `bash scripts/notify.sh --run "<label>" <command...>`.

## 1. Where the data lives

```
SurgicalVLA/
  data/                      git-ignored; the launcher's default SURGICALVLA_DATA_ROOT
    libero/                  lerobot/libero            (1.9 GB)
    libero_plus/             lerobot/libero_plus       (15.8 GB)
    vla_bench/               lerobot/vlabench_unified  (23 GB; vlabench_unified/ also accepted)
  pseudolabels/              feature caches, one per dataset (written by step 3, pushed in step 4)
  outputs/<benchmark>/sft    training runs
```

Each directory must be a full `hf download <repo> --repo-type dataset --local-dir data/<name>`,
with `meta/info.json` present. The launcher uses a local copy when it finds one and otherwise
falls back to the Hugging Face cache. Check it with
`python3 scripts/benchmark.py train sft --benchmark vlabench --dry-run`: the command must show
`dataset.root=.../data/vla_bench`, not `dataset.root=null`. Elsewhere on disk: set
`SURGICALVLA_DATA_ROOT=/path`, or `LIBERO_ROOT`, `LIBERO_PLUS_ROOT`, `VLABENCH_ROOT` for one
dataset each. Keep `HF_HOME` (model weights, about 10 GB) on a disk with space.

## 2. Model weights and a pilot

```bash
bash scripts/prepare_data.sh models    # SAM 3, DINOv2, Depth Anything V2, SmolVLM2, smolvla_{base,libero,vlabench}, TAPNext++
bash scripts/prepare_data.sh pilot     # 3 episodes per benchmark, one process each
```

The pilot writes `results/pseudolabels/<benchmark>_all_cost.json` (seconds per episode and peak
VRAM on this GPU). Its episodes stay in the cache. Use it to size the full run:

| | episodes | frames/ep | estimate on an RTX A4500 (1 process) |
|---|---|---|---|
| LIBERO | 1,693 | 161 | ≈ 27 GPU-h (measured 57 s/ep) |
| LIBERO-plus | 14,347 | 156 at 20 fps (strides 8/2) | ≈ 110 GPU-h (scaled from LIBERO) |
| VLABench | 10,977 | 284 | ≈ 300 GPU-h (scaled from LIBERO) |

The A100 and parallel processes cut this several-fold. Divide by the pilot's speed-up and by
the worker count, then check the real rate in the logs after an hour.

## 3. Feature caches, in parallel

```bash
nohup bash scripts/prepare_data.sh cache > results/plan/prepare.log 2>&1 &
tail -f results/plan/plan.log                     # one line per benchmark start/end
tail -f results/plan/libero_all_labels_w0.log     # one JSON line per labelled episode
```

- **Each benchmark in turn** (`BENCHES`, default `libero libero_plus vlabench`):
  - `WORKERS` (default 5) labelling processes split the episodes (worker *i* takes every 5th,
    starting at *i*) and all write the same cache;
  - every manifest update takes a file lock, so the workers never collide;
  - each process is capped at `GPU_CAP` = 14 GiB (peak 7–11 GiB) and gets 12/5 = 2 CPU threads;
  - then 5 processes score the bootstrap CoT traces.
- **What a cache holds per episode:**
  - policy inputs: SAM 3 detections, DINOv2 features, depth and the box of the target and the
    end effector, every frame, past-only;
  - targets: SAM 3 video masks, DINOv2 target feature, relative depth and motion, TAPNext++
    point tracks, phase boundaries, noun phrase;
  - the CoT traces.
- **Resumable** after a crash, kill, reboot or dropped SSH session. Run
  `nohup bash scripts/prepare_data.sh resume > results/plan/prepare.log 2>&1 &` and it continues
  where it stopped:
  - finished steps are skipped (`results/plan/state/*.done`);
  - finished episodes are skipped;
  - a killed worker first writes its pending episodes;
  - traces are saved every 20 episodes;
  - a failing benchmark is retried 3 times (`RETRIES`).

  `bash scripts/prepare_data.sh status` shows labelled and traced episodes per benchmark.
- **Tuning:** `nvidia-smi` shows 5 processes at about 60–70 GiB. If GPU utilisation stays below
  ~80 %, try `WORKERS=6 GPU_CAP=12`. If you see out-of-memory errors, use `WORKERS=4`.
- **Disk:** about 20 GB of cache in total (`--max-gb` is 100 per cache). Overlay videos are off
  for the full run (`--no-videos`).

## 4. Push everything to the Hub

```bash
bash scripts/prepare_data.sh push      # -> hf.co/datasets/Shirish24/surgical-vla-cache (private, resumable)
```

This uploads `pseudolabels/` (every manifest, frame/episode shard and CoT shard) with
`hf upload-large-folder`. It is one resumable upload, not thousands of per-shard commits. On
another machine, fetch it into the same place before training:

```bash
pixi run -e dev hf download Shirish24/surgical-vla-cache --repo-type dataset --local-dir pseudolabels
```

A training run checks the manifest against its config (`aux.expect`) and refuses a cache built
with other settings.

## 5. SFT, three benchmarks at once

```bash
bash scripts/train_parallel.sh -- run.steps=50 project.val_every=25 hub.push=false   # smoke test, all three
nohup bash scripts/train_parallel.sh > results/plan/train.log 2>&1 &                 # the real runs (50k steps each)
bash scripts/train_parallel.sh                                                       # after any stop: same command
```

Resuming is always on:
- each run continues from its newest complete checkpoint (weights, optimizer, schedule, EMA,
  RNG, data position, W&B run);
- finished runs are skipped;
- a crashed run restarts from its checkpoint up to 3 times (`RETRIES`);
- checkpoints are written atomically every 1,000 steps (`SAVE_EVERY`), and only the newest 2
  are kept (`KEEP_LAST`; each is about 12 GB).

Re-run with the same trainer overrides as the first run. If a run was stopped after its final
checkpoint but before the Hub upload, push it with `scripts/push_to_hub.py` (`docs/training.md` §8).

- **Sharing the machine:**
  - the three runs share the GPU at about 11 GiB each, which leaves room;
  - each run is pinned to 4 of the 12 cores, with 3 data-loader workers (`run.num_workers`);
  - effective batch stays 24 × 4 = 96, the protocol of every config.
- **Is parallel actually faster?** Compare `perf/samples_per_s` and `perf/data_stall_frac`
  (W&B) between the smoke test above and a single run
  (`python3 scripts/benchmark.py train sft --benchmark libero -- run.steps=50 hub.push=false run.num_workers=10`).
  If three runs together give less than about twice one run's throughput, train them one after
  another with `run.num_workers=10` instead.
- **Output:** checkpoints go to `outputs/<benchmark>/sft/checkpoints/step_N`. The final model
  is pushed to `Shirish24/surgical-vla-<benchmark>-sft-upcycled-s0`.

## 6. After SFT

1. **Evaluate** each SFT model on its benchmark (`docs/evaluation.md`). This number is the
   reference for RL.
2. **CoT round 2** (optional, R&B-EnCoRe): re-score the traces with the trained policy, which
   keeps only the positive-gain traces:
   `pixi run -e dev python scripts/make_cot_traces.py --repo-id <dataset> --root data/<dir> --episodes 0: --sidecar pseudolabels --checkpoint outputs/<b>/sft/checkpoints/step_50000 --overwrite --shard i/N`.
   Run N of these at once, as in step 3, then push again.
3. **RL** per benchmark: `python3 scripts/benchmark.py train rl --benchmark <b>` (`docs/training.md` §3).
