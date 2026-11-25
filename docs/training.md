# Training guide

How to train SurgicalVLA with supervised fine-tuning (SFT), online RL, and SFT and RL together,
on LIBERO, RoboTwin 2.0 and VLABench. Every key named here is explained in
[`docs/configs.md`](configs.md); evaluation is in [`docs/evaluation.md`](evaluation.md).

- [0. Before the first run](#0-before-the-first-run)
- [1. The three training regimes](#1-the-three-training-regimes)
- [2. SFT](#2-sft)
- [3. RL (ReinFlow-GRPO)](#3-rl-reinflow-grpo)
- [4. SFT + RL together](#4-sft--rl-together)
- [5. Per benchmark](#5-per-benchmark)
- [6. Logging: where the graphs are](#6-logging-where-the-graphs-are)
- [7. Resuming](#7-resuming)
- [8. Hugging Face Hub](#8-hugging-face-hub)
- [9. Feature cache and ablations](#9-feature-cache-and-ablations)
- [10. Integrity rules](#10-integrity-rules)
- [11. Troubleshooting](#11-troubleshooting)

## 0. Before the first run

| Check | Command | Why |
|---|---|---|
| environments installed | `pixi install -e dev && pixi install -e sim-eval` | `dev` trains; `sim-eval` runs the LIBERO simulator (RL, local eval) |
| Hub login | `pixi run -e dev hf auth whoami` → `Shirish24` | datasets, base weights, model upload |
| W&B login | `pixi run -e dev wandb login` | curves and images (`project.wandb_mode: online`) |
| Docker | `docker info` | RoboTwin / VLABench simulators and the canonical LIBERO evaluation image |
| GPU free | `nvidia-smi` | SFT on LIBERO uses ~10.4 GiB at batch 24; labelling needs 5–8 GiB more |
| data local (optional) | `hf download lerobot/libero --repo-type dataset` | with `run.streaming: false` the dataset is read from the local cache (`~/.cache/huggingface/lerobot/hub/datasets--lerobot--libero`), downloaded once on first use |

Every command goes through one launcher, `scripts/benchmark.py`. `--dry-run` prints the
underlying commands without running them. Arguments after `--` are passed to the trainer as
`section.key=value` overrides (`-- run.steps=20000 run.seed=1`).

## 1. The three training regimes

| Regime | What it is | Start from | Command | Output |
|---|---|---|---|---|
| **SFT** | flow-matching imitation on the benchmark's demonstrations (+ auxiliary losses when switched on) | `lerobot/smolvla_base` | `train sft --benchmark <b>` | `outputs/<b>/sft/checkpoints/step_N` |
| **RL** (`training.mode: sequential`) | ReinFlow-GRPO in the simulator, KL-anchored to the SFT model | the SFT model | `train rl --benchmark <b>` | `outputs/<b>/rl/final` |
| **SFT + RL** (`training.mode: joint` or `interleaved`) | RL updates plus a flow-matching loss on streamed demonstrations in the same run | the SFT model | `train rl --benchmark <b> -- training.mode=joint` | `outputs/<b>/rl/final` |

"SFT + RL" here always means *RL with demonstrations kept in the loss*, after an SFT stage.
Plain "SFT then RL" is the `sequential` mode. RL-only from `smolvla_base` (no SFT) is possible
(`-- checkpoint=lerobot/smolvla_base`) but is only an ablation arm: with no task-specific
policy, GRPO groups are nearly all failures and carry no signal.

```
smolvla_base ──SFT──► SFT model ──eval──► SFT number (required before any RL)
                          │
                          ├──RL, sequential────────────────► RL model ──eval──► RL number + forgetting
                          └──RL, joint / interleaved (+ FM)─► SFT+RL model ──eval──► same
```

One model per benchmark: each benchmark has its own configs, outputs, normaliser and Hub
repositories. A LIBERO model is never evaluated on RoboTwin, and so on.

## 2. SFT

### Command

```bash
python scripts/benchmark.py train sft --benchmark libero                      # SurgicalVLA
python scripts/benchmark.py train sft --benchmark libero --out results/libero_train
python scripts/benchmark.py train sft --benchmark libero --policy smolvla     # unmodified SmolVLA baseline
python scripts/benchmark.py train sft --benchmark libero --resume             # continue (section 7)
```

### What happens

1. `scripts/train.py` reads `configs/benchmarks/<b>/sft.yaml` and applies the overrides.
2. **Model:** `smolvla_moe` initialised from `lerobot/smolvla_base` (VLM, connector, state and
   action projections). The expert FFNs are up-cycled into the MoE (`smolvla_moe.moe_variant`).
   With every switch off, the model matches `smolvla_base` at step 0 (tested).
3. **Data:** `--repo_id` (`lerobot/libero`, `lerobot/robotwin_unified`,
   `lerobot/vlabench_unified`). With `run.streaming: false`, the whole dataset is read from the
   local Hugging Face cache. With `true`, frames come from the Hub on demand, which uses less
   disk but has no validation split.
4. **Split:** `dataset.val_fraction` of the episodes (default 1 %, by episode, seed
   `dataset.split_seed`) are held out for validation and saved to `<out>/split.json`.
   `dataset.episodes` restricts training to a list, e.g. one suite.
5. **Each step:**
   - `run.grad_accum` micro-batches of `run.batch_size` frames;
   - bf16 autocast, gradient clipping, AdamW with cosine decay (SmolVLA preset);
   - an EMA copy of the weights (`run.ema_decay`).
6. **Logs, validation and checkpoints:**
   - scalars every `project.log_every` steps;
   - validation action error every `project.val_every` steps (raw and EMA weights);
   - images every `logging.log_every_n_steps` (400);
   - a checkpoint every `project.save_every` steps, with weights, config, normaliser,
     optimizer and RNG state.
7. **Upload:** the final checkpoint is pushed to the Hub (section 8).

SFT never runs the simulator, so there is no success rate during SFT. Success is measured by
evaluation (`docs/evaluation.md`).

### Settings that matter

| Key | Default | Set it to | Notes |
|---|---|---|---|
| `run.steps` | 50,000 | 20k–50k | LIBERO: 2.7 s/step at batch 24 × accum 4 on an RTX A4500, so 50k ≈ 37 h |
| `run.batch_size` × `run.grad_accum` | 24 × 4 = 96 | keep the product ≥ 64 | the released SmolVLA recipes use 32–64 |
| `smolvla_moe.moe_variant` | `upcycled` | `matched` for the compute-fair comparison | `docs/architecture.md` |
| `smolvla_moe.use_cot` | true | false unless `aux.cot_traces: true` | without traces there is no CoT loss, so the flag does nothing |
| `model.freeze_vision_encoder`, `model.train_expert_only` | true, true | keep | the published LIBERO checkpoint unfroze both (full fine-tune); we keep the frozen VLM until the VLM-change study |
| `model.n_action_steps` | 50 | keep 50 | inference only; the protocol horizon for every method |
| `dataset.episodes` | all | `configs/data/libero_<suite>_episodes.json` | one-suite runs (all four suite lists exist: 432 / 454 / 428 / 379 episodes) |
| `dataset.val_fraction` | 0.01 | 0.01–0.1 | offline action-error monitoring only |

Recommended first runs:

| Purpose | Command |
|---|---|
| main LIBERO model (all 4 suites) | `train sft --benchmark libero` |
| fast check that everything runs | `train sft --benchmark libero -- run.steps=50 project.val_every=25 hub.push=false` |
| one suite | `train sft --benchmark libero -- dataset.episodes=configs/data/libero_libero_spatial_episodes.json` |
| compute-matched MoE | `... -- smolvla_moe.moe_variant=matched` |

### Which data a LIBERO run sees

`lerobot/libero` contains 1,693 demonstrations of the 40 tasks of the four evaluation suites
(Spatial, Object, Goal, Long/libero_10; LIBERO-90 is not included). A default run trains on
99 % of them across all four suites. LIBERO measures success from 50 fixed starting layouts per
task that never appear in the demonstrations, so every demonstration may be used for training.
A held-out demonstration split measures offline action error, not success.

**Data-efficiency studies** (train on part of each task's demonstrations): there is no
per-task split option yet. Until there is, pass the training list explicitly with
`dataset.episodes` and set `dataset.val_fraction` to the held-out share; the held-out episodes
are drawn from that list. Success is still measured on the standard 50 layouts.

## 3. RL (ReinFlow-GRPO)

### When to start RL

Start RL only when all of these hold:

1. **SFT is finished and evaluated.** The final SFT checkpoint has its protocol number
   (`docs/evaluation.md`). That number is the reference for the RL gain and for forgetting.
2. **The SFT policy sometimes succeeds and sometimes fails on the RL tasks.** GRPO learns from
   differences inside a group of rollouts from the same start. A task the policy always fails
   (or always solves) gives zero advantage and no gradient (`rl/advantage_std` = 0,
   `rl/samples` = 0). Aim for tasks with roughly 10–90 % SFT success. Read this from RL
   validation at the start of the run (`rl/val_success/<suite>`, fresh layouts), **not** from the
   evaluation episodes.
3. **The simulator matches the evaluation.** Rollouts are built exactly as `lerobot-eval` builds
   them (`src/surgicalvla/rl/lerobot_sim.py`), and the config passes the benchmark contract check
   (state/action dims, cameras, normaliser, gripper dims). The script refuses to start otherwise.

### Command

```bash
python scripts/benchmark.py train rl --benchmark libero                                  # from outputs/libero/sft
python scripts/benchmark.py train rl --benchmark libero -- checkpoint=results/libero_train
python scripts/benchmark.py train rl --benchmark libero -- run.iters=2 run.val_every=1 hub.push=false   # smoke
python scripts/benchmark.py train rl --benchmark libero --resume
```

LIBERO rollouts run in the local `sim-eval` environment. For RoboTwin and VLABench the launcher
starts LeRobot's Docker simulator server, runs RL against it, and stops it.

### One iteration

1. Pick a task and one start layout.
   - LIBERO uses `env_kwargs.init_states: false`: layouts are sampled from each task's placement
     regions by seed, never the 50 evaluation layouts.
2. Roll out `rl.group_size` (8) episodes from that start.
   - Actions come from the SDE sampler: `rl.sde_steps` denoising steps with a learned noise
     scale (ReinFlow noise head).
   - Each episode executes `env.n_exec_steps` actions per chunk, for up to `env.max_steps` steps.
3. **Reward:** `rl.success_reward` for success. The optional pose terms (`position_weight`,
   `orientation_weight`) apply only when the simulator reports a 6-D object pose.
4. **Advantage:** z-score of the rewards within the group.
5. **Update:** one clipped-surrogate step (`rl.clip`) with per-denoising-step importance
   ratios, plus:
   - a k3 KL to the frozen SFT reference (`rl.kl_coef`);
   - an entropy bonus on the noise scale (`rl.entropy_coef`).

   Log-probs count only the executed chunk steps, and the CoT that was active during the
   rollout is replayed.
6. **Validation and saving:**
   - every `run.val_every` iterations, validate on `env.val_suites` ×
     `rl.val_tasks_per_suite` × `rl.val_episodes_per_task`, from fixed seeds that are not
     evaluation layouts;
   - every `run.save_every` iterations, save a checkpoint and `rl_state.pt`;
   - when the run completes, save and push `final/`.

The evaluated model is **always `final/`** after the fixed iteration budget. Validation only
monitors; it never selects a checkpoint.

### RL settings

| Key | Default | Range to try | Effect |
|---|---|---|---|
| `env.suite` | `libero_10` | any one LIBERO suite; RoboTwin/VLABench: task list | which tasks RL trains on (LIBERO: one suite per run) |
| `env.val_suites` | all 4 LIBERO suites | keep all 4 | forgetting shows up on the suites RL does not train on |
| `rl.group_size` | 8 | 4–16 | larger = better advantage estimate, more simulator time per update |
| `rl.kl_coef` | 0.01 | 0–0.05 | anchor to the SFT model; 0 = free drift |
| `rl.clip` | 0.2 | 0.1–0.3 | PPO-style ratio clip |
| `rl.sde_steps` | 5 | 5–10 | denoising steps during rollouts (evaluation always uses the deterministic ODE) |
| `rl.entropy_coef` | 0.001 | 0–0.01 | keeps the noise scale from collapsing |
| `run.lr` | 1e-6 | 5e-7–3e-6 | RL is far more sensitive than SFT |
| `run.iters` | 200 | fixed in advance | the budget; never stop early on validation |
| `smolvla_moe.reinflow_min_std` / `reinflow_max_std` | 0.01 / 0.15 | — | bounds of the learned noise (model config) |

### Metrics to watch

| Metric | Healthy | Problem |
|---|---|---|
| `rl/train_success_rate` | rises slowly | stuck at 0 or 1: no signal, choose other tasks |
| `rl/advantage_std` | ≈ 1 | 0 most iterations: groups all fail or all succeed |
| `rl/approx_kl` (to the SFT reference) | small, slowly rising | jumps: lower `run.lr` or raise `rl.kl_coef` |
| `rl/ratio_mean`, `rl/ratio_max` | near 1, max < ~2 | large: the update moved too far; lower `run.lr` |
| `rl/val_success/<suite>` | trained suite up, others flat | others falling: forgetting, try `joint` |
| `rl/grad_norm` | stable | spikes |
| `rl/val_rollout` images | sensible motions | jitter or freezing |

### RL experiments

**None of these has been run yet.** The RL code is covered by unit tests (ratios, KL, masks,
resume) and the contract checks. Each experiment is run at matched compute (the same number of
environment steps and gradient updates), with 3 seeds, and evaluated on the protocol.

| # | Question | Arms | Report |
|---|---|---|---|
| R1 | Does RL help after SFT? | SFT only vs `sequential` | success with CIs, per suite |
| R2 | Is joint SFT+RL better than SFT then RL? | `sequential`, `joint`, `interleaved` (+ RL-only from base as a floor) | success, sample efficiency (val success vs env steps), forgetting |
| R3 | How much demonstration weight? (joint) | λ constant 1.0; cosine 1 → 0.1; 0 | as R2 |
| R4 | KL anchor | `kl_coef` 0, 0.01, 0.05 | success, forgetting, KL curve |
| R5 | Group size | G = 4, 8, 16 at equal env steps | success vs env steps |
| R6 | Transfer of the gain | the R2 winner on RoboTwin and VLABench | per-benchmark success |

Order: R1 → R2 on one LIBERO suite (`libero_10`, the lowest SFT success). Go to R3–R5 only if
R1/R2 show a gain; go to R6 last. Nothing is scaled up without a signal in the pilot.

**Forgetting** is measured as the protocol success on the suites *not* trained by RL, before vs
after RL, together with `rl/val_success/<suite>` during training.

## 4. SFT + RL together

`training.mode` in `configs/benchmarks/<b>/rl.yaml`:

| Mode | Update every iteration | When to use |
|---|---|---|
| `sequential` | `L_GRPO + β·KL(π‖π_ref)` | default; the reference arm |
| `joint` | `L_GRPO + β·KL + λ_t · L_FM(demo batch)` | RL drifts away from the demonstrations (forgetting, odd motions) |
| `interleaved` | RL update, then `training.sft_steps` pure flow-matching updates every `training.interleave_every` iterations | the same, when the two gradients conflict (negative `joint/grad_cosine_rl_fm`) |

```bash
python scripts/benchmark.py train rl --benchmark libero -- training.mode=joint \
    training.lambda.start=1.0 training.lambda.end=0.1 training.lambda.kind=cosine training.lambda.steps=200
```

- **Demonstration batches** are streamed from the Hub (`demo:` section, the same dataset and
  camera rename as SFT; RoboTwin uses only the clean-50 episodes). Their stream position is
  saved in `rl_state.pt`, so a resumed run continues the stream.
- **λ_t** follows `training.lambda`: `constant | linear | cosine | exp`, from `start` to `end`
  over `steps` iterations.
- **Logged** per update: `joint/rl_loss`, `joint/fm_loss`, `joint/grad_norm_rl`,
  `joint/grad_norm_fm`, `joint/grad_norm_fm_weighted`, `joint/grad_cosine_rl_fm`,
  `joint/lambda`.
  - The two gradients are computed separately and then summed, which gives the same update as
    one joint backward (tested).
  - A persistently negative cosine means the demonstration term is working against the policy
    gradient.

A sensible default: `joint` with λ cosine from 1.0 to 0.1 over the whole run, `kl_coef` 0.01,
`group_size` 8, `lr` 1e-6. These values are a starting point; the R2/R3 pilot decides.

## 5. Per benchmark

| | LIBERO | RoboTwin 2.0 | VLABench | LIBERO-plus |
|---|---|---|---|---|
| Dataset | `lerobot/libero` (1,693 episodes, 273k frames, 40 tasks, 10 fps) | local copy of `robotwin_unified` (`dataset.root`), 50 clean episodes per task (`episode_blocks: {block: 550, take: 50}`) | `lerobot/vlabench_unified` (10,977 episodes, 3.1 M frames, 295 tasks, 10 fps) | local copy of `lerobot/libero_plus` (14,347 episodes, the 40 LIBERO tasks under perturbations, 20 fps); the labelled 480-episode subset (`dataset.episodes`) |
| Robot | Franka Panda, single arm | ALOHA, bimanual | Franka | Franka Panda, single arm |
| State / action | 8-D state, 7-D end-effector delta | 14-D joints / 14-D absolute joints | 7-D absolute end-effector pose | as LIBERO |
| Cameras → policy | `image`→camera1, `image2`→camera2 | `cam_high`, `cam_left_wrist`, `cam_right_wrist` → camera1–3 | `image`, `second_image`, `wrist_image` → camera1–3 | `front`→camera1, `wrist`→camera2 |
| `model.gripper_dims` | `[6]` | `[6, 13]` | `[6]` | `[6]` |
| SFT config | `configs/benchmarks/libero/sft.yaml` | `.../robotwin/sft.yaml` | `.../vlabench/sft.yaml` | `.../libero_plus/sft.yaml` (`latent_cot_horizon` 20 = 1 s) |
| RL config | `.../libero/rl.yaml` (local simulator, `init_states: false`) | `.../robotwin/rl.yaml` (Docker, all 50 tasks) | `.../vlabench/rl.yaml` (Docker, 21 primitive tasks) | `.../libero_plus/rl.yaml`: the unperturbed local LIBERO simulator, `init_states: false`; never a LIBERO-plus variant, since those are its evaluation set |
| RL horizon | `max_steps` 520 | 1,200 (`action_clip: null`, absolute joints) | as configured | as LIBERO |
| SmolVLA reference | `lerobot/smolvla_libero` | `lerobot/smolvla_robotwin` (trained on all episodes; not the protocol) | `lerobot/smolvla_vlabench` | `lerobot/smolvla_libero` (trained on `lerobot/libero`) |
| Baseline recipe (`--policy smolvla`) | full fine-tune, 25k steps, batch 32 | `scripts/train_robotwin_baseline.sh`, 20k steps, batch 64, expert only | 20k steps, batch 32 | the LIBERO recipe on `lerobot/libero_plus` |
| Feature cache | the four suites (`run_cache_plan.sh`) | 10-task subset, 500 episodes | 576-episode subset | 480-episode subset (`run_cache_plan.sh`) |
| Known limits | protocol baseline ~66 % on Spatial (`docs/baselines.md` §3) | Hard setting unavailable; one multi-task policy instead of 50 | — | whether the training perturbations overlap the evaluation variants is not yet checked |

Data paths: LIBERO and VLABench training read from the Hugging Face cache. RoboTwin reads
`/media/protomate/Seagate/shirish/robotwin_data/` (set in its config and in the launcher), and
LIBERO-plus `/media/protomate/Seagate/shirish/lerobot_data/libero_plus` (`dataset.root`). The RL
demonstration stream (`demo:`) reads from the Hub.

## 6. Logging: where the graphs are

| Where | What | How to open |
|---|---|---|
| W&B, project `surgical-vla` (`project.entity`) | every scalar; camera frames, action-chunk plots, MoE routing heatmap and pseudo-label overlays every 400 steps; RL validation rollouts | https://wandb.ai → project `surgical-vla`; the run id is printed at start and saved in each checkpoint (`wandb_run_id.txt`) |
| TensorBoard | the same scalars and images | `pixi run -e dev tensorboard --logdir <out>/logs` |
| `<out>/metrics.csv` | every scalar, long format `step,key,value` | pandas / any spreadsheet |
| `<out>/figures/` | PNG copies of every logged image | file browser |
| Trackio (baselines only) | `lerobot-train` runs | `trackio show --project surgical-vla` |

SFT scalars:
- `loss`, `fm_loss`, `moe_aux_loss`;
- when their switch is on: `phase_loss`, `cot_loss`, `aux_feat_loss`, `aux_motion_loss`,
  `aux_track_loss`, `aux_rel_loss`, `latent_cot_loss`;
- `grad_norm`, `lr`, `gpu_mem_gb`;
- validation (`val/*` raw weights, `val_ema/*` EMA weights): action MAE per horizon bucket
  (`mae_h0_10`, `mae_h10_25`, `mae_h25_50`), arm and gripper MAE, jitter.

A loss that is missing from the logs means its component is switched off.

Also logged:
- **Every log step:**
  - `perf/step_s`, `perf/data_wait_s`, `perf/data_stall_frac`, `perf/cache_read_s`,
    `perf/samples_per_s`;
  - `grad_clip_rate`;
  - `moe/entropy/*` and `moe/load_max_over_min/*` (router entropy and per-expert load);
  - `cot_token_acc` and `cot_trace_len`.
- **Every `logging.grad_diag_every` steps:** `grad/norm/<loss>` and `grad/cos/<a>__<b>`, the
  gradient of each weighted loss and the cosine between every pair.
- **Every validation:**
  - `val/cot_gain/<phase>`: the action-likelihood gain of the true reasoning trace over a
    mismatched one, a faithfulness probe;
  - generated trace samples in `texts.jsonl` and W&B.
- **At start:** `run_info.json`, also in the W&B config. It holds the git commit and diff
  hash, pixi.lock hash, seeds, hardware, cache config hash, per-module parameter counts and
  measured FLOPs.

W&B runs are grouped by `project.group`: `<suite>/<arm>` for ablations.

**Pseudo-label overlay** (`pseudolabels` image; needs `aux.sidecar_root` and labelled episodes
in the batch):
- green fill and box: the SAM 3 target mask;
- red box: the SAM 3 end-effector;
- magenta line: gripper to target;
- yellow dots: TAPNext++ points, with cyan segments for their motion over the next
  `aux.track_horizon` labelled frames;
- text: noun phrase and SAM 3 scores, Depth Anything relative depths, episode, frame, phase,
  and the CoT trace.

The overlays are drawn from the stored labels. They are never a training target unless the
matching switch is on.

## 7. Resuming

```bash
python scripts/benchmark.py train sft --benchmark libero --out results/libero_train --resume
python scripts/benchmark.py train rl  --benchmark libero --resume
```

**SFT** continues from the newest `checkpoints/step_N`. It restores:
- weights, optimizer, LR schedule and EMA;
- RNG state;
- the data position. The shuffle order is a function of (seed, position), so no data is
  re-read.

On resume, `metrics.csv` and TensorBoard drop the values logged after step N, which leaves one
clean curve. The W&B run is resumed from the id saved in the checkpoint; W&B keeps its own
values for the steps after N.

For checkpoints written before the id was saved, add `-- project.wandb_id=<id>` (the id is in
`wandb/run-*-<id>`). A resume must use the same config and overrides as the original run.

**RL** resumes from `rl_state.pt`: policy, optimizer, RNG, demo-stream position and iteration.

## 8. Hugging Face Hub

Every trained model gets its own repository, **`Shirish24/surgical-vla-<model-name>`**, and all
of them are gathered in the Hub collection **`surgical-vla`**. Separate repositories are used
because `lerobot-eval --policy.path` and `from_pretrained` load a model only from a repository
root, not from a sub-folder.

`<model-name>` comes from the run (`src/surgicalvla/utils/hub.py`):

| Run | Repository |
|---|---|
| LIBERO SFT, defaults | `Shirish24/surgical-vla-libero-sft-upcycled-s0` |
| SFT, matched MoE + aux heads + latent CoT, Spatial only, seed 1 | `Shirish24/surgical-vla-libero-sft-matched-aux-lcot-spatial-s1` |
| LIBERO RL, joint, libero_10 | `Shirish24/surgical-vla-libero-rl-joint-libero10-s0` |
| SmolVLA baseline | `Shirish24/surgical-vla-libero-smolvla-baseline-s1000` |

`hub:` section of every config:
- `push: true`: SFT pushes the final checkpoint and RL pushes `final/`. Set `false` for smoke
  tests.
- `namespace: "Shirish24"`.
- `model_name: null`, or a template such as `"{benchmark}-sft-{variant}-mytest-s{seed}"`.
  Use a distinct name for runs whose settings are otherwise identical; otherwise they share a
  repository, and the later push becomes a new commit there.
- `private: true` creates the repository private. Make it public on the Hub when you release.
- `push_every_save: false`.

What is uploaded:
- weights, `config.json`, the normaliser (`policy_preprocessor*`, `policy_postprocessor*`);
- `train_config.yaml` (the exact config with overrides);
- a model card.

Optimizer and resume state stay local.

Manual or late upload, e.g. a run made before this existed:
```bash
pixi run -e dev python scripts/push_to_hub.py results/libero_train/checkpoints/step_50000 \
    --config configs/benchmarks/libero/sft.yaml --stage sft --model-name libero-sft-upcycled-allsuites-s0
```
A failed upload never stops training: it prints a warning with this retry command.

## 9. Feature cache and ablations

The final model (the SFT default) needs the feature cache for **every training episode**:
object tokens are inputs, and episodes without cached data get no object tokens and no
auxiliary targets. Build the cache for the suites you train on first.

```bash
pixi run -e dev cache --suite libero_spatial [--gpu-mem-gb 8] [--upload-repo Shirish24/surgical-vla-cache]
pixi run -e dev ablate --suite libero_spatial --dry-run          # the ladder, 3 seeds each, then evaluation
pixi run -e dev ablate --suite libero_object                     # other suites: smolvla vs full
pixi run -e dev compare results/ablations/libero_spatial/smolvla results/ablations/libero_spatial/vision
pixi run -e dev figures                                          # results.json -> PDFs, LaTeX, docs/baselines.md

bash scripts/run_libero_plan.sh    # everything in order, resumable: caches, Spatial ladder, other suites, figures
bash scripts/run_cache_plan.sh     # caches only, every benchmark: RoboTwin, LIBERO x4, VLABench, LIBERO-plus
```

Caches per benchmark (`scripts/benchmark.py` `CACHE`). Every benchmark labels its **frontal
camera**, and each training config pins that camera in `aux.expect`:

| Benchmark | Dataset (where) | Labelled camera | Episodes | Strides (targets / inputs) |
|---|---|---|---|---|
| LIBERO | `lerobot/libero` (streamed) | `image` (agentview) | per suite, `configs/data/libero_<suite>_episodes.json` | 4 / 1 (10 fps) |
| RoboTwin | local copy, `ROBOTWIN_ROOT` | `cam_high` (head) | 10 tasks × 50 clean, `robotwin_subset10_episodes.json` | 12 / 3 (30 fps) |
| VLABench | `lerobot/vlabench_unified`, local copy under `DATA_ROOT` | `image` (front) | 2 per instruction (576), `vlabench_subset_episodes.json` | 4 / 1 (10 fps) |
| LIBERO-plus | `lerobot/libero_plus`, local copy under `DATA_ROOT` | `front` | 12 per task (480), `libero_plus_subset_episodes.json` | 8 / 2 (20 fps) |

The VLABench and LIBERO-plus subsets are fixed by rule before any result
(`scripts/subset_episodes.py`: a fixed number per task string, drawn with seed 0).

```bash
pixi run -e dev cache --benchmark vlabench --suite subset
pixi run -e dev cache --benchmark libero_plus --suite subset
```

- **Cache.**
  - Policy inputs are computed on every frame and targets on every 4th; traces are written
    afterwards.
  - Cost: 57 s per episode, about 6.9 GPU-h for LIBERO-Spatial and 27 GPU-h for all four
    suites (`docs/pseudolabels.md`).
  - The run is resumable, and the manifest pins every model hash and setting.
  - A training run refuses a mismatched cache.
- **Arms** (`configs/ablations/arms.yaml`):
  - `smolvla`, `oracle_vision`, `aux_only`, `objtok_only`, `frozen_control`, `vision`,
    `cot_explicit`, `cot_latent`, `cot_curriculum`, `full`, `full_moe_matched`, and the LoRA arms
    `vision_lora`, `vision_lora_vis`, `full_lora` (`docs/vlm_changes.md`);
  - all under one protocol (data, steps, batch, LR schedule, seeds, evaluation);
  - each arm sets every switch.
  - The predicted effect of each component is recorded in `docs/results_analysis.md` before
    any run.
- **Gates**, in order, stopping at the first that fails:
  1. harness validation;
  2. cache built and validated on one suite;
  3. own SmolVLA baselines on all four suites;
  4. the Spatial ladder, with the oracle arm first (components whose oracle fails are
     dropped);
  5. the full model per suite, then LIBERO-Plus, then RoboTwin.

## 10. Integrity rules

- **Evaluation layouts:** never train or tune on them. LIBERO RL uses fresh layouts
  (`init_states: false`), and validation uses separate fixed seeds.
- **Checkpoints:** fix the iteration and step budget before the run and evaluate the last
  checkpoint. Never pick a checkpoint by evaluation success.
- **Seeds:** 3 per arm, all reported, never selected.
- **Pseudo-labels:** come only from frozen models. Simulator ground truth is used only to
  validate them and for the oracle gate.
- **Negative results:** reported (e.g. the explicit-CoT gain was negative before bootstrapping,
  `docs/pseudolabels.md`).

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Instruction 'train' corresponds to no data` | partial dataset cache; download the full dataset or pass `--root` |
| CUDA out of memory with caching running | lower `cache --gpu-mem-gb`, or `run.batch_size` with a larger `run.grad_accum` |
| `feature cache does not match the training config` | the message names the field; rebuild the cache or change `aux.expect` deliberately |
| resume starts at step 0 | `--resume` looks under `run.ckpt_dir`; pass the same `--out` as the original run |
| a new W&B run on resume | old checkpoint without a run id: add `-- project.wandb_id=<id>` |
| RL refuses to start (`BenchmarkMismatch`) | config and checkpoint disagree on dims, cameras or normaliser; use that benchmark's own SFT model |
| RL `rl/samples` = 0 for many iterations | all-fail or all-success groups; choose tasks with mixed SFT success |
| `push ... failed` warning | network or token; rerun `scripts/push_to_hub.py` later |
| no `pseudolabels` image | `aux.sidecar_root` unset, or no labelled episode in the logged batch |
