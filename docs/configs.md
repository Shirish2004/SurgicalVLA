# Config reference

Every benchmark has two configs, `configs/benchmarks/<benchmark>/sft.yaml` and `rl.yaml`.
Any key can be overridden on the command line after `--` as `section.key=value`, for example:

```bash
python scripts/benchmark.py train sft --benchmark libero -- run.steps=20000 smolvla_moe.moe_variant=matched hub.push=false
```

Defaults below are the LIBERO values; RoboTwin and VLABench differ only where noted. "Change?"
says whether a user normally touches the key.

## SFT config (`sft.yaml`)

### Top level

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `benchmark` | `libero` | benchmark name; used in the Hub model name | no |

### `project:` logging and run identity

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `wandb_project` | `surgical-vla` | W&B project | no |
| `wandb_mode` | `online` | `online`, `offline` (sync later with `wandb sync`) or `disabled` | `disabled` for smoke tests |
| `entity` | `building-text` | W&B team or user; `null` = your default account | if your W&B entity differs |
| `run_name`, `notes`, `tags` | — | W&B labels | optional |
| `wandb_id` | unset | resume a W&B run whose checkpoint lacks the run id | only for old checkpoints |
| `log_every` | 20 | scalar logging interval (steps) | rarely |
| `val_every` | 200 | offline validation interval (action error on held-out demos) | rarely |
| `save_every` | 2000 | checkpoint interval | larger for long runs to save disk |

### `run:` optimisation

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `steps` | 50000 | total optimiser steps (the budget; fix it before the run) | yes, per experiment |
| `seed` | 0 | torch / numpy / random seed and the data order | 0, 1, 2 for the 3-seed protocol |
| `batch_size` | 24 | frames per micro-batch | lower if out of memory |
| `grad_accum` | 4 | micro-batches per step (effective batch = batch_size × grad_accum) | raise when lowering `batch_size` |
| `device` | `cuda` | device | no |
| `ckpt_dir` | `outputs/<b>/sft` | output directory (also set by `--out`) | per run |
| `streaming` | false | false = read the local dataset copy (has a validation split); true = stream from the Hub | false |
| `streaming_buffer_size`, `streaming_seed` | 1000, 42 | shuffle buffer when streaming | no |
| `amp` | true | bf16 autocast | no |
| `ema_decay` | 0.999 | EMA of weights, validated as `val_ema/*` | no |
| `val_samples` | 640 | validation frames per check | no |
| `resume` | unset | `true` or a path; prefer `--resume` | via the launcher |
| `pretrained` | unset | start from a checkpoint other than `smolvla_base` | ablations only |

### `dataset:`

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `root` | null (RoboTwin: local path) | local dataset directory | only for local copies |
| `camera_map` | per benchmark | dataset camera key → policy camera (`camera1..3`) | no; must match `rl.yaml` and evaluation |
| `episodes` | all | list or JSON file of training episodes (`configs/data/libero_<suite>_episodes.json`) | one-suite and data-efficiency runs |
| `episode_blocks` | RoboTwin: `{block: 550, take: 50}` | first `take` episodes of every `block` (the clean-50 protocol) | no |
| `val_fraction` | 0.01 | share of episodes held out for offline validation | 0.01–0.1 |
| `split_seed` | 0 | seed of the train/val split | no |

### `model:` the SmolVLA backbone

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `policy_type` | `smolvla_moe` | SurgicalVLA; `surgical` is the legacy skill-routed model (`docs/legacy_skill_routed_moe.md`) | no |
| `vlm_model_name` | SmolVLM2-500M-Video-Instruct | VLM backbone | no (VLM changes are a separate, planned study) |
| `chunk_size` | 50 | actions predicted per chunk | no |
| `n_action_steps` | 50 | actions executed per chunk at inference (protocol horizon) | no |
| `n_obs_steps` | 1 | observation history | no |
| `freeze_vision_encoder`, `train_expert_only` | true, true | which parts train | VLM-change study only |
| `num_vlm_layers` | 16 | truncated VLM depth | no |
| `expert_width_multiplier` | 0.75 | action-expert width relative to the VLM | no |
| `gripper_dims` | `[6]` (RoboTwin `[6, 13]`) | action dims that command grippers (phases, pseudo-labels) | no |
| `scheduler_decay_steps` | 50000 | cosine decay length; keep equal to `run.steps` | with `run.steps` |
| `attention_mode`, `load_vlm_weights`, `add_image_special_tokens`, `adapt_to_pi_aloha` | — | SmolVLA settings | no |

### `smolvla_moe:` SurgicalVLA components

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `moe_variant` | `upcycled` | action-expert FFN: `upcycled` (3× dense compute), `matched` (1× compute), `dense`, `dense_param_matched` | ablations |
| `num_experts`, `num_shared_experts`, `moe_top_k` | 4, 1, 2 | MoE shape | no |
| `moe_aux_loss_weight` | 0.01 | Switch load-balancing loss | no |
| `aux_vision` | false | auxiliary vision heads trained on pseudo-labels (needs `aux.sidecar_root`) | pilot arm `vision_aux` |
| `latent_cot` | false | latent reasoning tokens predicting a future visual embedding | pilot arm `cot_latent` |
| `use_cot` | true | explicit CoT text in the prefix; trains only with `aux.cot_traces: true` | set false when there are no traces |
| `cot_max_tokens`, `cot_refresh_interval` | 48, 10 | CoT length and refresh interval (chunks) at inference | no |
| `cot_curriculum_steps` | 0 | >0: probability of explicit text decays to 0 over these steps (latent stays) | curriculum arm |
| `phase_loss_weight` | 0.1 | phase classification from pseudo phase segments | no |
| `pose_aux_loss_weight`, `num_object_tokens`, `num_objects`, `obj_tracker_mode` | — | object-centric 6-D tokens; off without labels | no |
| `expert_cross_attn_vlm_layers` | 0 | must equal `num_vlm_layers` (0 = all) | no |
| `use_reinflow` | false | ReinFlow noise head; switched on by the RL stage | no |

Further `SmolVLAMoEConfig` fields can be set here too: `aux_feat_weight`, `aux_motion_weight`,
`aux_track_weight` (0.1 each), `aux_rel_weight` (0.02), `latent_cot_tokens` (4),
`latent_cot_horizon` (10), `latent_cot_weight` (0.1), `latent_ema_decay` (0.99),
`reinflow_min_std` / `reinflow_max_std` (0.01 / 0.15).

The `moe:` section applies only to the legacy `surgical` policy and is ignored by `smolvla_moe`.

### `aux:` pseudo-label sidecar

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `sidecar_root` | `pseudolabels` (others: null) | where `scripts/make_pseudolabels.py` wrote labels. Used as targets only when a switch needs them; always used to draw overlays | set once labels exist |
| `track_horizon` | 5 | labelled frames ahead for the point-track target and the overlay trail | no |
| `cot_traces` | false | feed explicit reasoning traces (`scripts/make_cot_traces.py`) | pilot arm `cot_explicit` |

### `logging:` images

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `enable_visual_logging` | true | camera frames, action plots, MoE heatmap, pseudo-label overlays | no |
| `log_every_n_steps` | 400 | image interval | no |
| `log_dir` | `logs` | TensorBoard sub-directory | no |

### `hub:` model upload

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `push` | true | push the final checkpoint to `<namespace>/surgical-vla-<model-name>` | false for smoke tests |
| `namespace` | `Shirish24` | Hub user or organisation | no |
| `model_name` | null | null = derived from the run (`libero-sft-upcycled-s0`); or a template with `{benchmark}`, `{stage}`, `{variant}`, `{mode}`, `{seed}` | to tell apart runs with identical settings |
| `private` | true | create the repository private | false at release |
| `push_every_save` | false | also push intermediate checkpoints | rarely |
| `collection` | `surgical-vla` | Hub collection that gathers all models | no |

## RL config (`rl.yaml`)

### Top level

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `benchmark` | `libero` | selects the benchmark contract the run is checked against | no |
| `checkpoint` | `outputs/<b>/sft` | SFT model to start from (newest policy under a directory, or a Hub id) | point it at your SFT run |
| `checkpoint_revision` | null | Hub revision | when using a Hub id |
| `stats.path` | null | null = the checkpoint's own normaliser (required for a fair start) | no |
| `project.*` | — | as in SFT | — |

### `env:` simulator

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `backend` | `lerobot` | environments built exactly as `lerobot-eval` builds them | no |
| `endpoint` | null (RoboTwin/VLABench: `tcp/127.0.0.1:7448`) | Docker simulator server | no |
| `env_kwargs` | LIBERO: 360×360, `init_states: false` | `init_states: false` = fresh seeded layouts, never the 50 evaluation layouts | **never set true** |
| `suite` | `libero_10` (RoboTwin: 50 tasks; VLABench: 21 primitive) | tasks RL trains on | per experiment |
| `tasks`, `val_tasks` | null | task subsets | optional |
| `val_suites` | all 4 LIBERO suites (others: null = the training suite) | validation suites, which show forgetting | keep all |
| `camera_map` | per benchmark | policy camera → simulator camera | no |
| `action_slice`, `action_scale`, `action_clip`, `gripper_sign`, `env_gripper_dims` | per benchmark | action conversion; checked against the contract | no |
| `n_exec_steps` | 50 | actions executed per chunk in rollouts (the protocol horizon) | no |
| `max_steps`, `max_decisions` | 520, 11 (RoboTwin 1200, 24) | episode length cap | no |
| `frame_every` | 5 | frame stride of the logged validation rollouts | no |

### `training:` SFT + RL

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `mode` | `sequential` | `sequential` = RL only; `joint` = RL + λ·FM every update; `interleaved` = FM updates between RL updates | per experiment |
| `lambda` | `{start: 1.0, end: 0.1, kind: cosine, steps: 200}` | λ schedule for `joint` (`constant, linear, cosine, exp`) | R3 experiment |
| `interleave_every`, `sft_steps` | 1, 1 | `interleaved` cadence | interleaved only |

### `demo:` demonstrations for joint and interleaved modes

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `repo_id`, `revision` | the SFT dataset | streamed demonstrations | no |
| `batch_size`, `buffer_size` | 8, 1000 | demo batch and shuffle buffer | no |
| `episode_blocks` | RoboTwin clean-50 | same episode filter as SFT | no |
| `rename_map` | as SFT `camera_map` | dataset → policy cameras | no |

### `rl:` ReinFlow-GRPO

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `group_size` | 8 | rollouts per start state (GRPO group) | R5 experiment |
| `sde_steps` | 5 | denoising steps during rollouts | rarely |
| `clip` | 0.2 | ratio clip | rarely |
| `kl_coef` | 0.01 | k3 KL to the frozen SFT reference | R4 experiment |
| `entropy_coef` | 0.001 | entropy bonus on the noise scale | rarely |
| `adv_eps` | 1e-8 | numerical floor of the advantage std | no |
| `success_reward` | 1.0 | sparse success reward | no |
| `position_weight`, `orientation_weight`, `target_pose_6d` | 0.3, 0.3, null | dense pose terms, active only if the simulator reports object pose | no |
| `val_seed`, `val_tasks_per_suite`, `val_episodes_per_task`, `val_frames_per_task` | 1234, 10, 2, 4 | validation size and logged frames | lower for speed |

### `run:`

| Key | Default | What it is for | Change? |
|---|---|---|---|
| `iters` | 200 | RL iterations (the fixed budget) | per experiment |
| `lr` | 1e-6 | learning rate | carefully |
| `micro_batch` | 4 | chunks per backward pass | lower if out of memory |
| `val_every`, `save_every` | 10, 25 | validation and checkpoint intervals | no |
| `out_dir` | `outputs/<b>/rl` | output (`final/` = the trained model). Use a separate directory per mode | per run |
| `seed` | 0 | seed | 0, 1, 2 |
| `resume` | false | continue from `rl_state.pt` (or `--resume`) | via the launcher |

### `hub:`

As in SFT. RL models are named `<b>-rl-<mode>-<suite>-s<seed>`.
