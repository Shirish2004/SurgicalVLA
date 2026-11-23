# Training guide

## One model per benchmark

Each benchmark gets its own SurgicalVLA model, trained in two stages, with its own configs
and outputs:

| Benchmark | SFT config → output | RL config → output | Data / simulator |
|---|---|---|---|
| LIBERO | `configs/benchmarks/libero/sft.yaml` → `outputs/libero/sft` | `configs/benchmarks/libero/rl.yaml` → `outputs/libero/rl` | `lerobot/libero`; local sim (`pixi -e sim-eval`) |
| RoboTwin 2.0 | `configs/benchmarks/robotwin/sft.yaml` → `outputs/robotwin/sft` | `configs/benchmarks/robotwin/rl.yaml` → `outputs/robotwin/rl` | 50 demo_clean / task; Docker sim |
| VLABench | `configs/benchmarks/vlabench/sft.yaml` → `outputs/vlabench/sft` | `configs/benchmarks/vlabench/rl.yaml` → `outputs/vlabench/rl` | `lerobot/vlabench_unified`; Docker sim |

All SFT configs train `smolvla_moe` initialised from `smolvla_base`, and use the
`camera1..3` naming, so one rename map per benchmark serves training, RL and evaluation.
The RL stage starts from the newest policy under the SFT output and writes `final/` when it
completes. Checkpoints carry their normaliser (`policy_preprocessor.json`), which the RL
stage and `lerobot-eval` read.

```bash
python3 scripts/benchmark.py train sft --benchmark libero        # or robotwin / vlabench
python3 scripts/benchmark.py train rl  --benchmark libero
python3 scripts/benchmark.py pipeline                             # SFT -> RL on all three, then evaluate all
python3 scripts/benchmark.py pipeline --libero-plus --dry-run     # print every command
```

Baselines (unmodified SmolVLA, LeRobot's trainer and the released recipes):
`train sft --benchmark <b> --policy smolvla`.

Architecture switches (`moe_variant`, `aux_vision`, `latent_cot`, `cot_curriculum_steps`) and
the `aux:` sidecar section live in the SFT configs, all off by default (`docs/architecture.md`).
`run.seed` seeds torch / numpy / random; `dataset.episodes` restricts SFT to an episode list
(`scripts/libero_suite_episodes.py --suite libero_spatial` writes one).

## Label-free auxiliary supervision

```bash
pixi run -e dev labels --suite libero_spatial --limit 3      # pseudo-labels, videos, bootstrap traces
pixi run -e sim-eval python scripts/validate_pseudolabels.py --hdf5 <LIBERO demo file>   # quality vs sim GT
# then train with e.g.  smolvla_moe.aux_vision=true aux.sidecar_root=pseudolabels
```

Reasoning traces follow R&B-EnCoRe's bootstrap: round 1 trains on unfiltered traces
(`make_cot_traces.py --bootstrap`); round 2 re-scores them with the trace-trained policy and keeps
only those that lower its flow-matching loss on the demonstrated actions.

## Step-4 pilot

```bash
pixi run -e dev pilot --suite libero_spatial --steps 20000 --dry-run
```

Arms `baseline`, `vision_aux`, `cot_latent`, `cot_explicit`, `both` (and the MoE variants) are
trained with identical steps, batch and data (the suite's 432 demonstrations) for seeds 0, 1, 2,
then evaluated on that suite with the protocol settings. Gate order: oracle upper bound, this
pilot, transfer to another dataset; stop at the first gate that fails.

## RL stage (ReinFlow-GRPO)

`scripts/train_reinflow_grpo.py`. Per iteration: one task and one start state, `G` SDE
rollouts from it (GRPO groups share the start state), group z-scored rewards, then one
clipped-surrogate step with per-denoising-step importance ratios, a k3 KL to the frozen
SFT reference and an entropy bonus on the learned noise scale. Log-probs cover only the
chunk steps that were executed, and the CoT active at rollout time is replayed.

Rollouts use `src/surgicalvla/rl/lerobot_sim.py`, which builds the environment exactly as
`lerobot-eval` does, so RL sees the evaluation observations. LIBERO RL runs with
`init_states: false`: layouts are sampled from each task's placement regions by seed, so RL never
touches the 50 fixed init states per task that LIBERO evaluates on. Validation (every
`env.val_suites` suite × `rl.val_tasks_per_suite` × `rl.val_episodes_per_task`, fixed seeds) only
monitors training; the evaluated model is always `final/` after the fixed iteration budget. RoboTwin and VLABench run in
LeRobot's Docker images; the launcher starts and stops the server. Every config is checked
against its benchmark contract (`src/surgicalvla/rl/benchmarks.py`): state/action
dimensions, action semantics, cameras and normalisation must all match.

## Joint SFT + RL (`training.mode`)

| Mode | Update |
|---|---|
| `sequential` (default) | RL only, from the SFT checkpoint |
| `joint` | `L = L_GRPO + β·KL(π‖π_ref) + λ_t · L_FM(demo batch)` every update |
| `interleaved` | `sft_steps` pure flow-matching updates every `interleave_every` RL updates |

The demo batch is streamed from the Hub (`StreamingLeRobotDataset`, `demo:` section; RoboTwin
uses the clean-50 episodes) and its position is saved in `rl_state.pt`, so a resumed run
continues the stream where it stopped. `λ_t` follows `training.lambda`
(`constant | linear | cosine | exp`, from `start` to `end` over `steps`).

Logged per update (`joint/*`): `rl_loss`, `fm_loss`, `grad_norm_rl`, `grad_norm_fm`,
`grad_norm_fm_weighted`, `grad_cosine_rl_fm`, `lambda`. The two gradients are computed
separately and then summed, which leaves the update identical to a single joint backward
(tested); a negative cosine means the demo term is pulling against the policy gradient.

Resume (`run.resume: true`) restores the policy weights saved with the last checkpoint,
the optimizer, torch and numpy RNG, the demo-stream position and the iteration. The state
file is written atomically.

### Planned pilot (not run yet)

One LIBERO suite, three arms at matched compute (same number of environment steps and
gradient updates), 3 seeds each: RL-only from `smolvla_base`, SFT then RL (`sequential`),
and `joint`. Report success vs environment steps (sample efficiency), final success with
CIs, and forgetting (success on the SFT tasks before vs after RL). It runs only after the
LIBERO protocol gap (`docs/baselines.md` §3) is closed; nothing is scaled up without a
signal in the pilot.

## Logging

SurgicalVLA training (`scripts/train.py`, RL script) logs to W&B (`project.wandb_*`) and
local files; the LeRobot-trained SmolVLA baselines log with Trackio to the local database
(`trackio show --project surgical-vla`), nothing is uploaded.
