# Baselines and evaluation protocol

*Status (2026-11-24): protocol fixed (canonical LIBERO image, 50-step execution horizon, §3).
**Harness not validated**: the official checkpoint does not yet reproduce its published number
(§3). No arm of ours has been trained or evaluated yet; §0 lists every planned arm, and §6 is
generated from the results file.*

## 0. Baseline and ablation inventory

| # | Arm | Benchmark / suite | Source | Training data and steps | Seeds | Success (CI) | Status / notes |
|---|---|---|---|---|---|---|---|
| 1 | `smolvla_libero` harness check | LIBERO, 4 suites | official checkpoint (`lerobot/smolvla_libero@a24f1f38`) | as released: `lerobot/libero`, 25k steps, batch 32, full fine-tune | eval 1000, 2000, 3000 | Spatial **66.0 %** (Wilson 59.2–72.2; 10 tasks × 20 states × 1 seed) vs published 90 % | **not validated**: gap explained by the 50-step open-loop horizon (85.0 % at 10 steps, exploratory only; §3). Full 4 suites × 50 states × 3 seeds pending |
| 2 | `smolvla` (ours) | LIBERO, one model per suite | ours: SmolVLA architecture (dense expert) in our trainer | the suite's demos (432 / 454 / 428 / 379), 8k steps, batch 16 × 2 accumulation (effective 32) | train 0, 1, 2; eval 1000, 2000, 3000 | – | planned; the matched baseline every arm is compared with |
| 3 | other VLAs | LIBERO | published | their own (see §6) | – | §6 | reference only, never mixed into the matched comparison; no other model re-run in our harness yet (cost: one full protocol ≈ 6,000 episodes per model) |
| 4a | `oracle_vision` | LIBERO-Spatial | ours | as 2, simulator ground truth as targets and inputs | 0, 1, 2 | – | planned first on the ladder; needs the oracle cache (not built) |
| 4b | `aux_only`, `objtok_only`, `vision`, `frozen_control` | LIBERO-Spatial | ours | as 2 | 0, 1, 2 | – | planned; `frozen_control` = same encoders and token count, 2×2 grid, no objects |
| 4c | `cot_explicit`, `cot_latent`, `cot_curriculum` (vs `vision` = no CoT) | LIBERO-Spatial | ours | as 2 | 0, 1, 2 | – | planned |
| 4d | `full`, `full_moe_matched` | LIBERO-Spatial, then every suite | ours | as 2 | 0, 1, 2 | – | `full` = final model (sft.yaml defaults) |
| 4f | `vision_lora`, `vision_lora_vis`, `full_lora` | LIBERO-Spatial; `full_lora` on every suite | ours | as 2, LoRA r16 on the VLM decoder (+ vision) | 0, 1, 2 | – | `docs/vlm_changes.md` |
| 4e | joint vs sequential SFT+RL | LIBERO, best variant | ours | RL on fresh layouts (`init_states: false`), matched env steps | 0, 1, 2 | – | after 4a–4d |
| 5 | LIBERO-Plus zero-shot | each suite's model on its perturbed suite | ours (`smolvla`, `full`) | none (zero-shot) | eval 1000 | – | per-perturbation breakdown |
| 6 | RoboTwin SFT baseline | RoboTwin 2.0, 50 tasks × 100, Easy | ours | clean-50 demos per task (2,500 episodes) | 0 | – | no public SmolVLA checkpoint on the protocol data; Hard unavailable in LeRobot |

Arms 2 and 4 share one protocol (`configs/ablations/arms.yaml`): the same trainer, initialisation,
episodes, steps, batch, LR schedule, image resolution, evaluation harness and starting states;
each arm sets every switch, and a test checks that compared arms differ only in the factor under
test. Run: `pixi run -e dev ablate --suite libero_spatial`.

## 1. Software and hardware

| Item | Version |
|---|---|
| LeRobot | 0.7.1, upstream commit `8ec1a474da5f3f90a670d9343386995ec9f81264` (vendored at `lerobot/`, unmodified) |
| torch / transformers | 2.11.0 (cu130) / 5.22.2 |
| Local LIBERO stack (`pixi -e sim-eval`) | hf-libero 0.1.5, MuJoCo 3.8.1, robosuite 1.4.0 |
| Canonical LIBERO stack (LeRobot `docker/Dockerfile.sim`, `SIM=libero`) | hf-libero 0.1.4, MuJoCo 3.3.1, robosuite 1.4.0 |
| LIBERO-Plus image | fork `sylvestf/LIBERO-plus@4976dc3`, robosuite 1.4.1, MuJoCo 3.7.0, 6.4 GB assets |
| GPU / CPU | NVIDIA RTX A4500 20 GB (driver 580.196.07) / i7-12700K, 20 threads, 62 GB RAM |

Checkpoints (all pinned):

| Checkpoint | Revision | Trained on |
|---|---|---|
| `lerobot/smolvla_base` | `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` | SmolVLA pretraining |
| `lerobot/smolvla_libero` | `a24f1f380461b665158d52acd140152c79e8b36d` | `lerobot/libero` (dataset revision not recorded), full fine-tune, 25k steps, batch 32 |
| `lerobot/smolvla_vlabench` | `7ef370813eb35823caa94088b50f5389a51e4c47` | `lerobot/vlabench_unified`, 20k steps |
| `lerobot/smolvla_robotwin` | `eb55d3a74f6cd26c6388aecd116b37b63fac144d` | `pepijn223/robotwin_unified_v3` (all episodes, **not** the 50-clean protocol) |

## 2. Protocols

All evaluation goes through LeRobot's own integrations (`lerobot-eval`, or
`scripts/lerobot_eval.py` which only adds the `smolvla_moe` policy type), driven by
`scripts/benchmark.py`.

| Benchmark | Protocol | Command |
|---|---|---|
| LIBERO | LeRobot's canonical Docker image (default; `--local` is not the protocol); 4 suites (spatial, object, goal, 10) × 10 tasks × **all 50 benchmark init states** × 3 seeds (1000, 2000, 3000); the checkpoint's own `n_action_steps` (50); suite horizons 280/280/300/520; 10 settle steps; images rotated 180°; 8-D state | `python3 scripts/benchmark.py eval --benchmark libero --checkpoint lerobot/smolvla_libero --revision a24f1f38…` |
| LIBERO-Plus | every registered variant of the 4 perturbed suites (10,030), 1 episode each (upstream protocol), seed 1000 | `… eval --benchmark libero_plus …` |
| RoboTwin 2.0 | 50 tasks × 100 episodes, **Easy** (demo_clean). LeRobot's integration hard-codes `task_config/demo_clean.yml`, so Hard (demo_randomized) is not available yet | `… eval --benchmark robotwin …` |
| VLABench | 21 primitive + 22 composite tasks × 10 episodes × 3 seeds (VLABench paper protocol) | `… eval --benchmark vlabench …` |

Statistics (`scripts/eval_libero_baseline.py aggregate`): mean ± std over seeds; a 95 %
**cluster bootstrap** CI resampling (task, init state) clusters with all their seeds
(episodes sharing an init state are correlated, so a pooled Wilson interval is too
narrow); the pooled Wilson interval for comparison with `lerobot-eval`'s output.

`batch_size` divides `n_episodes` (10 | 50); vector environment `j` of `b` uses init states
`j, j + b, …` (LeRobot's LIBERO env), so every init state is used exactly once per seed and
episode `i` of a task runs init state `i`. Per-episode outcomes are saved with that id
(`episodes.jsonl`), so arms are compared pairwise on identical starting states
(`scripts/compare_arms.py`: paired difference, cluster-bootstrap CI, McNemar's exact test).

## 3. LIBERO reproduction: open gap

Published reference (SmolVLA paper, 0.45 B, as reported): Spatial 90, Object 96,
Goal 92, Long 71, average 87.3.

| Run (all `lerobot/smolvla_libero@a24f1f38`, LIBERO-Spatial) | Episodes | Success | Notes |
|---|---|---|---|
| local stack (MuJoCo 3.8.1), seed 1000 | 10 × 50 | **57.6 %** (Wilson 53.2–61.9) | task 5: 0/50 |
| local, tasks 4–5, render 256 instead of 360 | 2 × 20 | 25 % | resolution ruled out |
| local, tasks 4–5, `HuggingFaceVLA/smolvla_libero` | 2 × 20 | 57.5 % | second public checkpoint, same gap |
| **canonical Docker stack (MuJoCo 3.3.1, hf-libero 0.1.4)** | 10 × 20 | **66.0 %** (Wilson 59.2–72.2) | task 5: 0 % → 45 % |
| **canonical, `n_action_steps=10` (replan every 10)** | 10 × 20 | **85.0 %** (Wilson 79.4–89.3) | task 4: 65 %, task 5: 75 % |

Findings so far:
1. **Simulator version matters** (+8.4 pts, one task 0 → 45 %). LIBERO must be evaluated
   with LeRobot's canonical image (the launcher default); the local `sim-eval` env resolves newer
   MuJoCo / hf-libero than LeRobot pins. *Action:* the launcher's LIBERO default should
   become the canonical image once the remaining gap is understood.
2. Ruled out: observation resolution, config drift (the checkpoint config is explicit
   for every field), our adapter (these runs use `lerobot-eval` directly), a single bad
   checkpoint.
3. **Execution horizon matters most** (+19 pts). Executing the full 50-step chunk open-loop
   (the checkpoint's saved `n_action_steps`) gives 66 %; replanning every 10 steps gives 85 %,
   within about 5 pts of the published 90 % (and the CI includes 89 %).
4. **Decision (2026-11-23): the protocol keeps the 50-step execution horizon** (the
   checkpoint's saved `n_action_steps`) for every method, baselines and ours, so that nothing
   is selected on evaluation episodes. The 10-step result was found by looking at evaluation
   success and is reported only as an exploratory observation, never as the protocol number.
   Consequence: the protocol baseline on LIBERO-Spatial is ~66 % (canonical stack), below the
   published ~90 %; the paper states this and the reason.

The full 6,000-episode LIBERO run and LIBERO-Plus are **on hold** until this is resolved.

## 4. RoboTwin 2.0 data and baseline

The local `robotwin_unified` copy (27,500 episodes, 6.07 M frames, 30 fps) stores each of the
50 tasks as a contiguous block of 550 episodes in alphabetical task order: the first 50 are
`demo_clean`, the next 500 `demo_randomized` (verified visually on several blocks). The
leaderboard protocol trains on the 50 clean demos per task: episodes `550·k + j`, `j < 50`
(`dataset.episode_blocks: {block: 550, take: 50}`).

Baseline (unmodified SmolVLA, `scripts/train_robotwin_baseline.sh`): from `smolvla_base`,
2,500 clean episodes (550 K frames), multi-task over all 50 tasks (one GPU; the paper trains
one policy per task — documented deviation), LeRobot's SmolVLA recipe (batch 64, 20 k steps,
frozen VLM / expert-only), Trackio logging to the local database. Smoke-tested (30 steps,
loss 2.5 → 1.2). Queued after the LIBERO evaluation because it needs the whole GPU.

The repo's pre-audit `checkpoints*/` were trained on all 27,500 episodes (clean +
randomized) and are not comparable with leaderboard numbers.

## 5. VLABench

Reference checkpoint `lerobot/smolvla_vlabench` (trained on `lerobot/vlabench_unified`,
10,977 episodes, 295 tasks, 10 fps, 7-D absolute eef pose). Served from LeRobot's
canonical image. Not yet run.

## 6. Results

<!-- results:begin -->
*Generated by `scripts/build_baselines_doc.py` from `results/results.json`; do not edit by hand.*

No arm of ours has been evaluated yet.

Published numbers (reference only; other data, models and harnesses):

| Model | libero_spatial | libero_object | libero_goal | libero_10 | Citation | Date | Caveats |
|---|---|---|---|---|---|---|---|
| SmolVLA (0.45B) | 90.0 | 96.0 | 92.0 | 71.0 | Shukor et al., SmolVLA, arXiv:2506.01844 | 2025-06 | our harness reproduces 66.0% on Spatial with the 50-step horizon (docs/baselines.md §3) |
| OpenVLA (7B, fine-tuned) | 84.7 | 88.4 | 79.2 | 53.7 | Kim et al., OpenVLA, arXiv:2406.09246 | 2024-06 | different training data filtering; 256 px images |
| Octo (fine-tuned) | 78.9 | 85.7 | 84.6 | 51.1 | reported in Kim et al., OpenVLA, arXiv:2406.09246 | 2024-06 | baseline trained by the OpenVLA authors |
| Diffusion Policy (from scratch) | 78.3 | 92.5 | 68.3 | 50.5 | reported in Kim et al., OpenVLA, arXiv:2406.09246 | 2024-06 | baseline trained by the OpenVLA authors |
| pi0 (3.3B) | 96.8 | 98.8 | 95.8 | 85.2 | Black et al., pi0, arXiv:2410.24164; LIBERO numbers as reported in Kim et al., OpenVLA-OFT, arXiv:2502.19645 | 2025-02 | much larger model and pretraining corpus |
| OpenVLA-OFT (7B) | 97.6 | 98.4 | 97.9 | 94.5 | Kim et al., OpenVLA-OFT, arXiv:2502.19645 | 2025-02 | parallel decoding, action chunking, continuous actions, FiLM; 7B |
<!-- results:end -->
