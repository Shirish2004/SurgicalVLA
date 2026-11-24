# Limitations and experimental components

Read this before quoting a number or a claim. IDs refer to `docs/audit.md`.

## Experimental: not yet validated by an ablation

| Component | Status |
|---|---|
| Auxiliary vision heads (`aux_vision`) | Trained from label-free pseudo-labels; mask/centroid/motion targets validated against simulator ground truth (`docs/pseudolabel_quality.md`); gripper-relative displacement is weak (low weight). No ablation yet. |
| Latent CoT (`latent_cot`) | Trains end to end (EMA future-latent target). No ablation yet. |
| Explicit CoT (`use_cot` + traces) | Traces come from pseudo-labels. Scored with a policy never trained on traces, every candidate *raises* the flow-matching loss, so round 1 must be unfiltered (bootstrap); filtering is untested until round 2. |
| Explicit-to-latent curriculum | Implemented; no ablation yet. |
| Phase head | Pseudo phase segments (gripper events + speed minima); grasp boundary is the gripper-close event, 0.5-0.75 s before lift. |
| Object tokens (`object_input: objects`) | Past-only per-frame perception (SAM 3 image mode + DINOv2 + Depth Anything), 3 phrase candidates + end effector; on in the final model. Agreement of the inputs with simulator ground truth not measured yet; no ablation yet. Adds 236 ms per action chunk at inference. |
| 6-D pose tokens, pose auxiliary loss | No labels; off. 6-D pose is reserved for real-robot instruments with scanned meshes. |
| MoE (`upcycled`, `matched`) | Compute- and parameter-matched baselines exist; no ablation yet. MoE inference is ~2x slower than dense at equal FLOPs (dispatch loop). |

## Fixed in the audit (with regression tests)

A1 half the action expert skipped · A2 prefix embedding parity · A3 flow-time schedule ·
A4 untrained CoT injected at inference · A5 reward shaping sign · A6 sequence-level ratio ·
A7 GRPO groups across different start states · A8 stale CoT in replay · A9 log-probs over
unexecuted actions · A10 cross-benchmark RL configs · A12 RL resume (weights, optimizer, RNG,
demo stream; atomic) · A16 CoT attention mismatch · A17 no-op `full_vlm_layers_for_cot`
removed · LIBERO adapter 7-D state (now 8-D) · checkpoints saved without their normaliser.

## Open

| ID | Limitation | Plan |
|---|---|---|
| A13/A14 | Up-cycled experts start identical and cost 3× dense; `matched` (distinct row-slices, 1× compute) and dense baselines now exist | Run the ablation |
| — | Gripper-relative displacement pseudo-label is weak (cosine ~0.7) | Fingertip point tracks |
| — | LIBERO protocol baseline is ~66 % on Spatial with the 50-step horizon (published ~90 %) | Documented in docs/baselines.md |
| — | RoboTwin **Hard** setting unavailable (LeRobot's integration hard-codes demo_clean) | Patch the integration or run RoboTwin's own evaluator |
| — | RoboTwin baseline is one multi-task policy, not 50 single-task policies as in the paper | Documented deviation (single GPU) |
| — | Ablation arms (`configs/ablations/arms.yaml`) and joint vs sequential SFT+RL not run | `pixi run -e dev ablate`; predictions in `docs/results_analysis.md` |
| — | **The VLM is frozen**: auxiliary, CoT and phase losses cannot change the visual features. The auxiliary heads read the action expert (`aux_source: expert`); with `prefix` they would reach only ~10 M trainable prefix parameters | VLM-change study (LoRA), ranked shortlist for the user to pick |
| — | The explicit CoT head reads layer 16 of a 32-layer VLM through a copy of the final LM head: its initial loss is ~22 nats (above uniform, ~10.8) | Trains from there; reported per run (`cot_loss`, `cot_token_acc`) |
| — | **Harness not validated**: `smolvla_libero` gives 66 % on Spatial in our protocol vs 90 % published | No SOTA claims until resolved (docs/baselines.md §3) |
| — | Oracle arm needs a ground-truth cache built by replaying the LIBERO demonstrations in the simulator and matching them to `lerobot/libero` episodes | Build before the Spatial ladder (gate 1) |
| — | LeRobot's eval output has no episode length; `scripts/lerobot_eval.py` records it, and failures are categorised only as timeouts | Finer failure categories need simulator state at episode end |
| A20 | SFT validation split is per episode; tasks overlap train/val | Task-level held-out split for generalisation claims |
