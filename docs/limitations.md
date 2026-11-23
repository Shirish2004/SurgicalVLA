# Limitations and experimental components

Read this before quoting a number or a claim. IDs refer to `docs/audit.md`.

## Experimental: no training signal yet

These branches exist in the model and run, but **no dataset provides their labels**, so
they receive no supervision. They must not be claimed in the README or paper text until
the label pipeline (decision pending, audit A18) exists and an ablation shows an effect.

| Component | Status |
|---|---|
| Chain-of-thought (`use_cot`) | Decoder is correct (training attention pattern, A16) and gated off at inference until trained (A4). No CoT labels. With the frozen VLM it would be a linear probe on layer-16 features (A17). |
| Phase head / phase-conditioned routing | Without CoT, `e_phase` is a learned constant, so the phase loss learns only the label marginal (A15). |
| Object-centric 6-D tokens | Tracker runs in `passthrough` / mock mode in training; no object features in any dataset. |
| Pose auxiliary loss | No `target_relative_pose_6d` labels. |

What *does* train today: flow matching and the MoE load-balancing loss.

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
| — | **LIBERO baseline not reproduced** (66 % vs ~90 % on Spatial with the canonical stack) | `docs/baselines.md` §3 |
| A13 | MoE uses 3× the dense FFN FLOPs (1 shared + 2 of 4 full-width routed experts); a MoE-vs-dense claim is capacity-confounded | Compute-matched and param-matched dense ablations (task 3d) |
| A14 | Routed experts start identical; symmetry broken only by the aux loss | Random row-sliced up-cycling |
| A11 | RL validation is still a few episodes on a few tasks | Multi-suite validation (task 3e) |
| A18 | No CoT / phase / pose / object labels | Options to be proposed; your pick |
| — | RoboTwin **Hard** setting unavailable (LeRobot's integration hard-codes demo_clean) | Patch the integration or run RoboTwin's own evaluator |
| — | RoboTwin baseline is one multi-task policy, not 50 single-task policies as in the paper | Documented deviation (single GPU) |
| — | Joint SFT+RL pilot not run | After the LIBERO protocol is reproduced |
| A20 | SFT validation split is per episode; tasks overlap train/val | Task-level held-out split for generalisation claims |
