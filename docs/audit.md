# Phase 0 Audit — SurgicalVLA on LeRobot SmolVLA

*Date: 2026-11-23. Scope: `src/surgicalvla` (SmolVLA-MoE stack + RL), `scripts/`, `configs/`,
vendored `lerobot/` (v0.7.1, commit `8ec1a474d`), `SimpleVLA-RL/` (reference only).*

Every finding below is tagged **[verified]** (reproduced by a script or test) or **[read]**
(established by reading code paths end to end; no runtime repro yet). Fixed items cite their
regression test.

---

## 1. Architecture map

Two model stacks live side by side:

| Stack | Files | `config.type` | Status |
|---|---|---|---|
| Legacy skill-routed MoE | `modules/policy.py`, `moe.py`, `skills.py` | `surgicalvla` | Default in `configs/train.yaml` (`policy_type: surgical`); **all existing checkpoints** (`checkpoints*/step_*`) are this type, trained on RoboTwin (aloha, 14-D) |
| SmolVLA-MoE (AGENTS.md design) | `modules/configuration_smolvla_moe.py`, `modeling_smolvla_moe.py`, `moe_layers.py`, `object_encoder.py` | `smolvla_moe` | Target architecture; never trained to a checkpoint |

This audit focuses on the `smolvla_moe` stack, which is the one the paper will describe.

### Data flow (`SmolVLAMoEModel`)

```
images ─► SigLIP + connector ─► ×sqrt(d) ─┐
lang tokens ─► embed ×sqrt(d) ────────────┤ att=0 (bidirectional block)
object feats ─► ObjectCentric6DProjector ─┤ att=0   (K_obj tokens, slot-masked)
state ─► state_proj ──────────────────────┤ att=1
CoT ids ─► embed ×sqrt(d) ────────────────┘ att=1 each (causal)
        │
        ▼  SmolVLM2 text layers 0..15 (truncated from 32), KV cache
        ├─ cot_lm_head(prefix_out[CoT]) ─► CoT CE        (training only)
        └─ phase_proj(prefix_out[phase pos]) ─► e_phase  (or learned null_phase when no CoT)
                                                │
noisy actions x_τ + τ ─► action_time_mlp ─► action expert (16 layers, 720-d),
                cross-attn to prefix KV every other layer,
                FFN = PhaseConditionedMoEFFN(router on [h, e_phase, sg(obj_ctx)])
                                                │
                                   action_out_proj ─► v_θ(x_τ, τ)   (32-d, padded)
```

Flow convention: `NOISE_AT_ONE` (openpi/SmolVLA): `x_τ = τ·ε + (1−τ)·a`, target `u = ε − a`,
Euler `τ: 1 → 0`, `dt = −1/K`. **[verified]** consistent between `forward`, `sample_actions`,
`sample_actions_sde` and LeRobot's `make_flow_matching_inputs`.

RL stage: `sample_actions_sde` → Euler–Maruyama with learned σ (`ReinFlowNoiseHead`), transition
log-probs; `scripts/train_reinflow_grpo.py` does group rollouts in LIBERO through a socket
bridge (`scripts/libero_server.py`, py3.10 env) and a clipped surrogate update.

---

## 2. Findings

Severity: **C** = invalidates results, **H** = materially biases results, **M** = limits
claims or efficiency, **L** = hygiene.

### 2.1 Fixed in this phase (each has a regression test)

| ID | Sev | Finding | Evidence | Fix | Test |
|---|---|---|---|---|---|
| A1 | **C** | `expert_cross_attn_vlm_layers` defaulted to `num_vlm_layers//2 = 8`, applied by temporarily setting `vlm.num_vlm_layers = 8`. SmolVLA pairs expert layer *i* with VLM layer *i* (`get_model_layers`: `multiple_of = 8 // 16 = 0`), so **expert layers 8–15 never ran**. | [verified] On real `lerobot/smolvla_base` weights: MLP hooks fire on layers 0–7 only; actions differ from `SmolVLAPolicy` by **152 % relative L2** at init. The pretrained action expert was effectively discarded. | Default = full depth; any shallower value raises. `_expert_depth` removed. Stub tests that asserted the buggy half-depth were corrected. | `test_pretrained_parity.py::test_partial_expert_depth_is_rejected`, `::test_upcycled_policy_matches_smolvla_base_at_init` |
| A2 | M | Prefix embedding not bit-identical to upstream: image scale computed as `math.sqrt(d)` in fp32 (upstream casts `sqrt(960)` to bf16 = 31.0), image special tokens additionally ×sqrt(d), and fully-masked object slots changed sequence length (bf16 kernel accumulation). | [verified] Residual after A1: 7.2e-3 → 4.2e-3 → **<2e-3** rel. after each fix; same-prefix velocity and MLP are bit-exact (0.0). | Match upstream op order; drop object slots when no sample in the batch has a valid object. | same parity test (tolerance 2e-3) |
| A3 | H | Training flow time `τ ~ U(0.001, 0.999)`; SmolVLA pretraining used `0.999·Beta(1.5, 1) + 0.001` (mean 0.60 vs 0.50, more mass at the noise end). Fine-tuning from `smolvla_base` on a different τ-distribution re-weights the regression objective away from where the pretrained field was fit. | [read] `modeling_smolvla.py:602` vs ours | `SmolVLAMoEModel.sample_time` uses LeRobot's `sample_time_beta` with SmolVLA's constants. | `test_flow_time_matches_smolvla_beta_schedule` |
| A4 | **C** | With `use_cot=True` (default), inference decodes CoT from the never-trained `cot_lm_head` and appends it to the prefix (also switching `e_phase` from `null_phase` to a hidden state), even though **no dataset provides `cot_input_ids`** (A18), so every SFT step saw a CoT-free prefix. Train/test prefix mismatch. | [read] `prepare_prefix` refresh path | Persistent buffer `cot_trained_steps`, incremented on CoT-supervised training steps; CoT decoding only when > 0. | `test_cot_not_generated_until_cot_supervised` |
| A5 | H | `compute_handover_reward` returned `reward − shaped` on failure, where `shaped` *grows* as the object nears the target: among failures, getting closer was punished. Also no guarantee success > any failure. | [read] + test | `reward + shaped` in both cases; require `success_reward > w_p + w_r`. | `test_handover_reward_success_and_shaping`, `test_handover_reward_rejects_shaping_that_beats_success` |
| A6 | H | GRPO ratio was the product over all K SDE steps of densities over T×D = 50×7 entries: `log r` sums 1750 Gaussian terms, Var[log r] ∝ K·T·D. Small updates push nearly all samples outside the clip → vanishing gradient. Flow-GRPO uses per-denoising-step ratios; SimpleVLA-RL per-token. | [verified] synthetic: per-step clip fraction < sequence clip fraction at equal drift | `ratio_level="step"` (default): one ratio per transition, clipped terms averaged over (k, b). Same first-order gradient at r = 1 (up to 1/K scale). | `test_step_ratio_has_lower_variance_than_sequence_ratio`, `test_step_and_sequence_ratio_share_gradient_at_ratio_one` |
| A7 | **C** | GRPO groups did not share an initial state: each of the G rollouts drew its own LIBERO init state. The group mean then mixes easy/hard layouts, so `A_i` mostly measures layout difficulty, not action quality, and the variance-reduction argument for the group baseline fails. SimpleVLA-RL groups by `index` (same start). | [read] `rollout_group` | One `init_state` per group; only noise seed varies. | `test_grpo_group_shares_initial_state` |
| — | L | Two broken tests in `tests/test_smolvla_moe.py` (`moe` undefined; batch helper hard-coded to B=2). | [verified] | fixed | — |

Suite: **72 passed** (pre-audit: 62 passed, 2 failed). One commit per fix, on top of tag `pre-audit`.

### 2.2 Second round (fixed, with regression tests)

| ID | Sev | Finding | Fix |
|---|---|---|---|
| A10 | **C** | RL configs mixed RoboTwin checkpoints/stats with the LIBERO env | Benchmark contracts (`rl/benchmarks.py`) validated at start-up; one config per benchmark (`configs/benchmarks/*`) |
| A8 | H | RL replay used whichever CoT was cached | CoT recorded per chunk and replayed |
| A9 | H | Log-probs included unexecuted chunk steps | Executed-step mask for old/new/reference log-probs |
| A12 | M | No RL resume | Weights, optimizer, RNG, demo-stream position, atomic `rl_state.pt` |
| A16 | H | CoT decoding used a causal prefix, training a bidirectional one | Decoding through `vlm_with_expert` with the training mask; teacher-forcing test |
| A17 | M | `full_vlm_layers_for_cot` was a no-op | Removed (the linear-probe issue remains, see limitations) |
| A22 | **C** | LIBERO adapter sent a 7-D state tiled to 8-D (dim 6 ≈ 70σ out of distribution) | 8-D LeRobot state, image flip, settle steps; equality test against LeRobot's `LiberoEnv` |
| A23 | H | SFT checkpoints were saved without their normaliser (not evaluable by `lerobot-eval`) | Processors saved with every checkpoint |
| A24 | H | Local LIBERO stack (MuJoCo 3.8.1) differs from LeRobot's canonical one (3.3.1): −8 pts on Spatial, one task 45 % → 0 % | Evaluate LIBERO in the canonical image (`--server`); see `docs/baselines.md` |
| A25 | M | RL only possible on LIBERO (custom socket bridge) | Generic LeRobot rollout client: RL on LIBERO, RoboTwin, VLABench |

### 2.3 Open — to fix next (ordered by priority)

| ID | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| ~~A10~~ (fixed, §2.2) | **C** | `configs/reinflow_grpo.yaml` runs LIBERO RL on `checkpoints/checkpoints/step_2000` — a **legacy `surgicalvla`, RoboTwin aloha 14-D joint-space** checkpoint with RoboTwin stats — and slices actions `[0:7]` for LIBERO's 7-D *EEF-delta* space. Camera map sends `agentview` into `cam_right_wrist`. The policy's action semantics do not match the env; no result from this config is interpretable. | [read] config + checkpoint `config.json` | Require `policy action_dim == env action_dim` and matching feature semantics; RL must start from a LIBERO-SFT checkpoint (or `HuggingFaceVLA/smolvla_libero`). |
| A18 | **C** (for the paper's claims) | **No data pipeline produces** `cot_input_ids`, `subtask_phase_id`, `target_relative_pose_6d`, `obj_sam_feat`, `obj_pose_vec`. In SFT only `fm_loss` + `moe_aux_loss` are active; CoT, phase, pose and object branches receive **zero supervision**. Object tracker in `passthrough` mode returns a mock. | [read] grep over `src/`, `scripts/` | Build label generation (Phase 2): phase/CoT from LIBERO/RoboTwin task structure + gripper events; object pose from simulator state (train) and the tracker (eval), with the tracker-vs-GT gap reported. |
| ~~A8~~ (fixed, §2.2) | H | RL replay mismatch: `sde_log_probs` rebuilds the prefix using whatever `_cot_cache` is current, not the CoT used during the rollout. Ratios would compare different conditionals. Latent while A4 keeps CoT off. | [read] | Store CoT ids/positions in each rollout record and replay them. |
| ~~A9~~ (fixed, §2.2) | H | Log-probs (and thus ratios/KL) cover all 50 chunk steps but only `n_exec_steps=25` are executed. Unexecuted actions carry pure noise into the gradient (no effect on reward). | [read] `rollout.py`, `sde_log_probs` (no `action_is_pad`) | Mask log-probs to executed steps (pass `action_is_pad` from rollout). |
| A11 | H | RL validation = **1 episode × ≤4 tasks × 1 fixed seed** (`evaluate`). Success-rate SE at n=4 is ±25 pp; no CIs; validation tasks default to the training tasks. | [read] | Separate eval script: all tasks × 50 init states × ≥3 seeds; Wilson CIs. |
| A13 | H | MoE is not parameter- or FLOP-matched: each layer holds 1 shared + 4 routed *full-width* copies of the 4.42 M-param dense FFN (16 layers: 70.8 M → 354 M FFN params; model ≈ 450 M → ≈ 733 M). Active path = 3 FFNs/token = **3× dense FFN FLOPs**. A "MoE beats dense" claim would be confounded by capacity. | [verified] param count on `smolvla_base` | Fine-grained experts (width/3 each, sliced up-cycling) so active FLOPs = dense; dense baseline at matched total params for the other axis. |
| A14 | M | Up-cycled routed experts are identical copies and the router is zero-init → uniform probs, `topk` tie-break deterministically picks experts {0, 1}; zero-init `down_proj` means routed output and router gradient are both 0 at step 0. Symmetry is broken only by the aux loss; experts 2–3 start starved. | [read] | Random row-slicing of the dense FFN per expert (Komatsuzaki et al. 2023 / Qwen-MoE style) + small router noise. |
| A15 | M | Without CoT, `e_phase = null_phase` (a constant), so the 16 per-layer `phase_classifier`s learn only the label marginal; `phase_loss` is degenerate. Also 16 identical heads on one input is redundant. | [read] | Single phase head on a pooled prefix representation (works without CoT); CoT becomes an optional refinement. |
| ~~A16~~ (fixed, §2.2) | H | CoT train/inference attention mismatch: training puts CoT after a **bidirectional** image/text/object block (custom 2-D mask); `generate_cot` calls the HF text model with a 1-D mask, i.e. **fully causal** prefix (images cannot see the instruction). | [read] | Decode through `vlm_with_expert.forward` with the same 2-D mask and cache extension. |
| ~~A17~~ (fixed, §2.2) | M | `full_vlm_layers_for_cot` is a no-op: SmolVLA truncates the text model to 16 of 32 layers at load, so "full" = 16. `cot_lm_head` (copied from SmolVLM's `lm_head`, trained on layer-32 features) is applied to layer-16 features, and with `train_expert_only=True` the VLM is frozen → CoT is a **linear probe** on frozen mid-layer features. | [read] `smolvlm_with_expert.py:102–105` | Either LoRA on VLM layers for CoT, or reframe CoT as a *latent* plan (small decoder head), which is cheaper and honest. |
| ~~A12~~ (fixed, §2.2) | M | No RL resume (RNG, rollout state), no atomic checkpoint writes, no keep-last-k; RL `save_pretrained` + `rl_state.pt` only. | [read] | Phase 3 infra. |
| A19 | L | Top-k Switch loss uses `f_i` with Σf = k, so its minimum is k, not 1 (scale only). | [read] | Normalize by k for logging clarity. |
| A20 | M | SFT split (`checkpoints/split.json`) is by episode index; with ~23.6 k RoboTwin tasks, train and val share tasks/layouts → val loss measures interpolation, not generalisation. | [read] | Task-level held-out split for any generalisation claim. |
| A21 | L | Trainer validation uses `τ ~ U` while training (now) uses Beta — val loss is a differently-weighted objective. Fine as long as reported as such. | [read] | Document; optionally fixed τ grid. |

### 2.4 Verified correct (no change needed)

- Gaussian transition log-prob `−½Σ[(x−μ)²/σ² + 2 log σ + log 2π]`, masked to valid action dims (`masked_gaussian_log_prob`); σ_k = σ_θ′·√|Δτ| as specified.
- Euler–Maruyama mean `μ_k = x_k + v_θ Δτ` with Δτ = −1/K under `NOISE_AT_ONE`.
- 6-D rotation → SO(3) via Gram–Schmidt on columns (Zhou et al. 2019); geodesic angle `arccos((tr(RᵀR*)−1)/2)` with clamping.
- k3 KL estimator `e^{Δ} − Δ − 1`, Δ = log π_ref − log π (non-negative, unbiased for KL(π‖π_ref)).
- Group z-score advantage `(R − mean)/(std + ε)` as specified (population std). Note: SimpleVLA-RL additionally **filters groups whose success rate is 0 or 1** (zero advantage, wasted compute); worth adopting.
- Normalisation: MEAN_STD for state/action via LeRobot processors; `load_stats` refuses to run without stats (good).

---

## 3. Mismatches with the cited methods

| Method | Paper | Here | Status |
|---|---|---|---|
| SmolVLA (Shukor et al. 2025) | τ ~ Beta(1.5,1); expert interleaves with all VLM layers | U(0,1); half the expert dropped | fixed (A3, A1) |
| GRPO (Shao et al. 2024) / SimpleVLA-RL | group = same prompt/start; token-level ratio; dynamic-sampling filter | different starts; trajectory-level ratio; no filter | fixed (A7, A6); filter open |
| ReinFlow (Zhang et al. 2025) | noise injected per step, learnable σ, log-prob of executed chain | as paper, but log-prob includes unexecuted actions | open (A9) |
| Flow-GRPO (Liu et al. 2025) | per-step ratio, very small clip on mean log-prob | per-step now; clip range untuned | tune in Phase 2 |
| MoE up-cycling (Komatsuzaki et al. 2023) | experts from dense, FLOP-matched comparisons | 3× active FLOPs | open (A13) |

---

## 4. Evaluation protocol (baseline reproduction)

**Protocol.** Benchmarks are evaluated through LeRobot's own integrations (`lerobot-eval`,
`env.type=libero`, `lerobot/sim/integrations/libero.py`), which is the protocol behind the
SmolVLA paper's LIBERO numbers: 8-D state (EEF pos, axis-angle, 2 gripper qpos), images flipped
180°, 10 settle steps, suite max horizons 280/280/300/520, 50 init states per task, success = env
`success` flag. New pixi env `sim-eval` (py3.12, `lerobot[smolvla,libero]`, `MUJOCO_GL=egl`).
Reported: per-task success over 50 episodes, per-suite mean with Wilson 95 % CI, ≥3 seeds
(seed varies policy noise and env seed; init states are fixed by the benchmark).

The repo's own `scripts/libero_server.py` emits a **7-D** state and different image conventions,
so it cannot evaluate `smolvla_libero` faithfully; it stays for RL rollouts once it is aligned
with the LeRobot wrapper (tracked under A10).

**Status.**
- LIBERO / SmolVLA (`HuggingFaceVLA/smolvla_libero`, downloaded): environment install in
  progress; numbers pending — see `docs/baselines.md` once run.
- LIBERO-Plus: supported by the same integration (`is_libero_plus=True`); assets not yet
  installed.
- RoboTwin 2.0: sibling checkout at `../RoboTwin` and a LeRobot integration
  (`sim/integrations/robotwin.py`) exist; no public SmolVLA RoboTwin checkpoint is cached — the
  baseline requires SFT on `lerobot/robotwin_unified` (cached locally).

**Do not proceed to Phase 1 claims until these numbers are in.**
