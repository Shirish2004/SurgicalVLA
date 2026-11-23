# Task: Implement `SurgicalVLA` in LeRobot (SFT + ReinFlow Online RL)

You are an expert robotics ML engineer working inside a Hugging Face `lerobot` repository. Your task is to implement **SurgicalVLA**, an upgraded Vision-Language-Action architecture subclassing/extending `SmolVLAPolicy` and `VLMWithExpertModel` (`lerobot/policies/smolvla/`), designed for precision orientation-aware manipulation and bimanual handover (evaluated on LIBERO and RoboTwin 2.0).

Before writing any code, inspect `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` and `src/lerobot/policies/smolvla/modeling_smolvla.py` and my exisiting architecture under the src/surgicalvla/modules,rl,training so that all tensor shapes, config fields, normalization hooks, and `embed_prefix` / `embed_suffix` / `denoise_step` method signatures match the exact installed version of `lerobot`.

---

## 1. File Structure to Create
Create and modify my policy package under `src/surgicalvla`:
1. `__init__.py` — Register `SmolVLAMoEConfig` and `SmolVLAMoEPolicy` with LeRobot's policy factory.
2. `configuration_smolvla_moe.py` — Subclass `SmolVLAConfig` with new hyperparameters:
   - `num_object_tokens: int = 4`
   - `obj_sam_feat_dim: int = 256`
   - `obj_pose_dim: int = 13` (4D normalized xyxy bbox + 3D centroid + 6D continuous rotation representation)
   - `use_cot: bool = True`, `cot_max_tokens: int = 48`, `cot_refresh_interval: int = 10`
   - `full_vlm_layers_for_cot: bool = True` (use all VLM layers for CoT decoding, while Action Expert still cross-attends to the first `expert_cross_attn_vlm_layers` layers)
   - `num_experts: int = 4`, `num_shared_experts: int = 1`, `moe_top_k: int = 2`, `moe_aux_loss_weight: float = 0.01`, `phase_loss_weight: float = 0.1`
   - `pose_aux_loss_weight: float = 0.2`
   - `use_reinflow: bool = False`, `reinflow_min_std: float = 0.01`, `reinflow_max_std: float = 0.15`
3. `object_encoder.py` — `ObjectCentric6DProjector` and a modular `ZeroShotObjectTrackerWrapper` interface (with mock/cached feature mode for fast offline training + live Grounded-SAM2 + 6D pose hook for eval).
4. `moe_layers.py` — `PhaseConditionedMoEFFN` and `ReinFlowNoiseHead`.
5. `modeling_smolvla_moe.py` — `SmolVLAMoEModel` and `SmolVLAMoEPolicy` implementing:
   - Multi-task SFT forward pass (`FlowMatching + CoT_CE + MoE_Aux + Phase_CE + Pose_Aux`)
   - Asynchronous System-2 CoT KV-cache generation + fast System-1 ODE action chunking
   - SDE rollout + exact step-wise `compute_sde_log_probs()` for ReinFlow / Flow-GRPO
6. `train_reinflow_grpo.py` — Standalone online RL fine-tuning loop implementing Group Relative Policy Optimization (GRPO) over flow-matching SDE trajectories with geodesic 6D orientation reward.
7. `tests/test_smolvla_moe.py` — Comprehensive pytest suite verifying shapes, attention masks, KV-cache parity, gradient flow, and SDE log-prob computation.

---

## 2. Detailed Component Specifications

### A. `ObjectCentric6DProjector` (`object_encoder.py`)
- **Inputs:**
  - `sam_roi_feat`: `(B, N_obj, C_sam)` — mask-pooled visual features from frozen SAM 2.1 (default `N_obj=1` or `2` for bimanual handover; pad/mask inactive objects).
  - `obj_pose_vec`: `(B, N_obj, 13)` — `[x1, y1, x2, y2, tx, ty, tz, r1..r6]` where `r1..r6` is the 6D continuous rotation representation (first two columns of the 3x3 rotation matrix).
  - `obj_valid_mask`: `(B, N_obj)` boolean mask.
- **Architecture:**
  - Project `sam_roi_feat` via `Linear -> LayerNorm -> SiLU` and `obj_pose_vec` via a 2-layer Fourier-feature + MLP encoder (`Linear -> SiLU -> Linear -> LayerNorm`).
  - Concatenate and project into `num_object_tokens` tokens of dimension `D_vlm` (`(B, num_object_tokens, D_vlm)`), adding learned token-slot positional embeddings.
  - Also output a pooled `obj_context_emb` of shape `(B, D_exp)` to condition the MoE router and auxiliary pose head.
  - **Dropout/Noise Robustness:** During training (`self.training == True`), apply `p=0.15` token dropout and small Gaussian jitter (`std=0.01` on 3D translation, `0.02` on 6D rotation) to prevent overfitting to simulator ground-truth poses.

### B. `embed_prefix` & Asynchronous CoT (`modeling_smolvla_moe.py`)
- Extend `embed_prefix` to construct the prefix token sequence in strict order:
  `[Image_Tokens | Text_Instruction_Tokens | Object6D_Tokens | Robot_State_Token | Cached_CoT_Tokens]`
- **Attention Mask Rules (`make_att_2d_masks`):**
  - `Image`, `Text`, and `Object6D` tokens share `att_mask = 0` (bidirectional attention among perception/instruction tokens).
  - `Robot_State` gets `att_mask = 1` (cannot leak into perception/text tokens, preserving SmolVLA's attention isolation).
  - `Cached_CoT_Tokens` use causal attention (`att_mask = 1` per token) so autoregressive CoT decoding matches training.
- **Dual-Speed VLM Execution:**
  - During `select_action()` inference: every `cot_refresh_interval` steps (or when `self._cached_cot_embeds is None`), run the full VLM (`layers 0..num_vlm_layers-1`) with `lm_head` to autoregressively generate up to `cot_max_tokens` structured reasoning tokens:
    `<cot> phase: <ID> | part: <STR> | target_6d: <TOK> | action: <STR> </cot>`
  - Extract the hidden representation of the predicted phase token to form `e_phase: (B, D_exp)` and cache the prefix KV states up to `num_expert_cross_attn_layers` for fast inner-loop ODE denoising.

### C. `PhaseConditionedMoEFFN` (`moe_layers.py`)
- Replace the FFN block inside each Action Expert transformer layer (`lm_expert.layers[i].mlp`) with `PhaseConditionedMoEFFN`:
  - **1 Shared Expert** (always active, weight `1.0`) + **`num_experts` (4) Routed Experts** (`top_k = 2`), each implemented as a SwiGLU MLP matching the original expert FFN parameter budget per active path.
  - **Router Input:** `concat([action_token_hidden, broadcast(e_phase), broadcast(obj_context_emb.detach())], dim=-1)` -> `Linear -> logits (B, T_act, num_experts)`.
  - Compute standard Switch-Transformer load-balancing loss `aux_loss` and store it on the module so `SmolVLAMoEModel.forward()` aggregates `sum(layer.aux_loss)`.
  - Include a lightweight `phase_classifier` head on `e_phase` trained with Cross-Entropy when `batch["subtask_phase_id"]` is present in the batch.

### D. Multi-Task SFT Loss (`SmolVLAMoEModel.forward`)
Compute and return a dictionary of losses + scalar `loss`:
1. `fm_loss`: Standard SmolVLA conditional flow-matching MSE loss $\|v_\theta(x_\tau, \tau) - (a_1 - a_0)\|^2$ (masked by `action_is_pad`).
2. `cot_loss`: Causal LM Cross-Entropy loss on `batch["cot_input_ids"]` (ignored if `cot_input_ids` not in batch, allowing seamless warm-start on datasets without CoT labels).
3. `moe_aux_loss`: Load-balancing loss across all `PhaseConditionedMoEFFN` layers.
4. `phase_loss`: Cross-entropy loss predicting `batch["subtask_phase_id"]` (`0..3`).
5. `pose_aux_loss`: Smooth-L1 + 6D rotation geodesic loss predicting relative target handover pose from `obj_context_emb` + VLM pooled features (`batch["target_relative_pose_6d"]`).

### E. ReinFlow SDE Conversion & GRPO (`moe_layers.py` & `train_reinflow_grpo.py`)
- Implement `ReinFlowNoiseHead(nn.Module)` taking `(expert_final_hidden, timestep_emb)` -> `log_std`, clamped to `[log(reinflow_min_std), log(reinflow_max_std)]`.
- Implement `sample_actions_sde(obs, num_steps=5)` returning `(final_action_chunk, trajectory_latents [K+1, B, 50, 32], step_log_probs [K, B])`:
  - At each step $k \in \{0, \dots, K-1\}$ with $\Delta\tau = 1/K$:
    $$\mu_k = x_k + v_\theta(x_k, \tau_k)\Delta\tau, \quad \sigma_k = \sigma_{\theta'}(x_k, \tau_k)\sqrt{\Delta\tau}$$
    $$x_{k+1} = \mu_k + \sigma_k \odot \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)$$
  - Exact transition log-probability:
    $$\log \pi_\theta(x_{k+1} \mid x_k, s) = -\frac{1}{2} \sum_{t, d} \left( \frac{(x_{k+1} - \mu_k)^2}{\sigma_k^2} + 2\log\sigma_k + \log(2\pi) \right)$$
    (masked over valid action dimensions `action_dim`, ignoring padded dimensions up to `max_action_dim=32`).
- In `train_reinflow_grpo.py`, implement Group Relative Policy Optimization (GRPO):
  - Sample $G=8$ rollouts per environment initial state; compute group-normalized advantage $A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + 10^{-8}}$.
  - Reward function helper `compute_handover_reward()` combining sparse task success, 3D position error, and exact **SO(3) geodesic orientation error** $\arccos\left(\frac{\text{tr}(R_{\text{obj}}^\top R_{\text{target}}) - 1}{2}\right)$ converted from 6D rotation via Gram-Schmidt orthogonalization.
  - Optimize clipped surrogate objective + KL penalty against the frozen SFT reference policy $\pi_{\text{ref}}$ + entropy bonus on $\sigma_{\theta'}$.

---

## 3. Verification & Constraints
- **Backward-Compatible Pretrained Loading:** Provide `SmolVLAMoEPolicy.from_smolvla_pretrained("lerobot/smolvla_base", config)` which loads all matching VLM, vision connector, state_proj, and action_proj weights from `lerobot/smolvla_base`, copies pretrained MLP weights into all `num_experts` + `shared_expert` initializations (warm-start upcycling), and initializes new projection/router layers cleanly.
- **No Hardcoded Dimensions:** Query `vlm_hidden_size` and `expert_hidden_size` dynamically from the instantiated `VLMWithExpertModel`.
- Run `pytest tests/test_smolvla_moe.py` and fix any issues until all tests pass cleanly.
## 4. Logging
Log all losses and metrics, and as the training grow also record the task success rate as it increases. You can make use of the SimpleVLA-Rl repository as it explains how the VLAs can be trained with RL there 