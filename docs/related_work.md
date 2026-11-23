# Related work and component choices

Checked 2026-11-23. Licenses were read from each project's own LICENSE file, not from
third-party summaries. "Release-safe" means the component can ship with, or be used to
produce artefacts for, an open-source release that allows commercial use.

**Design constraint.** All auxiliary targets are produced offline by frozen models from raw
demonstrations (video, proprioception, language) and used only during training. Simulator
ground truth is used only to validate pseudo-labels and as an oracle ablation.

## 1. Open-vocabulary segmentation and tracking

| Method | Link | Date | License | Release-safe | What we take |
|---|---|---|---|---|---|
| SAM 3 | [paper/blog](https://ai.meta.com/blog/segment-anything-model-3), [code](https://github.com/facebookresearch/sam3), [weights](https://huggingface.co/facebook/sam3) | Nov 2025 | SAM License (Meta, custom) | **Yes, with conditions**: pass on the agreement, acknowledge use in publications, trade-control / military-use exclusion | Text-prompted masks and video tracks for the instruction's target object |
| SAM 3.1 | [weights](https://huggingface.co/facebook/sam3.1) | 2026 | SAM License | Same as SAM 3 | Faster multi-object *video* tracking (multiplexed); image detector unchanged. Adopt once the transformers port supports it |

**Measured on our GPU (RTX A4500, transformers `Sam3Model`, fp16, 480×640 frame, prompt "bottle"):**
840 M parameters, **2.0 GiB peak VRAM**, **171 ms per image**. Third-party posts cite
4–8 GB; our measurement is lower, so SAM 3 fits alongside training. The correct object was
segmented at score 0.96; the robot gripper was also returned at score 0.55, so the labeler
needs a score threshold and gripper suppression (§ Step 1 design).

## 2. Point tracking and features

| Method | Link | Date | License | Release-safe | What we take |
|---|---|---|---|---|---|
| CoTracker3 | [paper](https://arxiv.org/abs/2410.11831), [code](https://github.com/facebookresearch/co-tracker) | Oct 2024 (ICCV 2025) | **CC BY-NC 4.0** | **No** (non-commercial) | Not used. Reference point only |
| TAPNext | [paper](https://arxiv.org/abs/2504.05579), [code](https://github.com/google-deepmind/tapnet) | Apr 2025 (ICCV 2025) | Apache 2.0 | Yes | **Proposed replacement for CoTracker3**: online tracker, real-time for 1,024 points per the authors (to verify) |
| TAPNext++ | [paper](https://arxiv.org/abs/2604.10582) | Apr 2026 | not checked | — | Newer, better occlusion handling per the authors; check license before use |
| AllTracker | [paper](https://arxiv.org/abs/2506.07310), [code](https://github.com/aharley/alltracker) | Jun 2025 | MIT (code; weight license not checked) | Code yes | Dense high-resolution alternative |
| DINOv2 | [code](https://github.com/facebookresearch/dinov2) | 2023 | Apache 2.0 | Yes | **Default** mask-pooled object features (ViT-S/14 is cached) |
| DINOv3 | [blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [code](https://github.com/facebookresearch/dinov3) | Aug 2025 | DINOv3 License (Meta, custom; commercial allowed, trade-control exclusion) | Yes, with conditions | Optional stronger features; gated weights |
| Video Depth Anything (Small) | [code](https://github.com/DepthAnything/Video-Depth-Anything) | Jan 2025 (CVPR 2025) | Apache 2.0 (code; Small weights Apache per README; Large is CC BY-NC) | Small: yes | Temporally consistent *relative* depth for gripper-relative displacement. The metric release is based on the Large model (non-commercial) |

## 3. Zero-shot 6D pose (optional, real-robot only)

| Method | Link | Date | License | Release-safe | Notes |
|---|---|---|---|---|---|
| FoundationPose | [paper](https://arxiv.org/abs/2312.08344), [code](https://github.com/NVlabs/FoundationPose) | Dec 2023 (CVPR 2024) | NVIDIA Source Code License | **No**: "only may be used … non-commercially" | **Blocks open release.** NVIDIA's NGC-packaged model lists a different (Open Model) license; check that one if needed |
| HIPPo (model-free) | [paper](https://arxiv.org/abs/2502.10606), [site](https://hippope.github.io) | Feb 2025 | **No code released** ("Due to IP policies, we are unable to release…") | n/a | Not usable |
| FreeZeV2 | [paper](https://arxiv.org/abs/2506.09784) | Jun 2025 | not found | — | Model-based, training-free; BOP 2024 best overall per the authors |
| ConceptPose (model-free) | [paper](https://arxiv.org/abs/2512.09056) | Dec 2025 | code status unknown | — | Watch |

Consequence: 6D pose stays **off by default**, as specified. If enabled for the Piper with
scanned instrument meshes, the estimator must be release-safe (NGC FoundationPose under its
own license, or another model) and is never a training-time dependency.

## 4. Reasoning for VLAs

| Method | Link | Date | License | What we take | Verified? |
|---|---|---|---|---|---|
| LaRA-VLA (latent reasoning) | [paper](https://arxiv.org/abs/2602.01166), [code](https://github.com/LoveJu1y/LaRA-VLA) | Feb 2026 (ICML 2026) | MIT header with non-standard extra text; copyright "StarVLA Team" — **ambiguous** | Three-stage curriculum: explicit CoT → compact latent tokens; aligned visual-prediction latents with an EMA target encoder; reports up to 90 % lower latency than explicit CoT and 97.9 % average on LIBERO | Claims not reproduced; ideas only, no code reuse until the license is clarified |
| R&B-EnCoRe | [paper](https://arxiv.org/abs/2602.08167), [code](https://github.com/rnb-encore) | Feb 2026 (RSS 2026) | LIBERO repo MIT; others unlabelled | Keep reasoning traces in proportion to how much they raise the expert action's likelihood (importance-weighted, self-supervised; no external verifier) | To test in Step 3 as our trace filter |
| DreamVLA | [paper](https://arxiv.org/abs/2507.04447), [code](https://github.com/Zhangwenyao1/DreamVLA) | Jul 2025 (NeurIPS 2025) | **No LICENSE file** (README badges link to another project) | Predict compact future "world knowledge" (dynamic regions, depth, DINOv2/SAM semantics) instead of frames; their ablation: dynamic-region prediction gives the largest gain | Ideas only, no code reuse |

## 5. Flow-matching RL and joint SFT + RL

| Method | Link | Date | License | Relation to our design |
|---|---|---|---|---|
| Flow-GRPO | [paper](https://arxiv.org/abs/2505.05470) | May 2025 | — | ODE → SDE for exact per-step likelihoods; per-step ratios (we adopted per-step ratios) |
| ReinFlow | [code](https://github.com/ReinFlow/ReinFlow) | 2025 | MIT | Learned noise network makes the flow a discrete-time MDP with exact likelihoods (our `ReinFlowNoiseHead`) |
| πRL | [paper](https://arxiv.org/abs/2510.25889) | Oct 2025 | — | Flow-Noise (= ReinFlow-style) and Flow-SDE for π0/π0.5 with PPO; reports large LIBERO gains (versions of the paper report different numbers). **Proposal:** add Flow-SDE as an RL ablation next to our Flow-Noise head |
| SimpleVLA-RL | local `SimpleVLA-RL/` | 2025 | MIT | Group-relative advantages per start state, dynamic sampling (drop all-success/all-fail groups) |
| VLA-RFT / Z-1 | [VLA-RFT](https://arxiv.org/abs/2510.00406), [Z-1](https://arxiv.org/abs/2606.31846) | 2025–2026 | — | Learned σ per step; chunk log-prob as sum of per-transition log-probs (same as ours) |
| RL-Co | [paper](https://arxiv.org/abs/2602.12628) | Feb 2026 | — | Auxiliary supervised loss on demonstrations during RL to prevent forgetting: closest prior work to our `training.mode: joint` |
| Non-decoupling of SFT and RL | [paper](https://arxiv.org/abs/2601.07389) | Jan 2026 | — | Argues SFT and RL should be optimised jointly; motivates the joint-vs-sequential ablation |
| MIFO | [paper](https://arxiv.org/abs/2510.04454) | Oct 2025 | — | Interleaved SFT/RL with forgetting mitigation (LLMs); reference for `interleaved` |

**Novelty check for our joint mode.** RL-Co already adds a demo loss during RL. What is not
in these papers is measuring the interaction directly (per-term gradient norms and
cos(g_RL, g_FM) logged every update) and a λ schedule chosen from that signal; that, plus the
forgetting measurement, is what the pilot should establish, not the joint loss itself.

## 6. Recommended stack (all release-safe)

| Target | Component |
|---|---|
| Target noun phrase | SmolVLM2 (already in the backbone) or a rule-based parser as fallback |
| Masks and tracks | SAM 3 (transformers), SAM 3.1 when ported |
| Object features | DINOv2 ViT-S/14 (Apache); DINOv3 optional |
| Point tracks | **TAPNext** (Apache); decided 2026-11-23: the release must allow commercial use, so CoTracker3 (CC BY-NC) is excluded |
| Depth | Video Depth Anything Small (Apache), relative depth only |
| Phases | Gripper open/close events and end-effector speed minima from proprioception |
| 6D pose | Off; optional real-robot module only |

Claims marked "per the authors" have not been reproduced here. The SAM 3 numbers above were
measured in this repository (`scratchpad` probe; to be turned into `scripts/` with the
pseudo-label pipeline).
