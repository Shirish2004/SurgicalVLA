====================================================================================================
  FROZEN ZERO-SHOT PERCEPTION STREAM (Runs at camera rate; GroundingDINO @ t=0, SAM2+Pose @ 30Hz)
====================================================================================================
  RGB_t + Depth_t + Instruction
        │
        ├──► [GroundingDINO / Florence-2] (keyframe t=0 or phase change) ──► Initial 2D Box / Prompt
        ├──► [SAM 2.1 Video Tracker] (every frame via memory bank) ──────► Binary Mask M_t & Mask Feat F_sam
        └──► [Any6D / FoundationPose / Part-Keypoints] ──────────────────► 3D Centroid p_t (3) + 6D Rot R_t (6)
                                                                                 │
                                          ┌──────────────────────────────────────┘
                                          ▼
                         [ObjectCentric6DProjector (Trainable MLP)]
                         Fuses: Pool(F_sam * M_t) + BBox(4) + [p_t, R_t](9)
                                          │
                                          ▼
                        K_obj = 4 Object-Centric Tokens: (B, 4, D_vlm)
====================================================================================================
  SYSTEM 2: ASYNC EMBODIED CoT + FULL VLM (SmolVLM2-500M, Layers 1..L, runs @ 1-2 Hz or on trigger)
====================================================================================================
  Multi-View RGB ──► [SigLIP + PixelShuffle + Connector] ──► (B, 64 * N_cam, D_vlm) ──┐
  Instruction    ──► [SmolLM2 Tokenizer + Embed]        ──► (B, L_text, D_vlm)       ──┼─► Prefix
  Object Tokens  ──► (from Projector above)              ──► (B, 4, D_vlm)            ──┤   Concat
  Robot State    ──► [state_proj Linear]                 ──► (B, 1, D_vlm)            ──┘
                                                                  │
                    ┌─────────────────────────────────────────────┤
                    ▼ (Full Layers 1..L, Causal LM Head)          ▼ (Cached K/V up to Layer N = L/2)
     [Autoregressive Structured CoT Decoder]             [Prefix KV-Cache]
     Emits: <cot> phase: GRASP | part: handle |          (Shared across K denoising steps &
            target_rot: [6D] | gripper: OPEN </cot>       reused until next CoT refresh)
                    │
                    ├──► Cached CoT Tokens (B, L_cot, D_vlm) ──► Appended to Prefix KV-Cache
                    └──► Phase Embedding e_phase (B, D_exp) ───► Feeds MoE Router
====================================================================================================
  SYSTEM 1: PHASE-CONDITIONED MoE FLOW-MATCHING ACTION EXPERT (Runs @ 20-30 Hz, K=5..10 Euler steps)
====================================================================================================
  Noisy Action Chunk x_tau (B, 50, 32) + Flow Timestep tau ──► [action_time_mlp] ──► (B, 50, D_exp)
                                                                                           │
         ┌─────────────────────────────────────────────────────────────────────────────────┘
         ▼  (Repeated for Expert Layers 1..N_exp)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │  1. Self-Attention over Action Chunk (B, 50, D_exp)                                    │
  │  2. Cross-Attention: Q = Action Tokens, K/V = Prefix KV-Cache [Img | Text | Obj6D | CoT]│
  │  3. Phase-Conditioned Top-2 MoE FFN (Replaces dense MLP in Expert):                    │
  │       Router Input: concat(mean_pool(Action_Tokens), e_phase, StopGrad(Obj_Pose_Emb))  │
  │       Experts (E=4): [E0: Approach, E1: Precision Grasp, E2: Reorient, E3: Handover]   │
  └────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──► Velocity Head v_theta(x_tau, tau) ──► ODE Step: x_{tau+dt} = x_tau + v_theta * dt
         └──► [RL ONLY] ReinFlow Noise Head sigma_theta'(x_tau, tau) ──► SDE Exploration & log_prob
====================================================================================================