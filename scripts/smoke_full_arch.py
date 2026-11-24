"""Full-architecture smoke test on real weights: one training step and inference per MoE variant.

    pixi run -e dev python scripts/smoke_full_arch.py --out results/smoke/full_arch.json

Builds SurgicalVLA from ``lerobot/smolvla_libero`` (warm start) with every experimental
component enabled (auxiliary vision heads, latent CoT, explicit CoT tokens, pseudo phase
labels) and, for each ``moe_variant``, reports total / trainable / active-FFN parameters, the
time and peak VRAM of one forward + backward on a batch of synthetic LIBERO-shaped inputs, and
the latency of one action-chunk inference (10 Euler steps). Numbers are for the RTX A4500 this
repository runs on; they are a feasibility check, not a benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

CHECKPOINT = ("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d")


def build(variant: str, device: str):
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
    from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy

    path = snapshot_download(CHECKPOINT[0], revision=CHECKPOINT[1])
    base = SmolVLAConfig.from_pretrained(path)
    fields = {f.name for f in dataclasses.fields(base)}
    kw = {f.name: getattr(base, f.name) for f in dataclasses.fields(SmolVLAMoEConfig) if f.name in fields}
    kw.update(moe_variant=variant, aux_vision=True, latent_cot=True, use_cot=True, cot_curriculum_steps=1000, object_input="objects")
    policy = SmolVLAMoEPolicy.from_smolvla_pretrained(path, SmolVLAMoEConfig(**kw))
    return policy.to(device)


def batch_for(policy, bsize: int, device: str) -> dict:
    cfg = policy.config
    tok = policy.model._tokenizer()
    b = {k: torch.rand(bsize, *ft.shape, device=device) for k, ft in cfg.image_features.items()}
    b.update({f"future.{k}": torch.rand(bsize, *ft.shape, device=device) for k, ft in cfg.image_features.items()})
    b["observation.state"] = torch.randn(bsize, cfg.robot_state_feature.shape[0], device=device)
    b["action"] = torch.randn(bsize, cfg.chunk_size, cfg.action_feature.shape[0], device=device)
    b["action_is_pad"] = torch.zeros(bsize, cfg.chunk_size, dtype=torch.bool, device=device)
    lang = tok(["pick up the black bowl and place it on the plate\n"] * bsize, return_tensors="pt", padding="max_length", max_length=cfg.tokenizer_max_length)
    b["observation.language.tokens"] = lang.input_ids.to(device)
    b["observation.language.attention_mask"] = lang.attention_mask.bool().to(device)
    cot = tok(["<cot> phase: grasp | object: black bowl | motion: right </cot>"] * bsize, return_tensors="pt", padding="max_length", max_length=cfg.cot_max_tokens, add_special_tokens=False)
    b["cot_input_ids"], b["cot_attention_mask"] = cot.input_ids.to(device), cot.attention_mask.bool().to(device)
    b["subtask_phase_id"] = torch.randint(0, cfg.num_phases, (bsize,), device=device)
    b["obj_sam_feat"] = torch.randn(bsize, cfg.num_objects, cfg.obj_sam_feat_dim, device=device)
    b["obj_pose_vec"] = torch.rand(bsize, cfg.num_objects, cfg.obj_pose_dim, device=device)
    b["obj_valid_mask"] = torch.ones(bsize, cfg.num_objects, dtype=torch.bool, device=device)
    b["task"] = ["pick up the black bowl and place it on the plate"] * bsize
    b.update({"aux.obj_feat": torch.randn(bsize, cfg.aux_feat_dim, device=device), "aux.obj_feat_conf": torch.ones(bsize, device=device),
              "aux.obj_motion": torch.randn(bsize, 2, device=device) * 0.01, "aux.obj_motion_conf": torch.ones(bsize, device=device),
              "aux.track_disp": torch.randn(bsize, 2, device=device) * 0.01, "aux.track_disp_conf": torch.ones(bsize, device=device),
              "aux.rel_disp": torch.randn(bsize, 3, device=device) * 0.1, "aux.rel_disp_conf": torch.ones(bsize, device=device)})
    return b


def measure(variant: str, bsize: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy = build(variant, device)
    params = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    ffn = 0
    for layer in policy.model.vlm_with_expert.lm_expert.layers:
        m = layer.mlp
        if hasattr(m, "shared_experts"):
            ffn += sum(e.gate_proj.out_features for e in m.shared_experts) + m.top_k * m.routed_experts[0].gate_proj.out_features
        else:
            ffn += m.gate_proj.out_features
    batch = batch_for(policy, bsize, device)
    policy.train()
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    for i in range(2):  # warm-up + timed step
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = policy.forward(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        policy.update_ema()
        torch.cuda.synchronize()
        step_s = time.time() - t
    train_peak = torch.cuda.max_memory_allocated() / 2**30
    policy.eval()
    obs = {k: v[:1] for k, v in batch.items() if k.startswith("observation.") or k.startswith("obj_")}
    with torch.no_grad():
        policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
    latency_ms = (time.time() - t) / 5 * 1000
    # explicit CoT decoding alone (cot_refresh_interval chunks share one decode)
    with torch.no_grad():
        images, img_masks = policy.prepare_images(obs)
        torch.cuda.synchronize()
        t = time.time()
        policy.model.generate_cot(images, img_masks, obs["observation.language.tokens"], obs["observation.language.attention_mask"],
                                  policy.prepare_state(obs), policy._objects(obs))
        torch.cuda.synchronize()
    cot_ms = (time.time() - t) * 1000
    # live object perception (SAM 3 + DINOv2 + Depth Anything on one frame), once per chunk
    from surgicalvla.perception.object_inputs import ObjectInputExtractor

    extractor = ObjectInputExtractor(device=device)
    frame = (torch.rand(256, 256, 3) * 255).byte().numpy()
    extractor(frame[None], "black bowl")
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        extractor(frame[None], "black bowl")
    torch.cuda.synchronize()
    perception_ms = (time.time() - t) / 3 * 1000
    infer_peak = torch.cuda.max_memory_allocated() / 2**30
    del extractor
    loss_terms = sorted(k for k in metrics if k.endswith("_loss"))
    row = {"variant": variant, "params_m": params / 1e6, "trainable_m": trainable / 1e6, "active_ffn_width_sum": ffn,
           "train_step_s": step_s, "train_peak_vram_gib": train_peak, "chunk_latency_ms": latency_ms,
           "cot_decode_ms": cot_ms, "perception_ms": perception_ms, "peak_vram_with_perception_gib": infer_peak,
           "loss_terms": loss_terms,
           "loss": float(loss.detach())}
    del policy, opt
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["upcycled", "matched", "dense", "dense_param_matched"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", default="results/smoke/full_arch.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = []
    for v in args.variants:
        rows.append(measure(v, args.batch_size, args.device))
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rows[-1].items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
