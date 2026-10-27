"""Diagnostics for a stuck/noisy SurgicalVLA run.

python debug.py --config train.yaml --repo_id <repo> [--root <path>] [--num_samples 4]

Checks, in order, and stops reporting (but keeps going) on each failure:
  1. AV1 decoder availability (ffmpeg).
  2. Raw dataset samples: per-camera pixel stats (catches black-frame decode failures).
  3. DataLoader with num_workers=0 vs >0 (isolates worker-swallowed decode errors).
  4. SmolVLA pre-processor output: image stats survive normalization.
  5. Episode-boundary padding masks, if present in the batch.
  6. One policy forward pass: loss_dict, grad flow, MoE expert usage.
"""

import argparse
import shutil
import subprocess

import torch
import yaml
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from torch.utils.data import DataLoader

from surgicalvla.data.dataset import build_dataset

try:
    from .train import build_config, resolve_features  # reuse the exact config/feature logic train.py uses
except ImportError:  # run as a plain script: `python scripts/debug.py`
    from train import build_config, resolve_features


def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def check_av1_decoder():
    section("1. AV1 decoder availability")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("FAIL: ffmpeg not found on PATH.")
        return
    out = subprocess.run([ffmpeg, "-decoders"], capture_output=True, text=True).stdout
    av1_lines = [line for line in out.splitlines() if "av1" in line.lower()]
    if not av1_lines:
        print("FAIL: no AV1 decoder registered in this ffmpeg build.")
    else:
        print("Found AV1 decoder(s):")
        for line in av1_lines:
            print(" ", line.strip())
    try:
        import av  # noqa: F401

        print("PyAV import OK (used by most lerobot video backends).")
    except ImportError:
        print("NOTE: PyAV not importable; check which video backend LeRobotDataset is actually using.")


def image_stats(tensor: torch.Tensor) -> dict:
    t = tensor.float()
    return {
        "shape": tuple(tensor.shape),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "nonzero_frac": (t != 0).float().mean().item(),
    }


def check_raw_samples(dataset, image_keys: list[str], num_samples: int):
    section("2. Raw dataset samples (pre-dataloader, pre-preprocessor)")
    for i in range(num_samples):
        sample = dataset[i]
        print(f"-- sample {i} --")
        for key in image_keys:
            if key not in sample:
                print(f"  {key}: MISSING from sample")
                continue
            stats = image_stats(sample[key])
            flag = "FAIL (all zero)" if stats["max"] == 0 else "ok"
            print(f"  {key}: {stats}  [{flag}]")


def check_dataloader_workers(dataset, image_keys: list[str], batch_size: int) -> dict:
    section("3. DataLoader: num_workers=0 vs 4 (isolates worker-swallowed decode errors)")
    last_batch = None
    for nw in (0, 4):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
        batch = next(iter(loader))
        last_batch = batch
        print(f"-- num_workers={nw} --")
        for key in image_keys:
            if key not in batch:
                continue
            stats = image_stats(batch[key])
            flag = "FAIL (all zero)" if stats["max"] == 0 else "ok"
            print(f"  {key}: {stats}  [{flag}]")
    return last_batch


def check_padding(batch: dict):
    section("5. Episode-boundary padding masks")
    pad_keys = [k for k in batch if "pad" in k.lower()]
    if not pad_keys:
        print("No '*pad*' keys found in batch -- confirm your dataset/processor actually emits one "
              "(e.g. action_is_pad) if you expect episode-boundary masking.")
        return
    for key in pad_keys:
        v = batch[key]
        if torch.is_tensor(v):
            print(f"  {key}: shape={tuple(v.shape)} frac_padded={v.float().mean().item():.3f}")
        else:
            print(f"  {key}: {v}")


def check_preprocessor(policy_config, dataset_stats, raw_batch: dict, image_keys: list[str]):
    section("4. Pre-processor output (post-normalization image stats)")
    preprocessor, _ = make_smolvla_pre_post_processors(policy_config, dataset_stats)
    batch = preprocessor(dict(raw_batch))
    for key in image_keys:
        if key not in batch:
            print(f"  {key}: MISSING after preprocessing -- check camera_map / config.input_features keys match.")
            continue
        stats = image_stats(batch[key])
        flag = "FAIL (all zero)" if stats["max"] == stats["min"] else "ok"
        print(f"  {key}: {stats}  [{flag}]")
    return batch


def check_forward(policy, batch: dict):
    section("6. One policy forward pass")
    policy.train()
    loss, loss_dict = policy.forward({k: v.to(policy.device) if torch.is_tensor(v) else v for k, v in batch.items()})
    print("loss_dict:", {k: round(v, 4) if isinstance(v, float) else v for k, v in loss_dict.items()})
    loss.backward()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), float("inf"))
    print(f"total grad norm: {total_grad_norm.item():.4f}  [FAIL if 0.0 -- no gradient reached the params]")

    if hasattr(policy, "expert_usage"):
        usage = policy.expert_usage()
        for layer_idx, counts in usage.items():
            print(f"route usage (sum of expert weights over batch): {counts.tolist()}")
        for k in ("skill_ce", "skill_acc", "route_entropy", "moe_z"):
            print(f"  {k}: {loss_dict.get(k)}")
        sk = policy.model.skills
        print(f"skill clusters initialized={bool(sk.initialized)} frozen={bool(sk.frozen)} usage={sk.usage.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("dataset", {}).get("root", args.root)
    camera_map = cfg.get("dataset", {}).get("camera_map")

    check_av1_decoder()

    config = build_config(cfg)
    resolve_features(config, args.repo_id, camera_map, root=root)
    print("state feature:", config.input_features.get("observation.state"))

    image_keys = list(config.image_features)
    print(f"\nResolved image keys from config.image_features: {image_keys}")

    dataset = build_dataset(args.repo_id, config, root=root, camera_map=camera_map)
    check_raw_samples(dataset, image_keys, args.num_samples)
    raw_batch = check_dataloader_workers(dataset, image_keys, args.batch_size)
    check_padding(raw_batch)

    dataset_stats = getattr(getattr(dataset, "meta", None), "stats", None)
    processed_batch = check_preprocessor(config, dataset_stats, raw_batch, image_keys)
    print("state:", processed_batch.get("observation.state"))

    from surgicalvla.modules.policy import SurgicalVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SurgicalVLAPolicy(config).to(device)
    policy.device = device
    check_forward(policy, processed_batch)

    section("Done")


if __name__ == "__main__":
    main()