import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from torch.utils.data import DataLoader

from surgicalvla.data.dataset import build_dataset
from surgicalvla.modules.policy import SurgicalVLAPolicy
from surgicalvla.training.trainer import collapse_obs_history
from surgicalvla.utils.metrics import Accumulator, chunk_metrics
from surgicalvla.utils.splits import load_split, strided_subset


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--subset", choices=["val", "train", "all"], default="val")
    p.add_argument("--num_samples", type=int, default=1600)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--modes", nargs="+", default=["on", "off", "oracle"])
    p.add_argument("--num_steps", nargs="+", type=int, default=[10])
    p.add_argument("--ema", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="eval_results")
    return p.parse_args()


def load_ema(policy, ckpt, device):
    st = torch.load(os.path.join(ckpt, "trainer_state.pt"), map_location=device, weights_only=False)
    params = dict(policy.named_parameters())
    with torch.no_grad():
        for n, v in st["ema"]["shadow"].items():
            if n in params:
                params[n].copy_(v)


def pick_episodes(args):
    if args.subset == "all":
        return None
    split = args.split or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint.rstrip("/"))), "split.json")
    train, val = load_split(split)
    if args.subset == "val":
        return val
    stride = max(1, len(train) // max(1, len(val)))
    return train[::stride]


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    policy = SurgicalVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    if args.ema:
        load_ema(policy, args.checkpoint, device)
    model, cfg = policy.model, policy.config
    sk = model.skills
    camera_map = None
    if args.config:
        with open(args.config) as f:
            camera_map = yaml.safe_load(f).get("dataset", {}).get("camera_map")

    dataset = build_dataset(args.repo_id, cfg, root=args.root, episodes=pick_episodes(args), camera_map=camera_map)
    stats = getattr(getattr(dataset, "meta", None), "stats", None)
    pre, _ = make_smolvla_pre_post_processors(cfg, stats)
    loader = DataLoader(
        strided_subset(dataset, args.num_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    modes = [m for m in args.modes if not (m == "oracle" and not bool(sk.initialized))]
    configs = [(m, n) for m in modes for n in args.num_steps]
    amp = torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=(not args.no_amp and device.startswith("cuda")))
    cuda = device.startswith("cuda")
    devices = [torch.device(device).index or 0] if cuda else []
    grip = cfg.gripper_dims
    base_steps = cfg.num_steps

    acc, skill_acc, base = {}, {}, {"zero": Accumulator(), "static_first": Accumulator()}
    fwd, share, total = {}, torch.zeros(sk.k), 0
    gen = torch.Generator(device=device).manual_seed(1234)

    with torch.no_grad():
        for raw in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collapse_obs_history(pre(raw)).items()}
            target, pad = batch["action"], batch.get("action_is_pad")
            actions = policy.prepare_action(batch)
            state = policy.prepare_state(batch)
            labels = model.set_skill_labels(actions, state, pad, train=False)
            bsz = target.shape[0]
            total += bsz
            if labels is not None:
                share += torch.bincount(labels.cpu(), minlength=sk.k).float()

            model.moe_mode = "on"
            cfg.num_steps = base_steps
            noise = torch.randn(bsz, cfg.chunk_size, cfg.max_action_dim, generator=gen, device=device)
            time = torch.rand(bsz, generator=gen, device=device) * 0.998 + 0.001
            with amp:
                _, ld = policy.forward(batch, noise=noise, time=time)
            for k in ("loss", "losses_after_rm_padding", "skill_ce", "skill_acc", "route_entropy", "moe_load_balance"):
                if k in ld:
                    fwd[k] = fwd.get(k, 0.0) + float(ld[k]) * bsz

            base["zero"].add(chunk_metrics(torch.zeros_like(target), target, pad, grip))
            base["static_first"].add(chunk_metrics(target[:, :1].expand_as(target), target, pad, grip))

            for mode, ns in configs:
                cfg.num_steps = ns
                model.moe_mode = mode
                if mode == "oracle":
                    model.route.labels = labels
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(0)
                    with amp:
                        pred = policy.predict_action_chunk(batch)
                key = f"{mode}/steps{ns}"
                acc.setdefault(key, Accumulator()).add(chunk_metrics(pred, target, pad, grip))
                if labels is not None:
                    for k in range(sk.k):
                        m = labels == k
                        if m.any():
                            skill_acc.setdefault((key, k), Accumulator()).add(
                                chunk_metrics(pred[m], target[m], None if pad is None else pad[m], grip)
                            )
    model.moe_mode = "on"
    cfg.num_steps = base_steps

    results = {
        "checkpoint": args.checkpoint,
        "ema": args.ema,
        "subset": args.subset,
        "num_samples": total,
        "cluster_share": (share / share.sum().clamp_min(1)).tolist(),
        "forward": {k: v / max(total, 1) for k, v in fwd.items()},
        "baselines": {k: v.result() for k, v in base.items()},
        "runs": {},
    }
    for key, a in acc.items():
        r = a.result()
        r["by_skill"] = {
            str(k): {"mae_arm": skill_acc[(key, k)].result()["mae_arm"], "n": skill_acc[(key, k)].count("mae_arm")}
            for k in range(sk.k)
            if (key, k) in skill_acc
        }
        results["runs"][key] = r

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{args.subset} subset, {total} samples, ema={args.ema}")
    print("cluster share:", [round(x, 3) for x in results["cluster_share"]])
    print("forward:", {k: round(v, 4) for k, v in results["forward"].items()})
    cols = ["mae_arm", "mae_h0_10", "mae_h10_25", "mae_h25_50", "mae_grip", "jitter_ratio"]
    print(f"{'run':<22}" + "".join(f"{c:>13}" for c in cols))
    for key, r in {**results["runs"], **{f"baseline/{k}": v for k, v in results["baselines"].items()}}.items():
        print(f"{key:<22}" + "".join(f"{r[c]:>13.4f}" if c in r else f"{'-':>13}" for c in cols))

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, r in results["runs"].items():
        ax.plot(r["mae_per_step"], label=key, lw=1.5)
    for key, r in results["baselines"].items():
        ax.plot(r["mae_per_step"], label=f"baseline/{key}", lw=1, ls=":")
    ax.set_xlabel("chunk step")
    ax.set_ylabel("MAE (normalized units)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(args.out, "mae_per_step.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "mae_per_step.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.8 / max(1, len(results["runs"]))
    for i, (key, r) in enumerate(results["runs"].items()):
        ks = sorted(int(k) for k in r["by_skill"])
        ax.bar([k + i * w for k in ks], [r["by_skill"][str(k)]["mae_arm"] for k in ks], w, label=key)
    ax.set_xlabel("skill cluster")
    ax.set_ylabel("MAE (normalized units)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(args.out, "mae_by_skill.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "mae_by_skill.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()