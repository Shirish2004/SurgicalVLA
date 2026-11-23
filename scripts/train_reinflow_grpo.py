"""ReinFlow-GRPO online fine-tuning of SmolVLAMoEPolicy on LIBERO.

    pixi run python scripts/train_reinflow_grpo.py --config configs/reinflow_grpo.yaml

Per iteration: sample ``G`` SDE rollouts (``rl.group_size``) of one task, reward
them (success + 6-D geodesic handover shaping), z-score the rewards inside the
group, then take a clipped-surrogate step on the per-step flow-SDE log-probs
against the frozen SFT reference, with an entropy bonus on the noise head.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from surgicalvla.data.dataset import load_stats
from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy
from surgicalvla.rl import (
    LiberoClient,
    compute_handover_reward,
    frame_panel,
    group_normalized_advantages,
    grpo_surrogate,
    rollout_episode,
)
from surgicalvla.utils.config import load_config
from surgicalvla.utils.logrun import ExperimentLogger


def build_policy(cfg: dict, device: str) -> SmolVLAMoEPolicy:
    checkpoint = Path(cfg["checkpoint"])
    policy_type = json.loads((checkpoint / "config.json").read_text()).get("type")
    if policy_type == "smolvla_moe":
        policy = SmolVLAMoEPolicy.from_pretrained(str(checkpoint))
    else:
        base = SmolVLAConfig.from_pretrained(str(checkpoint))
        base_fields = {f.name for f in dataclasses.fields(base)}
        config = SmolVLAMoEConfig(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(SmolVLAMoEConfig) if f.name in base_fields}
        )
        policy = SmolVLAMoEPolicy.from_smolvla_pretrained(str(checkpoint), config)

    stats = load_stats(cfg["stats"]["path"])
    action_dim = int(policy.config.action_feature.shape[0])
    state_dim = int(policy.config.robot_state_feature.shape[0])
    for key, expected in (("action", action_dim), ("observation.state", state_dim)):
        got = int(stats[key]["mean"].shape[-1])
        if got != expected:
            raise ValueError(f"{key} stats are {got}-D but {checkpoint} expects {expected}-D")
    policy.config.device = str(device)
    pre, post = make_smolvla_pre_post_processors(config=policy.config, dataset_stats=stats)
    policy._rl_preprocessor = pre
    policy._rl_postprocessor = post
    return policy.to(device)


def build_optimizer(policy, lr: float):
    opt_cfg = policy.config.get_optimizer_preset()
    opt_cfg.lr = float(lr)
    return opt_cfg.build(policy.parameters()), float(opt_cfg.grad_clip_norm)


def free_reference(policy) -> SmolVLAMoEPolicy:
    ref = copy.deepcopy(policy)
    ref.eval()
    ref.requires_grad_(False)
    return ref


def _reward(cfg: dict, result) -> float:
    target = cfg["rl"].get("target_pose_6d")
    pose = getattr(result.final_observation, "object_pose_6d", None)
    return compute_handover_reward(
        result.success,
        object_pose_6d=None if pose is None else torch.as_tensor(pose),
        target_pose_6d=None if target is None else torch.as_tensor(target),
        success_reward=float(cfg["rl"].get("success_reward", 1.0)),
        position_weight=float(cfg["rl"].get("position_weight", 0.3)),
        orientation_weight=float(cfg["rl"].get("orientation_weight", 0.3)),
    )


def rollout_group(policy, client, cfg, task, *, device, rng, frames=False) -> list:
    lc, rl = cfg["libero"], cfg["rl"]
    # GRPO's baseline is the mean over rollouts *from the same start*: one initial
    # state per group, only the policy's SDE noise (and sim seed) varies inside it.
    # Audit A7: drawing a fresh init state per member made the advantage measure
    # layout difficulty instead of action quality.
    init_state = int(rng.integers(0, max(1, task.n_init_states)))
    episodes = []
    for _ in range(int(rl["group_size"])):
        seed = int(rng.integers(0, 2**31 - 1))
        generator = torch.Generator(device=device).manual_seed(seed)
        result = rollout_episode(
            policy,
            client,
            suite=lc["suite"],
            task=task.name,
            camera_map=lc["camera_map"],
            device=device,
            episode=init_state,
            seed=seed,
            n_exec_steps=int(lc["n_exec_steps"]),
            max_steps=lc["max_steps"],
            max_decisions=lc["max_decisions"],
            action_slice=tuple(lc["action_slice"]),
            action_scale=float(lc["action_scale"]),
            gripper_sign=float(lc["gripper_sign"]),
            env_gripper_dims=tuple(lc["env_gripper_dims"]),
            frame_every=int(lc["frame_every"]) if frames else 0,
            generator=generator,
            sde_steps=int(rl.get("sde_steps", 5)),
        )
        result.reward = _reward(cfg, result)
        episodes.append(result)
    return episodes


def evaluate(policy, client, cfg, tasks, *, device):
    lc = cfg["libero"]
    frames, successes, rewards, lengths = [], [], [], []
    for task in tasks:
        result = rollout_episode(
            policy,
            client,
            suite=lc["suite"],
            task=task.name,
            camera_map=lc["camera_map"],
            device=device,
            episode=0,
            seed=int(cfg["rl"].get("val_seed", 1234)),
            n_exec_steps=int(lc["n_exec_steps"]),
            max_steps=lc["max_steps"],
            max_decisions=lc["max_decisions"],
            action_slice=tuple(lc["action_slice"]),
            action_scale=float(lc["action_scale"]),
            gripper_sign=float(lc["gripper_sign"]),
            env_gripper_dims=tuple(lc["env_gripper_dims"]),
            frame_every=int(lc["frame_every"]),
            collect=False,
        )
        successes.append(float(result.success))
        rewards.append(_reward(cfg, result))
        lengths.append(float(result.steps))
        if result.frames:
            frames.extend(result.frames[: int(cfg["rl"].get("val_frames_per_task", 4))])
    return {
        "rl/val_success_rate": float(np.mean(successes)) if successes else 0.0,
        "rl/val_reward": float(np.mean(rewards)) if rewards else 0.0,
        "rl/val_episode_len": float(np.mean(lengths)) if lengths else 0.0,
    }, frames


def grpo_update(policy, reference, optimizer, pairs, cfg, *, clip_norm: float):
    """One clipped-surrogate step over ``(record, advantage)`` pairs.

    Each record carries the SDE latents and the behaviour log-probs from the rollout.
    """
    rl = cfg["rl"]
    num_steps = int(rl.get("sde_steps", 5))
    micro = max(1, int(cfg["run"].get("micro_batch", 4)))
    clip = float(rl.get("clip", 0.2))
    kl_coef = float(rl.get("kl_coef", 0.0))
    entropy_coef = float(rl.get("entropy_coef", 0.0))

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, float] = {}
    for start in range(0, len(pairs), micro):
        items = pairs[start : start + micro]
        outs = []
        for record, adv in items:
            batch = {k: (v.to(policy.config.device) if torch.is_tensor(v) else v) for k, v in record.batch.items()}
            new, entropy = policy.sde_log_probs(batch, record.latents, num_steps=num_steps, return_entropy=True)
            old = record.log_probs.to(new.device)
            ref = None
            if reference is not None and kl_coef > 0:
                with torch.no_grad():
                    ref = reference.sde_log_probs(batch, record.latents, num_steps=num_steps)
            outs.append(
                grpo_surrogate(
                    new,
                    old,
                    torch.tensor([adv], device=new.device),
                    ref_log_probs=ref,
                    clip=clip,
                    kl_coef=kl_coef,
                    entropy=entropy,
                    entropy_coef=entropy_coef,
                )
            )
        loss = torch.stack([out["loss"] for out in outs]).mean()
        share = len(items) / len(pairs)
        (loss * share).backward()
        for out in outs:
            for key, value in out.items():
                logs[f"rl/{key}"] = logs.get(f"rl/{key}", 0.0) + float(value.detach()) * share / len(items)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_norm)
    optimizer.step()
    logs["rl/grad_norm"] = float(grad_norm)
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. run.lr=3e-6")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    run_cfg, proj_cfg = cfg.get("run", {}), cfg.get("project", {})
    lc, rl = cfg["libero"], cfg["rl"]
    device = str(run_cfg.get("device", "cuda"))
    out_dir = run_cfg.get("out_dir", "runs/reinflow_grpo")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(int(run_cfg.get("seed", 0)))
    np.random.seed(int(run_cfg.get("seed", 0)))
    rng = np.random.default_rng(int(run_cfg.get("seed", 0)))

    policy = build_policy(cfg, device)
    optimizer, clip_norm = build_optimizer(policy, run_cfg.get("lr", 1e-5))
    reference = free_reference(policy) if float(rl.get("kl_coef", 0.0)) > 0 else None
    print(
        f"policy: {cfg['checkpoint']}\n"
        f"actions: {policy.config.action_feature.shape[0]}-D -> {lc['action_slice']} for LIBERO\n"
        f"sde steps: {rl.get('sde_steps', 5)}  group: {rl['group_size']}  kl_coef: {rl.get('kl_coef', 0.0)}"
    )

    logger = ExperimentLogger(
        out_dir,
        use_wandb=proj_cfg.get("wandb_mode", "disabled") != "disabled",
        project=proj_cfg.get("wandb_project", "surgicalvla"),
        entity=proj_cfg.get("entity"),
        run_name=proj_cfg.get("run_name"),
        tags=proj_cfg.get("tags"),
        notes=proj_cfg.get("notes"),
        mode=None if proj_cfg.get("wandb_mode", "disabled") == "disabled" else proj_cfg.get("wandb_mode"),
    )

    client = LiberoClient.spawn(
        camera_height=int(lc["camera_height"]),
        camera_width=int(lc["camera_width"]),
        max_cached_envs=int(lc.get("max_cached_envs", 8)),
    )
    try:
        tasks = client.list_tasks(lc["suite"])
        if lc.get("tasks"):
            tasks = [t for t in tasks if t.name in set(lc["tasks"])]
        if not tasks:
            raise RuntimeError(f"no tasks discovered in suite {lc['suite']}")
        val_tasks = [t for t in tasks if t.name in set(lc["val_tasks"])] if lc.get("val_tasks") else tasks
        val_tasks = val_tasks[: int(rl.get("max_val_tasks", 4))]
        print(f"{len(tasks)} train task(s), {len(val_tasks)} val task(s) in {lc['suite']}")

        iters = int(run_cfg.get("iters", 200))
        for it in range(1, iters + 1):
            task = tasks[int(rng.integers(0, len(tasks)))]
            episodes = rollout_group(policy, client, cfg, task, device=device, rng=rng)
            rewards = [ep.reward for ep in episodes]
            advantages = group_normalized_advantages(rewards)
            pairs = [
                (record, float(advantages[i]))
                for i, ep in enumerate(episodes)
                for record in ep.records
                if record.latents is not None
            ]
            metrics = {
                "rl/train_success_rate": float(np.mean([float(ep.success) for ep in episodes])),
                "rl/train_reward": float(np.mean(rewards)),
                "rl/train_episode_len": float(np.mean([ep.steps for ep in episodes])),
                "rl/advantage_std": float(advantages.std(unbiased=False)) if advantages.numel() > 1 else 0.0,
                "rl/samples": float(len(pairs)),
            }
            if pairs and float(advantages.abs().sum()) > 0:
                metrics.update(grpo_update(policy, reference, optimizer, pairs, cfg, clip_norm=clip_norm))
                usage = policy.expert_usage()
                if usage:
                    logger.log_expert_usage(usage, it)
            logger.log_scalars(metrics, it)
            print(
                f"[{it:5d}/{iters}] {task.name[:30]:30s} succ={metrics['rl/train_success_rate']:.2f} "
                f"r={metrics['rl/train_reward']:.2f} len={metrics['rl/train_episode_len']:4.0f} "
                f"loss={metrics.get('rl/loss', float('nan')):.4f} n={len(pairs):3d}"
            )

            if it % int(run_cfg.get("val_every", 10)) == 0:
                val_metrics, frames = evaluate(policy, client, cfg, val_tasks, device=device)
                logger.log_scalars(val_metrics, it)
                panel = frame_panel(frames) if frames else torch.zeros(0, 3, 1, 1)
                if panel.shape[0]:
                    logger.log_images(panel, it, tag="rl/val_rollout", nrow=min(4, panel.shape[0]))
                print(
                    f"           VAL succ={val_metrics['rl/val_success_rate']:.2f} "
                    f"reward={val_metrics['rl/val_reward']:.2f}"
                )

            if it % int(run_cfg.get("save_every", 25)) == 0:
                ckpt_dir = os.path.join(out_dir, f"step_{it:06d}")
                policy.save_pretrained(ckpt_dir)
                torch.save({"iteration": it, "optimizer": optimizer.state_dict()}, os.path.join(out_dir, "rl_state.pt"))
                print(f"           saved {ckpt_dir}")
    finally:
        client.close()
        logger.close()


if __name__ == "__main__":
    main()
