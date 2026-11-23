"""Online RL (GRPO-style group advantages) fine-tuning of a SurgicalVLA checkpoint.

    pixi run python scripts/train_rl.py --config configs/rl.yaml
    pixi run python scripts/train_rl.py --config configs/rl.yaml run.lr=3e-6 rl.group_size=4

Per iteration:

1. pick a task, run ``rl.group_size`` episodes in the LIBERO simulator with
   different init states (this is the *group*);
2. turn the per-episode rewards into per-episode weights
   (:func:`surgicalvla.rl.grpo.group_weights`), and give every sampled chunk of an
   episode that episode's weight;
3. take one advantage-weighted flow-matching step on those chunks;
4. every ``run.val_every`` iterations, roll out deterministically on a fixed set
   of val init states and log success rate, reward, episode length **and a strip
   of frames** -- that image panel is the honest answer to "is it actually
   doing the task", which a loss curve cannot give you.

The simulator cannot run in this process, so :class:`surgicalvla.rl.LiberoClient`
spawns ``scripts/libero_server.py`` under ``pixi run -e libero`` and talks to it
over a loopback socket. Nothing here needs the ``libero`` packages.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from surgicalvla.data.dataset import load_stats
from surgicalvla.modules.policy import SurgicalVLAPolicy
from surgicalvla.rl import (
    LiberoClient,
    collate,
    episode_reward,
    frame_panel,
    group_weights,
    rollout_episode,
)
from surgicalvla.rl.grpo import FrozenReference, kl_penalty
from surgicalvla.utils.config import load_config
from surgicalvla.utils.logrun import ExperimentLogger


# --------------------------------------------------------------------------- setup
def build_policy(cfg: dict, device: str) -> SurgicalVLAPolicy:
    """Load the checkpoint, attach normalizing processors built from the dataset stats."""
    checkpoint = cfg["checkpoint"]
    policy = SurgicalVLAPolicy.from_pretrained(checkpoint)

    stats = load_stats(cfg["stats"]["path"])
    action_dim = int(policy.config.action_feature.shape[0])
    state_dim = int(policy.config.robot_state_feature.shape[0])
    for key, expected in (("action", action_dim), ("observation.state", state_dim)):
        got = int(stats[key]["mean"].shape[-1])
        if got != expected:
            raise ValueError(
                f"{cfg['stats']['path']}: statistics for {key} are {got}-D but the checkpoint at "
                f"{checkpoint} expects {expected}-D. Using them anyway would feed the policy a "
                "silently misaligned state/action space."
            )

    policy.config.device = str(device)
    pre, post = make_smolvla_pre_post_processors(config=policy.config, dataset_stats=stats)
    # Attached under private names on purpose: they are part of the RL loop, not
    # of the policy, so they must not be pickled into the checkpoint or shadow
    # LeRobot's own processor plumbing.
    policy._rl_preprocessor = pre
    policy._rl_postprocessor = post
    return policy.to(device)


def build_optimizer(policy: SurgicalVLAPolicy, lr: float):
    """AdamW from the checkpoint's optimizer preset, with the RL learning rate."""
    opt_cfg = policy.config.get_optimizer_preset()
    opt_cfg.lr = float(lr)
    return opt_cfg.build(policy.parameters()), float(opt_cfg.grad_clip_norm)


def rollout_group(policy, client, cfg, task, *, device, rng, collect=True, frames=False):
    """Run ``rl.group_size`` episodes of one task with different init states."""
    lc = cfg["libero"]
    rl = cfg["rl"]
    n_init = max(1, task.n_init_states)
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
            episode=int(rng.integers(0, n_init)),
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
            collect=collect,
        )
        result.reward = episode_reward(
            result.success,
            result.steps,
            horizon=lc["max_steps"],
            efficiency_bonus=float(rl["efficiency_bonus"]),
        )
        episodes.append(result)
    return episodes


def evaluate(policy, client, cfg, tasks, *, device):
    """Deterministic val rollouts: same init states and seeds every time, so runs compare."""
    lc = cfg["libero"]
    rl = cfg["rl"]
    frames = []
    rewards, lengths, successes = [], [], []
    for task in tasks:
        generator = torch.Generator(device=device).manual_seed(int(rl["val_seed"]))
        result = rollout_episode(
            policy,
            client,
            suite=lc["suite"],
            task=task.name,
            camera_map=lc["camera_map"],
            device=device,
            episode=0,
            seed=int(rl["val_seed"]),
            n_exec_steps=int(lc["n_exec_steps"]),
            max_steps=lc["max_steps"],
            max_decisions=lc["max_decisions"],
            action_slice=tuple(lc["action_slice"]),
            action_scale=float(lc["action_scale"]),
            gripper_sign=float(lc["gripper_sign"]),
            env_gripper_dims=tuple(lc["env_gripper_dims"]),
            frame_every=int(lc["frame_every"]),
            generator=generator,
            collect=False,
        )
        successes.append(float(result.success))
        rewards.append(episode_reward(result.success, result.steps, lc["max_steps"], rl["efficiency_bonus"]))
        lengths.append(float(result.steps))
        if result.frames:
            frames.extend(result.frames[: int(cfg["rl"].get("val_frames_per_task", 4))])
    return {
        "rl/val_success_rate": float(np.mean(successes)) if successes else 0.0,
        "rl/val_reward": float(np.mean(rewards)) if rewards else 0.0,
        "rl/val_episode_len": float(np.mean(lengths)) if lengths else 0.0,
    }, frames


# --------------------------------------------------------------------------- update
def weighted_update(
    policy,
    optimizer,
    records,
    weights,
    *,
    micro_batch: int,
    clip: float,
    reference: FrozenReference | None = None,
    kl_coef: float = 0.0,
):
    """One advantage-weighted flow-matching step, split into micro-batches.

    Splitting the samples is necessary (a group of 8 episodes easily produces a
    few hundred chunks), but the objective must stay a single weighted mean over
    the whole group. Each micro-batch therefore contributes
    ``sum(|w|_micro) / sum(|w|_group)`` of its own normalized loss, which sums to
    exactly the group-level objective.

    ``reference``/``kl_coef`` add the optional one-sided flow-loss-gap penalty.
    It is applied to the loss *before* ``backward()`` -- a penalty that is only
    reported is not a penalty.
    """
    w = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    if w.numel() != len(records):
        raise ValueError(f"{w.numel()} weights for {len(records)} decision points")
    total = float(w.abs().sum())
    if total <= 1e-8:
        return None

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, float] = {}
    for start in range(0, len(records), micro_batch):
        chunk = records[start : start + micro_batch]
        wm = w[start : start + micro_batch]
        wm_abs = float(wm.abs().sum())
        if wm_abs <= 1e-8:  # e.g. only failures inside this slice
            continue
        batch = collate(chunk, policy)
        loss, loss_dict = policy.forward(batch, weights=wm)
        scale = wm_abs / total
        if reference is not None and kl_coef:
            current = float(loss_dict["losses_after_rm_padding"])
            ref_loss = reference(batch)
            penalty = kl_penalty(current, ref_loss, kl_coef)
            loss = loss + penalty
            logs["rl/flow_loss_gap"] = logs.get("rl/flow_loss_gap", 0.0) + (current - ref_loss) * scale
            logs["rl/flow_gap_penalty"] = logs.get("rl/flow_gap_penalty", 0.0) + penalty * scale
        (loss * scale).backward()
        for k, v in loss_dict.items():
            logs[k] = logs.get(k, 0.0) + float(v) * scale
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
    optimizer.step()
    logs["rl/grad_norm"] = float(grad_norm)
    logs["rl/micro_batches"] = float(int(np.ceil(len(records) / micro_batch)))
    return logs


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. run.lr=3e-6")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    run_cfg = cfg.get("run", {})
    proj_cfg = cfg.get("project", {})
    lc = cfg["libero"]
    rl = cfg["rl"]
    device = str(run_cfg.get("device", "cuda"))
    out_dir = run_cfg.get("out_dir", "runs/rl")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(int(run_cfg.get("seed", 0)))
    np.random.seed(int(run_cfg.get("seed", 0)))
    rng = np.random.default_rng(int(run_cfg.get("seed", 0)))

    policy = build_policy(cfg, device)
    optimizer, clip = build_optimizer(policy, run_cfg.get("lr", 1e-5))
    kl_coef = float(rl.get("kl_coef", 0.0))
    # Built *before* any update so it really is the starting policy, and only
    # when asked for: the copy is a second full model in GPU memory.
    reference = FrozenReference(policy) if kl_coef else None
    start_iter = 0
    state_path = os.path.join(out_dir, "rl_state.pt")
    if run_cfg.get("resume") and os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = int(state.get("iteration", 0))
        print(f"resumed optimizer state from {state_path} at iteration {start_iter}")
    print(
        f"policy: {cfg['checkpoint']}\n"
        f"actions: {policy.config.action_feature.shape[0]}-D -> {lc['action_slice']} for LIBERO\n"
        f"lr: {optimizer.param_groups[0]['lr']:g}  group: {rl['group_size']}  device: {device}"
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
            wanted = set(lc["tasks"])
            tasks = [t for t in tasks if t.name in wanted]
            missing = wanted - {t.name for t in tasks}
            if missing:
                raise KeyError(f"tasks not found in {lc['suite']}: {sorted(missing)}")
        if not tasks:
            raise RuntimeError(f"no tasks discovered in suite {lc['suite']}")
        val_tasks = tasks
        if lc.get("val_tasks"):
            wanted = set(lc["val_tasks"])
            val_tasks = [t for t in tasks if t.name in wanted]
        val_tasks = val_tasks[: int(rl.get("max_val_tasks", 4))]
        print(f"{len(tasks)} train task(s), {len(val_tasks)} val task(s) in {lc['suite']}")

        iters = int(run_cfg.get("iters", 200))
        val_every = int(run_cfg.get("val_every", 10))
        save_every = int(run_cfg.get("save_every", 25))
        log_every = int(run_cfg.get("log_every", 1))
        micro_batch = int(run_cfg.get("micro_batch", 8))

        for it in range(start_iter + 1, iters + 1):
            task = tasks[int(rng.integers(0, len(tasks)))]
            episodes = rollout_group(policy, client, cfg, task, device=device, rng=rng)

            rewards = [ep.reward for ep in episodes]
            weights, mode = group_weights(
                rewards, normalize=rl.get("adv_normalize", "rloo"), clip=float(rl.get("adv_clip", 5.0))
            )
            records = [r for ep in episodes for r in ep.records]
            sample_weights = np.repeat(weights.numpy(), [len(ep.records) for ep in episodes])

            metrics = {
                "rl/train_success_rate": float(np.mean([float(ep.success) for ep in episodes])),
                "rl/train_reward": float(np.mean(rewards)),
                "rl/train_episode_len": float(np.mean([ep.steps for ep in episodes])),
                "rl/advantage_std": float(weights.std(unbiased=False)) if weights.numel() > 1 else 0.0,
                "rl/advantage_absmax": float(weights.abs().max()),
                "rl/samples": float(len(records)),
                "rl/mode_advantage": float(mode == "advantage"),
                "rl/mode_rejection": float(mode == "rejection"),
            }

            if mode == "none" or not records:
                metrics["rl/skipped_update"] = 1.0
            else:
                if kl_coef and reference is None:
                    print("building the frozen reference policy for the flow-loss-gap penalty ...")
                    reference = FrozenReference(policy)
                update_logs = weighted_update(
                    policy,
                    optimizer,
                    records,
                    sample_weights,
                    micro_batch=micro_batch,
                    clip=clip,
                    reference=reference,
                    kl_coef=kl_coef,
                )
                if update_logs:
                    metrics.update(update_logs)
                metrics["rl/skipped_update"] = 0.0
                if policy._last_usage:
                    logger.log_expert_usage(policy._last_usage, it)

            if it % log_every == 0:
                logger.log_scalars(metrics, it)
                print(
                    f"[{it:5d}/{iters}] {task.name[:34]:34s} "
                    f"succ={metrics['rl/train_success_rate']:.2f} "
                    f"len={metrics['rl/train_episode_len']:5.0f} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"{mode:9s} n={len(records):3d}"
                )

            if it % val_every == 0:
                val_metrics, frames = evaluate(policy, client, cfg, val_tasks, device=device)
                logger.log_scalars(val_metrics, it)
                panel = frame_panel(frames) if frames else torch.zeros(0, 3, 1, 1)
                if panel.shape[0] > 0:
                    logger.log_images(panel, it, tag="rl/val_rollout", nrow=min(4, panel.shape[0]))
                print(
                    f"           VAL succ={val_metrics['rl/val_success_rate']:.2f} "
                    f"reward={val_metrics['rl/val_reward']:.2f} "
                    f"len={val_metrics['rl/val_episode_len']:.0f}"
                )

            if it % save_every == 0:
                ckpt_dir = os.path.join(out_dir, f"step_{it:06d}")
                policy.save_pretrained(ckpt_dir)
                torch.save(
                    {"iteration": it, "optimizer": optimizer.state_dict()},
                    os.path.join(out_dir, "rl_state.pt"),
                )
                print(f"           saved {ckpt_dir}")
    finally:
        client.close()
        logger.close()


if __name__ == "__main__":
    main()
