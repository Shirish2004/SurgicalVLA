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
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from surgicalvla.data.dataset import load_stats
from surgicalvla.modules.configuration_smolvla_moe import SmolVLAMoEConfig
from surgicalvla.modules.modeling_smolvla_moe import SmolVLAMoEPolicy
from surgicalvla.rl.benchmarks import LIBERO, BenchmarkMismatch, get_benchmark, validate_rl_setup
from surgicalvla.rl.joint import MODES, DemoStream, LambdaSchedule, combine_gradients, make_demo_stream, take_grads
from surgicalvla.rl.lerobot_sim import LeRobotSimClient
from surgicalvla.utils.checkpoints import latest_pretrained
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


def env_section(cfg: dict) -> dict:
    """The simulator section (``env:``; ``libero:`` kept for older configs)."""
    return cfg.get("env") or cfg["libero"]


def resolve_checkpoint(cfg: dict) -> Path:
    """Local checkpoint (or training output dir), or a Hub repo id at ``checkpoint_revision``."""
    ref = str(cfg["checkpoint"])
    if Path(ref).is_dir():
        return latest_pretrained(Path(ref))
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ref, revision=cfg.get("checkpoint_revision")))


def checkpoint_dataset_id(checkpoint: Path) -> str | None:
    """The dataset the checkpoint was trained on, when its train_config.json records it."""
    train_cfg = checkpoint / "train_config.json"
    if not train_cfg.is_file():
        return None
    return (json.loads(train_cfg.read_text()).get("dataset") or {}).get("repo_id")


def build_policy(cfg: dict, device: str) -> SmolVLAMoEPolicy:
    """Load the SFT checkpoint and its normalisation, refusing cross-benchmark setups."""
    spec = get_benchmark(cfg.get("benchmark", ""))
    if env_section(cfg).get("backend", "lerobot") == "socket" and spec is not LIBERO:
        raise BenchmarkMismatch("the socket backend is the LIBERO bridge only; use backend: lerobot")
    checkpoint = resolve_checkpoint(cfg)
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

    action_dim = int(policy.config.action_feature.shape[0])
    state_dim = int(policy.config.robot_state_feature.shape[0])
    policy.config.device = str(device)
    lc = env_section(cfg)

    # Normalisation must be exactly the SFT one: use the checkpoint's saved processors
    # when it has them; a separate stats file is only a fallback and is checked too.
    stats_cfg = cfg.get("stats") or {}
    if (checkpoint / "policy_preprocessor.json").is_file() and not stats_cfg.get("path"):
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        stats_dims = _processor_stat_dims(pre)
        stats_source = f"{checkpoint}/policy_preprocessor.json"
    else:
        if not stats_cfg.get("path"):
            raise BenchmarkMismatch(f"{checkpoint} has no saved processors; set stats.path to its training stats")
        stats = load_stats(stats_cfg["path"])
        stats_dims = (int(stats["observation.state"]["mean"].shape[-1]), int(stats["action"]["mean"].shape[-1]))
        pre, post = make_smolvla_pre_post_processors(config=policy.config, dataset_stats=stats)
        stats_source = stats_cfg["path"]

    validate_rl_setup(
        spec,
        policy_state_dim=state_dim,
        policy_action_dim=action_dim,
        policy_cameras=list(policy.config.image_features),
        camera_map=lc["camera_map"],
        action_slice=lc["action_slice"],
        stats_state_dim=stats_dims[0],
        stats_action_dim=stats_dims[1],
        dataset_id=cfg.get("dataset_id") or checkpoint_dataset_id(checkpoint),
        env_gripper_dims=lc["env_gripper_dims"],
    )
    print(f"normalisation: {stats_source}")
    policy._rl_preprocessor = pre
    policy._rl_postprocessor = post
    return policy.to(device)


def _processor_stat_dims(pipeline) -> tuple[int | None, int | None]:
    """(state_dim, action_dim) of the normaliser statistics inside a processor pipeline."""
    for step in getattr(pipeline, "steps", []):
        stats = getattr(step, "stats", None)
        if stats:
            state = stats.get("observation.state", {}).get("mean")
            action = stats.get("action", {}).get("mean")
            dim = lambda t: None if t is None else int(torch.as_tensor(t).shape[-1])  # noqa: E731
            return dim(state), dim(action)
    return None, None


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
    lc, rl = env_section(cfg), cfg["rl"]
    # GRPO's baseline is the mean over rollouts *from the same start*: one initial
    # state per group, only the policy's SDE noise (and sim seed) varies inside it, so
    # the advantage measures action quality rather than layout difficulty.
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
            action_clip=lc.get("action_clip", 1.0),
            frame_every=int(lc["frame_every"]) if frames else 0,
            generator=generator,
            sde_steps=int(rl.get("sde_steps", 5)),
        )
        result.reward = _reward(cfg, result)
        episodes.append(result)
    return episodes


def evaluate(policy, client, cfg, tasks, *, device):
    """Deterministic validation over every configured suite (monitoring only).

    The model that is evaluated on the benchmark is always ``final/`` after the fixed iteration
    budget; validation success never selects a checkpoint. With LIBERO's ``init_states: false``
    the validation layouts are sampled from fixed seeds and are disjoint from the benchmark's
    evaluation init states.

    ``env.val_suites`` (default: the training suite) x up to ``rl.val_tasks_per_suite`` tasks
    x ``rl.val_episodes_per_task`` episodes on distinct, fixed init states (episode ``e`` uses
    init state ``e`` and seed ``val_seed + e``), so iterations are comparable. ``tasks`` is the
    training suite's task list, reused when the suite is validated.
    """
    lc, rl = env_section(cfg), cfg["rl"]
    suites = lc.get("val_suites") or [lc["suite"]]
    n_tasks = int(rl.get("val_tasks_per_suite", rl.get("max_val_tasks", 4)))
    n_eps = int(rl.get("val_episodes_per_task", 2))
    frames, metrics, all_succ = [], {}, []
    for suite in suites:
        suite_tasks = tasks if suite == lc["suite"] else client.list_tasks(suite)
        succ, rewards, lengths = [], [], []
        for task in suite_tasks[:n_tasks]:
            for e in range(n_eps):
                result = rollout_episode(
                    policy,
                    client,
                    suite=suite,
                    task=task.name,
                    camera_map=lc["camera_map"],
                    device=device,
                    episode=e,
                    seed=int(rl.get("val_seed", 1234)) + e,
                    n_exec_steps=int(lc["n_exec_steps"]),
                    max_steps=lc["max_steps"],
                    max_decisions=lc["max_decisions"],
                    action_slice=tuple(lc["action_slice"]),
                    action_scale=float(lc["action_scale"]),
                    gripper_sign=float(lc["gripper_sign"]),
                    env_gripper_dims=tuple(lc["env_gripper_dims"]),
                    action_clip=lc.get("action_clip", 1.0),
                    frame_every=int(lc["frame_every"]) if e == 0 else 0,
                    collect=False,
                )
                succ.append(float(result.success))
                rewards.append(_reward(cfg, result))
                lengths.append(float(result.steps))
                if result.frames and e == 0:
                    frames.extend(result.frames[: int(rl.get("val_frames_per_task", 4))])
        name = suite.replace(",", "+")[:40]
        metrics[f"rl/val_success/{name}"] = float(np.mean(succ)) if succ else 0.0
        metrics[f"rl/val_reward/{name}"] = float(np.mean(rewards)) if rewards else 0.0
        metrics[f"rl/val_episode_len/{name}"] = float(np.mean(lengths)) if lengths else 0.0
        all_succ += succ
    metrics["rl/val_success_rate"] = float(np.mean(all_succ)) if all_succ else 0.0
    metrics["rl/val_episodes"] = float(len(all_succ))
    return metrics, frames


def demo_fm_loss(policy, demo: DemoStream) -> tuple[torch.Tensor, dict]:
    """Flow-matching loss on one streamed demonstration batch (the SFT objective)."""
    batch = policy._rl_preprocessor(demo.next_batch())
    loss, metrics = policy.forward(batch)
    return loss, metrics


def sft_update(policy, optimizer, demo: DemoStream, *, clip_norm: float) -> dict:
    """One pure flow-matching step (interleaved mode)."""
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = demo_fm_loss(policy, demo)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_norm)
    optimizer.step()
    return {"sft/fm_loss": float(loss.detach()), "sft/grad_norm": float(grad_norm)}


def grpo_update(policy, reference, optimizer, pairs, cfg, *, clip_norm: float, demo: DemoStream | None = None, lam: float = 0.0):
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
            # Only executed chunk steps carry reward signal; replay the rollout's CoT.
            pad, cot = record.exec_pad(), record.cot
            kw = dict(num_steps=num_steps, action_is_pad=pad, cot=cot)
            with torch.no_grad():
                # one optimiser step per rollout batch: the current weights are the behaviour policy
                old = policy.sde_log_probs(batch, record.latents, **kw)
            new, entropy = policy.sde_log_probs(batch, record.latents, return_entropy=True, **kw)
            ref = None
            if reference is not None and kl_coef > 0:
                with torch.no_grad():
                    ref = reference.sde_log_probs(batch, record.latents, **kw)
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
    if demo is not None and lam > 0:
        # joint mode: keep the two gradients apart long enough to measure them
        params = [p for p in policy.parameters() if p.requires_grad]
        g_rl = take_grads(params)
        fm_loss, _ = demo_fm_loss(policy, demo)
        fm_loss.backward()
        g_fm = take_grads(params)
        stats = combine_gradients(params, g_rl, g_fm, lam)
        logs["joint/fm_loss"] = float(fm_loss.detach())
        logs["joint/rl_loss"] = logs.get("rl/loss", 0.0)
        logs.update({f"joint/{k}": v for k, v in stats.items()})
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_norm)
    optimizer.step()
    logs["rl/grad_norm"] = float(grad_norm)
    return logs


def make_client(cfg: dict) -> LeRobotSimClient:
    """The benchmark's simulator built as lerobot-eval builds it (local, or Docker-served via endpoint)."""
    lc = env_section(cfg)
    return LeRobotSimClient(cfg["benchmark"], env_kwargs=lc.get("env_kwargs") or {}, endpoint=lc.get("endpoint"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. run.lr=3e-6")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    run_cfg, proj_cfg = cfg.get("run", {}), cfg.get("project", {})
    lc, rl = env_section(cfg), cfg["rl"]
    device = str(run_cfg.get("device", "cuda"))
    out_dir = run_cfg.get("out_dir", "runs/reinflow_grpo")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(int(run_cfg.get("seed", 0)))
    np.random.seed(int(run_cfg.get("seed", 0)))
    rng = np.random.default_rng(int(run_cfg.get("seed", 0)))

    resume_state = None
    if run_cfg.get("resume"):
        resume_state = torch.load(os.path.join(out_dir, "rl_state.pt"), weights_only=False)
        cfg["checkpoint"] = resume_state["policy_dir"]  # weights saved with that iteration
    policy = build_policy(cfg, device)
    optimizer, clip_norm = build_optimizer(policy, run_cfg.get("lr", 1e-5))
    tcfg = cfg.get("training") or {}
    mode = tcfg.get("mode", "sequential")
    if mode not in MODES:
        raise ValueError(f"training.mode must be one of {MODES}, got {mode!r}")
    schedule = LambdaSchedule(**(tcfg.get("lambda") or {}))
    demo = make_demo_stream(policy.config, cfg["demo"], int(run_cfg.get("seed", 0))) if mode != "sequential" else None
    start_it = 1
    if resume_state is not None:
        state = resume_state
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        if demo is not None and state.get("demo") is not None:
            demo.load_state_dict(state["demo"])
        start_it = int(state["iteration"]) + 1
        print(f"resumed at iteration {start_it} (mode={mode})")
    reference = free_reference(policy) if float(rl.get("kl_coef", 0.0)) > 0 else None
    print(
        f"policy: {cfg['checkpoint']}\n"
        f"benchmark: {cfg['benchmark']}  actions: {policy.config.action_feature.shape[0]}-D\n"
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

    client = make_client(cfg) if lc.get("backend", "lerobot") != "socket" else LiberoClient.spawn(
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
        for it in range(start_it, iters + 1):
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
                lam = schedule(it) if mode == "joint" else 0.0
                metrics.update(
                    grpo_update(policy, reference, optimizer, pairs, cfg, clip_norm=clip_norm, demo=demo, lam=lam)
                )
            if mode == "interleaved" and it % int(tcfg.get("interleave_every", 1)) == 0:
                for _ in range(int(tcfg.get("sft_steps", 1))):
                    metrics.update(sft_update(policy, optimizer, demo, clip_norm=clip_norm))
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
                state = {
                    "iteration": it,
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state,
                    "demo": None if demo is None else demo.state_dict(),
                    "policy_dir": ckpt_dir,
                }
                tmp = os.path.join(out_dir, "rl_state.pt.tmp")
                torch.save(state, tmp)
                os.replace(tmp, os.path.join(out_dir, "rl_state.pt"))  # atomic
                print(f"           saved {ckpt_dir}")
        # Only a completed run gets a final/ (the pipeline's hand-off point).
        final = os.path.join(out_dir, "final")
        policy.save_pretrained(final)
        print(f"saved {final}")
    finally:
        client.close()
        logger.close()


if __name__ == "__main__":
    main()
