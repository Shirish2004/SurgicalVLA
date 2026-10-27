"""Training loop for SurgicalVLA: optimizer/scheduler from config presets, grad clipping, periodic logging."""

import os
import re
import warnings
from functools import lru_cache

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch.utils.data import DataLoader
from surgicalvla.utils.splits import strided_subset
from surgicalvla.utils.ema import EMA
from surgicalvla.utils.metrics import Accumulator, chunk_metrics
from surgicalvla.utils.logrun import ExperimentLogger


@lru_cache(maxsize=4)
def _tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def add_language_tokens(batch: dict, config) -> dict:
    """Tokenises ``batch["task"]`` into SmolVLA's language inputs in place (idempotent)."""
    if OBS_LANGUAGE_TOKENS in batch:
        return batch
    task = batch.get("task")
    if task is None:
        return batch
    tasks = [task] if isinstance(task, str) else list(task)
    tasks = [t if t.endswith("\n") else t + "\n" for t in tasks]
    enc = _tokenizer(config.vlm_model_name)(
        tasks,
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = enc["input_ids"]
    batch[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].bool()
    return batch


def _collapse_obs_history(batch: dict) -> dict:
    """Takes the latest frame of any observation.* tensor carrying a history axis -- images
    (B, T, C, H, W) and vectors like state (B, T, D) alike. SmolVLA's forward expects neither
    to carry a history axis (only `action` legitimately keeps its (B, K, D) chunk dimension)."""
    for key, v in batch.items():
        if key.startswith("observation.") and torch.is_tensor(v) and v.dim() in (3, 5):
            batch[key] = v[:, -1]
    return batch


class Trainer:
    def __init__(
        self,
        policy,
        dataset,
        output_dir: str,
        batch_size: int = 32,
        steps: int = 100_000,
        log_every: int = 50,
        eval_every: int = 2_000,
        save_every: int = 5_000,
        num_workers: int = 2,
        device: str = "cuda",
        visual_every: int = 200,
        visual_logging: bool = True,
        logger_kwargs: dict | None = None,
        skill_fit_batches: int = 500,
        rebase_skills: bool = True,
        use_amp: bool = True,
        val_dataset=None,
        val_samples: int = 640,
        val_batch_size: int = 16,
        ema_decay: float | None = 0.999,
        resume_from: str | None = None,
        grad_accum: int = 4,
        repo_id: str | None = None,
    ):
        self.grad_accum = grad_accum
        self.policy = policy.to(device)
        self.device = device
        self.steps = steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.visual_every = visual_every or eval_every
        self.visual_logging = visual_logging
        self.output_dir = output_dir
        self.skill_fit_batches = skill_fit_batches
        self.rebase_skills = rebase_skills
        self.use_amp = use_amp and str(device).startswith("cuda")
        self.repo_id = repo_id

        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=False,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                strided_subset(val_dataset, val_samples),
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=False,
            )
        self.optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
        self.scheduler = policy.config.get_scheduler_preset().build(self.optimizer, steps)
        self.ema = EMA(self.policy, ema_decay) if ema_decay else None
        self.logger = ExperimentLogger(output_dir, config=policy.config, **(logger_kwargs or {}))

        dataset_stats = getattr(getattr(dataset, "meta", None), "stats", None)
        self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(
            policy.config, dataset_stats
        )
        self.start_step = 0
        if resume_from:
            self._load_state(resume_from)

    def _amp(self):
        return torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16, enabled=self.use_amp)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _prep(self, raw: dict) -> dict:
        return self._to_device(_collapse_obs_history(self.preprocessor(raw)))

    def _cycle(self):
        while True:
            yield from self.loader

    def _preprocessed(self):
        for raw in self.loader:
            yield _collapse_obs_history(self.preprocessor(raw))

    def _load_state(self, path: str):
        f = os.path.join(path, "trainer_state.pt")
        if os.path.exists(f):
            st = torch.load(f, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(st["optimizer"])
            self.scheduler.load_state_dict(st["scheduler"])
            if self.ema is not None and st.get("ema") is not None:
                self.ema.load_state_dict(st["ema"])
            self.start_step = st["step"]
            return
        m = re.search(r"step_(\d+)", path)
        if m:
            self.start_step = int(m.group(1))
            for _ in range(self.start_step):
                self.scheduler.step()

    def _get_wandb_run_id(self) -> str | None:
        if hasattr(self.logger, "run_id") and self.logger.run_id:
            return str(self.logger.run_id)
        if hasattr(self.logger, "wandb_run") and getattr(self.logger.wandb_run, "id", None):
            return str(self.logger.wandb_run.id)
        if hasattr(self.logger, "run") and getattr(self.logger.run, "id", None):
            return str(self.logger.run.id)
        return None

    def _save(self, step: int):
        ckpt = f"{self.output_dir}/checkpoints/step_{step}"
        os.makedirs(ckpt, exist_ok=True)
        self.policy.save_pretrained(ckpt)
        
        run_id = self._get_wandb_run_id()
        torch.save(
            {
                "step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "wandb_run_id": run_id,
                "repo_id": self.repo_id,
            },
            f"{ckpt}/trainer_state.pt",
        )
        if run_id:
            with open(os.path.join(ckpt, "wandb_run_id.txt"), "w") as f:
                f.write(run_id)

    def _safe_log_scalars(self, scalars: dict, step: int):
        try:
            self.logger.log_scalars(scalars, step)
        except Exception as e:
            warnings.warn(f"Logging scalars failed at step {step} (resuming catch-up step): {e}")

    def _safe_log_expert_usage(self, usage: dict, step: int):
        try:
            self.logger.log_expert_usage(usage, step)
        except Exception as e:
            pass

    def fit(self):
        self.policy.train()
        if hasattr(self.policy, "fit_skills") and (self.rebase_skills or not bool(self.policy.model.skills.frozen)):
            self.policy.fit_skills(self._preprocessed(), max_batches=self.skill_fit_batches)
        
        data_iter = self._cycle()

        if self.start_step > 0:
            batches_to_skip = self.start_step * self.grad_accum
            print(f"Resuming from step {self.start_step}. Fast-forwarding dataset iterator by {batches_to_skip} batches...")
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = self._cycle()
                    next(data_iter)

        for step in range(self.start_step + 1, self.steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(self.grad_accum):
                batch = self._prep(next(data_iter))
                with self._amp():
                    loss, loss_dict = self.policy.forward(batch)
                (loss / self.grad_accum).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.get_optim_params(), self.policy.config.optimizer_grad_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update()

            if step % self.log_every == 0:
                loss_dict["grad_norm"] = grad_norm.item()
                loss_dict["lr"] = self.scheduler.get_last_lr()[0]
                if torch.cuda.is_available():
                    loss_dict["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                self._safe_log_scalars(loss_dict, step)
                usage = self.policy.expert_usage() if hasattr(self.policy, "expert_usage") else {}
                if usage:
                    self._safe_log_expert_usage(usage, step)

            if step % self.eval_every == 0 or step == self.steps:
                self.validate(step)

            if self.visual_logging and step % self.visual_every == 0:
                self._log_preview(batch, step)

            if step % self.save_every == 0 or step == self.steps:
                self._save(step)

        self.logger.close()

    @torch.no_grad()
    def validate(self, step: int):
        if self.val_loader is None:
            return
        self._validate_once(step, "val", preview=True)
        if self.ema is not None:
            with self.ema.apply():
                self._validate_once(step, "val_ema")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _validate_once(self, step: int, prefix: str, preview: bool = False):
        pol = self.policy
        cfg = pol.config
        pol.eval()
        gen = torch.Generator(device=self.device).manual_seed(1234)
        cuda = str(self.device).startswith("cuda")
        devices = [torch.device(self.device).index or 0] if cuda else []
        acc = Accumulator()
        stats, count, first = {}, 0, None
        for raw in self.val_loader:
            batch = self._prep(raw)
            bsz = batch["action"].shape[0]
            noise = torch.randn(bsz, cfg.chunk_size, cfg.max_action_dim, generator=gen, device=self.device)
            time = torch.rand(bsz, generator=gen, device=self.device) * 0.998 + 0.001
            with self._amp():
                _, ld = pol.forward(batch, noise=noise, time=time)
            for k in ("loss", "losses_after_rm_padding", "skill_ce", "skill_acc", "route_entropy"):
                if k in ld:
                    stats[k] = stats.get(k, 0.0) + float(ld[k]) * bsz
            count += bsz
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                with self._amp():
                    pred = pol.predict_action_chunk(batch)
            acc.add(chunk_metrics(pred, batch["action"], batch.get("action_is_pad"), cfg.gripper_dims))
            if first is None:
                first = (pred.float().cpu(), batch["action"].float().cpu())
        out = {f"{prefix}/{k}": v / max(count, 1) for k, v in stats.items()}
        out.update({f"{prefix}/{k}": v for k, v in acc.result().items() if isinstance(v, float)})
        self._safe_log_scalars(out, step)
        if preview and first is not None:
            try:
                self.logger.log_action_trajectories(first[0], first[1], step, name="val_actions")
            except Exception:
                pass
        pol.train()
        return out

    @torch.no_grad()
    def _log_preview(self, batch: dict, step: int):
        import traceback

        n = 4
        bs = batch["action"].shape[0]
        sub = {k: v[:n] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bs else v for k, v in batch.items()}
        self.policy.eval()
        try:
            with self._amp():
                pred = self.policy.predict_action_chunk(sub)
            target = sub.get("action")
            if target is not None:
                self.logger.log_action_trajectories(pred.float().cpu(), target.float().cpu(), step, name="train_actions")
            for image_key in self.policy.config.image_features:
                if image_key in sub:
                    self.logger.log_images(sub[image_key].float().cpu(), step, tag=image_key.rsplit(".", 1)[-1])
        except Exception:
            traceback.print_exc()
        finally:
            self.policy.train()
            torch.cuda.empty_cache()