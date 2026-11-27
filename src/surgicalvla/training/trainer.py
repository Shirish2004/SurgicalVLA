"""Training loop for SurgicalVLA: optimizer/scheduler from config presets, grad clipping, periodic logging."""

import os
import random
import re
import time
import warnings
from functools import lru_cache

import numpy as np
import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch.utils.data import DataLoader, IterableDataset

from surgicalvla.training.sampler import ResumableRandomSampler
from surgicalvla.utils.ema import EMA
from surgicalvla.utils.logrun import ExperimentLogger
from surgicalvla.utils.metrics import Accumulator, chunk_metrics
from surgicalvla.utils.splits import strided_subset
from surgicalvla.pseudolabels.reader import split_future_frames


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


def collapse_obs_history(batch: dict) -> dict:
    """Takes the latest frame of any observation.* tensor carrying a history axis -- images
    (B, T, C, H, W) and vectors like state (B, T, D) alike. SmolVLA's forward expects neither
    to carry a history axis (only `action` legitimately keeps its (B, K, D) chunk dimension)."""
    for key, v in batch.items():
        if key.startswith("observation.") and torch.is_tensor(v) and v.dim() in (3, 5):
            batch[key] = v[:, -1]
    return batch


def router_stats(usage: dict) -> dict:
    """Per-layer routing entropy (normalised to [0, 1]) and the max/min expert load ratio."""
    import math

    out, ents = {}, []
    for layer, counts in usage.items():
        frac = counts.float() / counts.sum().clamp_min(1)
        ent = float(-(frac * frac.clamp_min(1e-12).log()).sum()) / math.log(max(len(frac), 2))
        ents.append(ent)
        out[f"moe/entropy/layer{layer}"] = ent
        out[f"moe/load_max_over_min/layer{layer}"] = float(frac.max() / frac.min().clamp_min(1e-6))
    if ents:
        out["moe/entropy_mean"] = sum(ents) / len(ents)
    return out


def get_rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_train_loader(dataset, batch_size: int, sampler, num_workers: int, prefetch: int = 4) -> DataLoader:
    """Training loader: each worker keeps ``prefetch`` batches ready, and workers survive the end of
    an iterator (streaming data); the shuffle order still comes from ``sampler`` in the main
    process, so resuming is unaffected. ``run.num_workers`` sets the
    count (A100 + 12 CPUs: ~10 for one run, ~3 each for three runs sharing the machine)."""
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=prefetch if num_workers else None, persistent_workers=num_workers > 0,
    )


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
        aux_targets=None,
        seed: int = 0,
        label_viz=None,
        label_image_key: str | None = None,
        push_fn=None,
        push_every_save: bool = False,
        grad_diag_every: int = 0,
        run_info: dict | None = None,
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

        self.aux_targets = aux_targets  # surgicalvla.pseudolabels.reader.SidecarTargets or None
        # sidecar reader used only to draw pseudo-labels on logged frames (never a training signal)
        self.label_viz = label_viz
        self.label_image_key = label_image_key
        self._last_ids = None
        # push_fn(checkpoint_dir, step): Hub upload of the final checkpoint (or every one)
        self.push_fn = push_fn
        self.push_every_save = push_every_save
        # every N steps: per-loss gradient norms and pairwise cosines (0 = off)
        self.grad_diag_every = int(grad_diag_every)
        self.run_info = run_info  # provenance dict (utils.runinfo); completed with parameters and FLOPs in fit()
        self._cache_s = 0.0
        self.batch_size = batch_size
        # Map-style data is shuffled by a sampler whose order is a function of (seed, position),
        # so resuming skips to the saved position without loading the skipped samples.
        self.sampler = None if isinstance(dataset, IterableDataset) else ResumableRandomSampler(len(dataset), seed)
        self.loader = make_train_loader(dataset, batch_size, self.sampler, num_workers)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                strided_subset(val_dataset, val_samples),
                batch_size=val_batch_size or batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),
                pin_memory=True,
                prefetch_factor=2 if num_workers else None,
                persistent_workers=False,
            )
        self.optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
        self.scheduler = policy.config.get_scheduler_preset().build(self.optimizer, steps)
        self.ema = EMA(self.policy, ema_decay) if ema_decay else None
        self.start_step = self._saved_step(resume_from) if resume_from else 0
        self.logger = ExperimentLogger(
            output_dir, config=policy.config, resume_step=self.start_step or None, **(logger_kwargs or {})
        )

        dataset_stats = getattr(getattr(dataset, "meta", None), "stats", None)
        self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(
            policy.config, dataset_stats
        )
        if resume_from:
            self._load_state(resume_from)

    def _amp(self):
        return torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16, enabled=self.use_amp)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _prep(self, raw: dict) -> dict:
        self._last_ids = (raw.get("episode_index"), raw.get("frame_index"))
        batch = self.preprocessor(raw)
        if getattr(self.policy.config, "latent_cot", False):
            batch = split_future_frames(batch, list(self.policy.config.image_features))
        batch = collapse_obs_history(batch)
        if self.aux_targets is not None and "episode_index" in raw and "frame_index" in raw:
            t = time.perf_counter()
            batch.update(self.aux_targets.for_batch(raw["episode_index"], raw["frame_index"]))
            self._cache_s += time.perf_counter() - t
        return self._to_device(batch)

    def _cycle(self):
        while True:
            yield from self.loader

    def _preprocessed(self):
        for raw in self.loader:
            yield collapse_obs_history(self.preprocessor(raw))

    @staticmethod
    def _saved_step(path: str) -> int:
        f = os.path.join(path, "trainer_state.pt")
        if os.path.exists(f):
            return int(torch.load(f, map_location="cpu", weights_only=False)["step"])
        m = re.search(r"step_(\d+)", path)
        return int(m.group(1)) if m else 0

    def _load_state(self, path: str):
        """Restore optimizer, scheduler, EMA, RNG and data position saved with a checkpoint."""
        f = os.path.join(path, "trainer_state.pt")
        if os.path.exists(f):
            st = torch.load(f, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(st["optimizer"])
            self.scheduler.load_state_dict(st["scheduler"])
            if self.ema is not None and st.get("ema") is not None:
                self.ema.load_state_dict(st["ema"])
            if st.get("rng"):
                set_rng_state(st["rng"])
        else:
            for _ in range(self.start_step):
                self.scheduler.step()
        if self.sampler is not None:
            self.sampler.set_position(self.start_step * self.grad_accum * self.batch_size)

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
        # Save the normaliser with the weights: lerobot-eval and the RL stage read it from
        # the checkpoint, so a checkpoint without it cannot be evaluated or fine-tuned.
        self.preprocessor.save_pretrained(ckpt)
        self.postprocessor.save_pretrained(ckpt)

        run_id = self._get_wandb_run_id()
        torch.save(
            {
                "step": step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "wandb_run_id": run_id,
                "repo_id": self.repo_id,
                "rng": get_rng_state(),
            },
            f"{ckpt}/trainer_state.pt",
        )
        if run_id:
            with open(os.path.join(ckpt, "wandb_run_id.txt"), "w") as f:
                f.write(run_id)
        if self.push_fn is not None and (step == self.steps or self.push_every_save):
            self.push_fn(ckpt, step)

    def _safe_log_scalars(self, scalars: dict, step: int):
        try:
            self.logger.log_scalars(scalars, step)
        except Exception as exc:
            warnings.warn(f"Logging scalars failed at step {step} (resuming catch-up step): {exc}", stacklevel=2)

    def _safe_log_expert_usage(self, usage: dict, step: int, heatmap: bool = True):
        try:
            self.logger.log_expert_usage(usage, step, heatmap=heatmap)
        except Exception:
            pass

    def _write_run_info(self, batch: dict | None) -> None:
        """Provenance with parameter counts and FLOPs measured on ``batch`` (the first training batch;
        decoding data here, before the loader workers start, would corrupt their video decoders)."""
        from surgicalvla.utils import runinfo

        info = runinfo.collect(self.run_info.get("config", {}), policy=self.policy, batch=batch,
                               seeds=self.run_info.get("seeds"), data=self.run_info.get("data"))
        runinfo.write(info, self.output_dir)
        if self.logger.wandb:
            try:
                self.logger.wandb.config.update({"run_info": {k: v for k, v in info.items() if k != "config"}}, allow_val_change=True)
            except Exception:
                pass

    def fit(self):
        self.policy.train()
        if hasattr(self.policy, "fit_skills") and (self.rebase_skills or not bool(self.policy.model.skills.frozen)):
            self.policy.fit_skills(self._preprocessed(), max_batches=self.skill_fit_batches)
        
        data_iter = self._cycle()

        if self.start_step > 0 and self.sampler is None:  # iterable data: no random access
            batches_to_skip = self.start_step * self.grad_accum
            print(f"Resuming from step {self.start_step}. Fast-forwarding dataset iterator by {batches_to_skip} batches...")
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = self._cycle()
                    next(data_iter)

        clip = float(self.policy.config.optimizer_grad_clip_norm)
        window = {"clipped": 0, "steps": 0, "data_s": 0.0, "cache_s": 0.0, "step_s": 0.0, "samples": 0}
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        for step in range(self.start_step + 1, self.steps + 1):
            t_step = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            # Average the reported metrics over the micro-batches: logging only the
            # last one makes the logged loss noisier than the loss being optimised.
            micro: dict[str, float] = {}
            for _ in range(self.grad_accum):
                t = time.perf_counter()
                raw = next(data_iter)
                window["data_s"] += time.perf_counter() - t  # time blocked on the data loader
                batch = self._prep(raw)
                if self.run_info is not None and step == self.start_step + 1 and not micro:
                    self._write_run_info(batch)
                    self.policy.train()
                window["samples"] += int(batch["action"].shape[0])
                with self._amp():
                    loss, loss_dict = self.policy.forward(batch)
                (loss / self.grad_accum).backward()
                for k, v in loss_dict.items():
                    micro[k] = micro.get(k, 0.0) + float(v) / self.grad_accum
            loss_dict = micro
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.get_optim_params(), clip)
            window["clipped"] += int(float(grad_norm) > clip)
            self.optimizer.step()
            self.scheduler.step()
            if hasattr(self.policy, "update_ema"):
                self.policy.update_ema()
            if self.ema is not None:
                self.ema.update()
            sync()
            window["step_s"] += time.perf_counter() - t_step
            window["steps"] += 1
            window["cache_s"] += self._cache_s
            self._cache_s = 0.0

            if self.grad_diag_every and step % self.grad_diag_every == 0:
                self._log_grad_diagnostics(batch, step)

            if step % self.log_every == 0:
                n = max(window["steps"], 1)
                loss_dict.update({
                    "perf/step_s": window["step_s"] / n, "perf/data_wait_s": window["data_s"] / n,
                    "perf/cache_read_s": window["cache_s"] / n, "perf/samples_per_s": window["samples"] / max(window["step_s"], 1e-9),
                    "perf/data_stall_frac": window["data_s"] / max(window["step_s"], 1e-9), "grad_clip_rate": window["clipped"] / n,
                })
                window = dict.fromkeys(window, 0) | {"data_s": 0.0, "cache_s": 0.0, "step_s": 0.0}
                loss_dict["grad_norm"] = grad_norm.item()
                loss_dict["lr"] = self.scheduler.get_last_lr()[0]
                if torch.cuda.is_available():
                    loss_dict["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                self._safe_log_scalars(loss_dict, step)
                usage = self.policy.expert_usage() if hasattr(self.policy, "expert_usage") else {}
                if usage:
                    self._safe_log_scalars(router_stats(usage), step)
                    heatmap = self.visual_logging and step % self.visual_every == 0
                    self._safe_log_expert_usage(usage, step, heatmap=heatmap)

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
        if torch.cuda.is_available():
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
        faith: dict[str, list[float]] = {}
        for raw in self.val_loader:
            batch = self._prep(raw)
            bsz = batch["action"].shape[0]
            noise = torch.randn(bsz, cfg.chunk_size, cfg.max_action_dim, generator=gen, device=self.device)
            time = torch.rand(bsz, generator=gen, device=self.device) * 0.998 + 0.001
            with self._amp():
                _, ld = pol.forward(batch, noise=noise, time=time)
            if hasattr(pol, "vlm_feature_drift") and getattr(cfg, "vlm_lora_rank", 0) > 0:
                with self._amp():
                    drift = pol.vlm_feature_drift(batch)
                stats["vlm_feature_cos"] = stats.get("vlm_feature_cos", 0.0) + drift * bsz
            if "cot_input_ids" in batch and bsz > 1:
                self._cot_faithfulness(batch, noise, time, faith)
            for k in (
                "loss",
                "losses_after_rm_padding",
                "fm_loss",
                "cot_loss",
                "cot_token_acc",
                "cot_trace_len",
                "aux_feat_loss",
                "aux_motion_loss",
                "aux_track_loss",
                "aux_rel_loss",
                "latent_cot_loss",
                "moe_aux_loss",
                "phase_loss",
                "pose_aux_loss",
                "skill_ce",
                "skill_acc",
                "route_entropy",
                "moe_load_balance",
                "moe_z",
            ):
                if k in ld:
                    stats[k] = stats.get(k, 0.0) + float(ld[k]) * bsz
            count += bsz
            if hasattr(pol, "reset"):
                pol.reset()
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0)
                with self._amp():
                    pred = pol.predict_action_chunk(batch)
            acc.add(chunk_metrics(pred, batch["action"], batch.get("action_is_pad"), cfg.gripper_dims))
            if first is None:
                first = (pred.float().cpu(), batch["action"].float().cpu())
        out = {f"{prefix}/{k}": v / max(count, 1) for k, v in stats.items()}
        out.update({f"{prefix}/{k}": v for k, v in acc.result().items() if isinstance(v, float)})
        out.update({f"{prefix}/cot_gain/{k}": float(np.mean(v)) for k, v in faith.items() if v})
        self._safe_log_scalars(out, step)
        if preview and getattr(cfg, "use_cot", False) and hasattr(pol, "model") and hasattr(pol.model, "generate_cot"):
            self._log_generated_traces(step)
        if preview and first is not None:
            try:
                self.logger.log_action_trajectories(first[0], first[1], step, name="val_actions")
            except Exception:
                pass
        pol.train()
        return out

    @torch.no_grad()
    def _cot_faithfulness(self, batch: dict, noise, time, faith: dict) -> None:
        """Action-likelihood gain of the true trace over a mismatched one (another sample's trace,
        same noise and flow time): positive = the policy's actions depend on what the trace says.
        Reported overall and per phase (``subtask_phase_id``)."""
        perturbed = {**batch, "cot_input_ids": batch["cot_input_ids"].roll(1, 0), "cot_attention_mask": batch["cot_attention_mask"].roll(1, 0)}
        with self._amp():
            loss_p, _ = self.policy.forward(perturbed, noise=noise, time=time, reduction="none")
            loss_t, _ = self.policy.forward(batch, noise=noise, time=time, reduction="none")
        gain = (loss_p.float() - loss_t.float()).cpu()
        faith.setdefault("all", []).extend(gain.tolist())
        phases = batch.get("subtask_phase_id")
        names = getattr(self.policy.config, "phase_names", ())
        if phases is not None:
            for g, ph in zip(gain.tolist(), phases.reshape(-1).tolist(), strict=False):
                faith.setdefault(names[int(ph)] if int(ph) < len(names) else str(ph), []).append(g)

    def _log_generated_traces(self, step: int, n: int = 2) -> None:
        """Decode the reasoning text the policy generates for a few validation frames."""
        try:
            raw = next(iter(self.val_loader))
            batch = self._prep(raw)
            sub = {k: v[:n] if torch.is_tensor(v) and v.dim() > 0 else v for k, v in batch.items()}
            pol = self.policy
            images, img_masks = pol.prepare_images(sub)
            with self._amp():
                ids = pol.model.generate_cot(images, img_masks, sub[OBS_LANGUAGE_TOKENS], sub[OBS_LANGUAGE_ATTENTION_MASK],
                                             pol.prepare_state(sub), pol._objects(sub))
            tok = pol.model._tokenizer()
            ids = ids[0] if isinstance(ids, tuple) else ids
            if ids is None:
                return
            texts = [tok.decode(row, skip_special_tokens=True) for row in ids.cpu()]
            self.logger.log_text("cot/generated", texts, step)
        except Exception:
            import traceback

            traceback.print_exc()

    def _log_grad_diagnostics(self, batch: dict, step: int) -> None:
        """Gradient norm of each weighted loss term and the cosine between every pair.

        One extra forward on the last micro-batch; each term's gradient over all trainable
        parameters is kept on the CPU in fp16, so memory stays bounded.
        """
        if not hasattr(self.policy, "loss_terms"):
            return
        params = [p for p in self.policy.parameters() if p.requires_grad]
        try:
            with self._amp():
                self.policy.forward(batch, keep_loss_terms=True)
            terms = {k: v for k, v in self.policy.loss_terms.items() if v.requires_grad}
            grads = {}
            for i, (name, term) in enumerate(terms.items()):
                g = torch.autograd.grad(term, params, retain_graph=i < len(terms) - 1, allow_unused=True)
                flat = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).float() for x, p in zip(g, params, strict=True)])
                grads[name] = flat.half().cpu()
                del g, flat
            out = {f"grad/norm/{k}": float(v.float().norm()) for k, v in grads.items()}
            names = list(grads)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    ga, gb = grads[a].float(), grads[b].float()
                    denom = float(ga.norm() * gb.norm())
                    out[f"grad/cos/{a}__{b}"] = float(ga @ gb) / denom if denom > 0 else 0.0
            self._safe_log_scalars(out, step)
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            self.policy.loss_terms = {}
            self.optimizer.zero_grad(set_to_none=True)

    def _log_pseudolabels(self, sub: dict, step: int, n: int):
        """Frames of the current batch with their SAM 3 / TAPNext++ / depth / phase pseudo-labels."""
        if self.label_viz is None or self._last_ids is None or self.label_image_key not in sub:
            return
        eps, frames = self._last_ids
        if eps is None or frames is None:
            return
        imgs = sub[self.label_image_key][:n].float().clamp(0, 1).cpu()
        drawn = []
        for img, ep, fr in zip(imgs, torch.as_tensor(eps).reshape(-1)[:n].tolist(), torch.as_tensor(frames).reshape(-1)[:n].tolist(), strict=False):
            arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            if max(arr.shape[:2]) < 512:  # draw at 2x so the text stays legible
                from PIL import Image

                arr = np.array(Image.fromarray(arr).resize((arr.shape[1] * 2, arr.shape[0] * 2)))
            out = self.label_viz.overlay(int(ep), int(fr), arr)
            if out is not None:
                drawn.append(torch.from_numpy(out).permute(2, 0, 1).float() / 255)
        if drawn:
            self.logger.log_images(torch.stack(drawn), step, tag="pseudolabels", nrow=min(4, len(drawn)))

    @torch.no_grad()
    def _log_preview(self, batch: dict, step: int):
        import traceback

        n = 4
        bs = batch["action"].shape[0]
        sub = {k: v[:n] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bs else v for k, v in batch.items()}
        self.policy.eval()
        if hasattr(self.policy, "reset"):
            self.policy.reset()  # a new set of frames: no reasoning carried over
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
        try:
            self._log_pseudolabels(sub, step, n)
        except Exception:
            traceback.print_exc()
        finally:
            self.policy.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()