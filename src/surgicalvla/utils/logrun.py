"""Publication-style logging: scalars, images, MoE routing and action trajectories."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        SummaryWriter = None
try:
    from torchvision.utils import make_grid
except Exception:
    def make_grid(*args, **kwargs):
        raise RuntimeError("torchvision required for visual logging")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


class ExperimentLogger:
    """Writes scalars/images/figures to TensorBoard and saves vector-graphics copies to disk."""

    def __init__(self, output_dir: str, config=None, use_wandb: bool = False, project: str = "surgicalvla", **kwargs):
        self.fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(output_dir, "logs")) if SummaryWriter is not None else None
        self._history: dict[str, list] = {}

        if config is not None:
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config.__dict__, f, default=str, indent=2)

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                init_kwargs = dict(project=project, config=getattr(config, "__dict__", {}))
                # pass through common wandb init args if provided
                for k in ["entity", "group", "job_type", "tags", "notes", "dir", "id", "resume", "reinit", "mode"]:
                    if k in kwargs:
                        init_kwargs[k] = kwargs[k]
                self.wandb.init(**init_kwargs)
            except Exception:
                self.wandb = None

    def log_scalars(self, metrics: dict, step: int):
        for k, v in metrics.items():
            v = float(v)
            if self.writer is not None:
                self.writer.add_scalar(k, v, step)
            self._history.setdefault(k, []).append((step, v))
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def log_images(self, images: torch.Tensor, step: int, tag: str = "images", nrow: int = 4):
        if images.min() < 0:
            images = (images.clamp(-1, 1) + 1) / 2
        grid = make_grid(images, nrow=nrow)
        if self.writer is not None:
            self.writer.add_image(tag, grid, step)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"{tag} — step {step}")
        self._save(fig, f"{tag}_{step}")

    def log_expert_usage(self, usage: dict, step: int, tag: str = "moe/expert_usage"):
        for layer_idx, counts in usage.items():
            frac = counts / counts.sum().clamp_min(1)
            for e, val in enumerate(frac.tolist()):
                if self.writer is not None:
                    self.writer.add_scalar(f"{tag}/layer{layer_idx}/expert{e}", val, step)
        self._plot_expert_heatmap(usage, step)

    def _plot_expert_heatmap(self, usage: dict, step: int):
        layers = sorted(usage.keys())
        mat = np.stack([(usage[idx] / usage[idx].sum().clamp_min(1)).numpy() for idx in layers])
        fig, ax = plt.subplots(figsize=(1 + mat.shape[1] * 0.6, 1 + mat.shape[0] * 0.4))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("expert"), ax.set_ylabel("expert layer")
        ax.set_xticks(range(mat.shape[1])), ax.set_yticks(range(mat.shape[0]), layers)
        fig.colorbar(im, ax=ax, label="routing fraction")
        ax.set_title(f"Expert utilization — step {step}")
        if self.writer is not None:
            self.writer.add_figure("moe/expert_usage_heatmap", fig, step)
        self._save(fig, f"expert_usage_{step}")

    def log_action_trajectories(self, pred: torch.Tensor, target: torch.Tensor, step: int, dims: int | None = None, name: str = "actions"):
        pred, target = pred.numpy(), target.numpy()
        d = dims or pred.shape[-1]
        fig, axes = plt.subplots(d, 1, figsize=(6, 1.4 * d), sharex=True)
        axes = np.atleast_1d(axes)
        for i in range(d):
            axes[i].plot(target[0, :, i], label="ground truth", lw=1.5)
            axes[i].plot(pred[0, :, i], label="predicted", lw=1.5, ls="--")
            axes[i].set_ylabel(f"a[{i}]")
        axes[0].legend(frameon=False, ncol=2)
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Action chunk ({name}) — step {step}")
        fig.tight_layout()
        if self.writer is not None:
            self.writer.add_figure(f"rollout/{name}", fig, step)
        self._save(fig, f"{name}_{step}")

    def log_loss_curve(self, key: str = "loss"):
        if key not in self._history:
            return
        steps, values = zip(*self._history[key])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(steps, values, lw=1.2)
        ax.set_xlabel("step"), ax.set_ylabel(key), ax.set_title(key)
        self._save(fig, f"curve_{key}")

    def _save(self, fig, name: str):
        # fig.savefig(os.path.join(self.fig_dir, f"{name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(self.fig_dir, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)

    def close(self):
        self.log_loss_curve("loss")
        if self.writer is not None:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()