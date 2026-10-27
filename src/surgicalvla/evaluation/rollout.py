"""Policy rollouts and offline evaluation against recorded episodes."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from surgicalvla.evaluation.metrics import action_error


def rollout_policy(
    model,
    dataloader: DataLoader,
    normalizer=None,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Replay predictions over a held-out dataloader.

    Runs the policy in eval mode (no gradients), collecting predicted and
    reference action chunks for metric computation.

    Returns:
        A dict with ``predicted``, ``target`` and optionally ``episode_index``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets, eps = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            obs_key = "image" if "image" in batch else "state"
            observation = batch[obs_key].to(device)
            action = batch["action"].to(device)
            pred = model(observation)
            if normalizer is not None:
                pred = normalizer.unnormalize(pred)
            preds.append(pred.cpu())
            targets.append(action.cpu())
            if "episode_index" in batch:
                eps.append(batch["episode_index"].cpu())

    out = {"predicted": torch.cat(preds, dim=0), "target": torch.cat(targets, dim=0)}
    if eps:
        out["episode_index"] = torch.cat(eps, dim=0)
    return out


def evaluate(model, dataloader: DataLoader, normalizer=None, device: str | None = None) -> dict[str, float]:
    """Convenience wrapper computing metrics over a full epoch."""
    rollout = rollout_policy(model, dataloader, normalizer=normalizer, device=device)
    if not (rollout["predicted"].shape == rollout["target"].shape):
        raise ValueError(
            f"Predicted {rollout['predicted'].shape} vs target {rollout['target'].shape} "
            "shapes are incompatible; cannot evaluate."
        )
    return action_error(rollout["predicted"], rollout["target"])