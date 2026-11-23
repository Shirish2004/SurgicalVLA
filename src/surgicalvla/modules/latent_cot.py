"""Latent chain of thought: learned query tokens that predict a future visual latent.

``K`` learned query tokens are appended to the VLM prefix (after the state and any text
CoT, with causal attention), so the action expert can attend to them like any other prefix
token. Their mean output is projected to the VLM width and trained to match the embedding of
the frame ``latent_cot_horizon`` steps ahead, produced by a target encoder that is an
exponential moving average of the VLA's vision encoder (stop-gradient). When the vision
encoder is frozen, the EMA of it is the encoder itself, so no copy is kept.

At inference the queries cost ``K`` prefix tokens and no decoding, which is what makes the
explicit-to-latent curriculum worthwhile.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LatentReasoner(nn.Module):
    def __init__(self, hidden: int, num_tokens: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden) * 0.02)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def tokens(self, batch_size: int) -> Tensor:
        return self.queries[None].expand(batch_size, -1, -1)

    def predict(self, token_outputs: Tensor) -> Tensor:
        return self.proj(token_outputs.mean(dim=1).to(self.proj[0].weight.dtype))

    @staticmethod
    def loss(prediction: Tensor, target: Tensor) -> Tensor:
        return (1 - F.cosine_similarity(prediction.float(), target.float().detach(), dim=-1)).mean()


class EMATargetEncoder(nn.Module):
    """Stop-gradient target embeddings of future frames from an EMA of the vision encoder."""

    def __init__(self, vlm_with_expert, decay: float = 0.99):
        super().__init__()
        self.decay = float(decay)
        get = getattr(vlm_with_expert, "get_vlm_model", None)
        vlm = get() if get is not None else None
        self._source = [vlm_with_expert]  # not a submodule: keeps the source out of this state dict
        trainable = vlm is not None and any(p.requires_grad for p in vlm.vision_model.parameters())
        if trainable:
            self.vision = copy.deepcopy(vlm.vision_model).requires_grad_(False)
            self.connector = copy.deepcopy(vlm.connector).requires_grad_(False)
        else:
            self.vision = self.connector = None

    @torch.no_grad()
    def update(self) -> None:
        if self.vision is None:
            return
        vlm = self._source[0].get_vlm_model()
        for ema, src in ((self.vision, vlm.vision_model), (self.connector, vlm.connector)):
            for pe, ps in zip(ema.parameters(), src.parameters(), strict=True):
                pe.mul_(self.decay).add_(ps.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, images: list[Tensor]) -> Tensor:
        """Mean token embedding over all cameras, ``(B, D_vlm)``."""
        embs = []
        for img in images:
            if self.vision is None:
                e = self._source[0].embed_image(img)
            else:
                h = self.vision(pixel_values=img.to(dtype=next(self.vision.parameters()).dtype)).last_hidden_state
                e = self.connector(h)
            embs.append(e.float().mean(dim=1))
        return torch.stack(embs).mean(dim=0)


__all__ = ["EMATargetEncoder", "LatentReasoner"]
