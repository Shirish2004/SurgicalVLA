"""Training-only auxiliary vision heads on the VLA's own hidden states.

The heads read the mean action-expert output (``aux_source="expert"``, the default) or the
mean-pooled VLM prefix (``"prefix"``) and predict label-free pseudo-targets produced offline
(``surgicalvla.pseudolabels``). With the VLM frozen, only the expert source lets the targets
shape features the policy acts on; the prefix source reaches only the trainable prefix inputs
(object projector, state projection, latent queries).

* ``feat``:   the target object's mask-pooled DINOv2 feature (cosine loss);
* ``motion``: the target's image-plane displacement to the next labelled frame;
* ``track``:  the mean displacement of the target's visible point tracks over the horizon;
* ``rel``:    target minus gripper (image offset and relative depth).

Every loss is weighted by the pseudo-label's confidence, so frames the labeler was unsure
about contribute little. The heads are not used at inference, so they cost nothing there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TARGETS = {"feat": None, "motion": 2, "track": 2, "rel": 3}


def weighted_mean(loss: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


class AuxVisionHeads(nn.Module):
    def __init__(self, hidden: int, feat_dim: int, width: int = 512, motion_scale: float = 10.0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, feat_dim if dim is None else dim) for name, dim in TARGETS.items()}
        )
        # displacements are a few hundredths of the image; scale them to O(1) for the loss
        self.motion_scale = float(motion_scale)

    def forward(self, pooled: Tensor) -> dict[str, Tensor]:
        h = self.trunk(pooled.to(self.trunk[0].weight.dtype))
        return {name: head(h) for name, head in self.heads.items()}

    def losses(self, pooled: Tensor, aux: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-target confidence-weighted losses for the targets present in ``aux``."""
        pred = self(pooled)
        out: dict[str, Tensor] = {}
        if "aux.obj_feat" in aux:
            target = aux["aux.obj_feat"].to(pred["feat"])
            cos = F.cosine_similarity(pred["feat"].float(), target.float(), dim=-1)
            out["aux_feat_loss"] = weighted_mean(1 - cos, aux["aux.obj_feat_conf"])
        for name, key in (("motion", "aux.obj_motion"), ("track", "aux.track_disp"), ("rel", "aux.rel_disp")):
            if key in aux:
                target = aux[key].to(pred[name]).float()
                scale = self.motion_scale if name != "rel" else 1.0
                l1 = F.smooth_l1_loss(pred[name].float(), target * scale, reduction="none").mean(-1)
                out[f"aux_{name}_loss"] = weighted_mean(l1, aux[f"{key}_conf"])
        return out


__all__ = ["AuxVisionHeads", "weighted_mean"]
