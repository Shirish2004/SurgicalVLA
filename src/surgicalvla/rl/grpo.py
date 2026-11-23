"""Group-relative advantages and the frozen reference for the online-RL stage.

This is a *GRPO-style* algorithm, and the honest description is "group-relative
advantage weighting of the flow-matching loss" rather than "GRPO":

* An episode generates one **group** of sampled action chunks (one per decision
  point), all of which inherit the episode's reward. Advantages are computed by
  comparing each episode against the *other* episodes of its rollout group
  (:func:`group_advantages`), so no value function is needed.
* The policy update re-uses the supervised flow-matching loss, weighting each
  sample by its advantage (``SurgicalVLAPolicy.forward(..., weights=...)``).
  Successful chunks are imitated more strongly, failed ones less.
* There is **no importance ratio, no clipping and no PPO surrogate objective**.
  A flow-matching action head has no tractable per-action log-probability, so
  the likelihood-ratio machinery that makes GRPO/PPO what they are simply does
  not exist here. Calling the result "GRPO" would be overclaiming; see the
  README section "Online RL" for what is and is not proven.
* The optional consistency term (:class:`FrozenReference`) is a **flow-loss gap
  against a frozen copy** of the starting checkpoint, not a KL divergence in
  action space. It is off by default because the frozen copy costs a full extra
  model in GPU memory.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch


def group_advantages(
    rewards: Sequence[float],
    *,
    normalize: str = "rloo",
    clip: float = 5.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Turn one group of episode rewards into leave-one-out advantages.

    For group member *i* with reward :math:`r_i`, the baseline is the mean
    reward of the *other* members:

    .. math:: A_i = r_i - \\frac{1}{N-1}\\sum_{j \\neq i} r_j

    This is the RLOO estimator. It is used instead of GRPO's z-score because it
    needs no standard deviation: with binary rewards a group where only one
    member succeeds has zero reward variance among the *others*, so the z-score
    divides by ``eps`` and saturates the clip (one sample at ``+5``, the rest at
    ``-0.4``). Since the weighted loss below divides by ``sum(|w|)``, only the
    *ratio* between positive and negative weights matters, and RLOO's ``+1`` vs
    ``-1/(N-1)`` is both well-conditioned and interpretable.

    Args:
        rewards: one reward per episode in the group.
        normalize: ``"rloo"`` (default) or ``"std"`` for the GRPO z-score
            ``(r - mean) / (std + eps)`` computed over the group. ``"std"`` is
            only applied when the group has at least three members; smaller
            groups always fall back to ``"rloo"`` because a 1- or 2-member
            baseline has no usable spread.
        clip: symmetric clip on the advantages, in units of the reward scale.
        eps: additive guard for the ``"std"`` normalization.

    Returns:
        Float tensor of shape ``(N,)``. A group of one has all-zero advantages
        (no baseline to compare against), which makes that iteration a no-op
        update rather than a spurious one.
    """
    r = torch.as_tensor(list(rewards), dtype=torch.float64).reshape(-1)
    n = r.numel()
    if n == 0:
        return r.to(torch.float32)
    if n == 1:
        return torch.zeros(1, dtype=torch.float32)

    total = r.sum()
    others_sum = total - r
    others_mean = others_sum / (n - 1)
    adv = r - others_mean

    if normalize == "std":
        if n >= 3:
            # population variance of the n-1 other rewards, from running sums
            others_sq = (r * r).sum() - r * r
            var = (others_sq - (others_sum * others_sum) / (n - 1)) / (n - 1)
            adv = adv / (var.clamp_min(0.0).sqrt() + eps)
    elif normalize != "rloo":
        raise ValueError(f"normalize must be 'rloo' or 'std', got {normalize!r}")

    if clip is not None and clip > 0:
        adv = adv.clamp(-float(clip), float(clip))
    return adv.to(torch.float32)


def group_weights(
    rewards: Sequence[float],
    *,
    normalize: str = "rloo",
    clip: float = 5.0,
    fallback: bool = True,
) -> tuple[torch.Tensor, str]:
    """Per-episode loss weights plus the name of the regime that produced them.

    Returns ``(weights, mode)`` where ``mode`` is one of:

    ``"advantage"``
        The group had reward variation, so the weights are
        :func:`group_advantages` -- the intended GRPO-style update.
    ``"rejection"``
        *Every* episode got the same reward but at least one succeeded. RLOO has
        nothing to compare against (all advantages are zero) yet the episodes
        that succeeded are still the ones worth imitating, so the weights fall
        back to the raw rewards: this is filtered behaviour cloning / rejection
        sampling, which is exactly the cold-start regime of VLA RL. Without this
        fallback an all-success group would silently produce no gradient.
    ``"none"``
        No episode succeeded and there is no variation: nothing to learn from.
        The caller should skip the update.

    The fallback is what makes the first iterations do anything at all -- a
    pretrained checkpoint dropped into a new simulator usually succeeds rarely,
    so ``mode`` is worth logging, and a long stretch of ``"none"`` means the
    task is out of reach of the initial policy.
    """
    adv = group_advantages(rewards, normalize=normalize, clip=clip)
    if float(adv.abs().sum()) > 1e-6:
        return adv, "advantage"
    if not fallback:
        return adv, "none"
    r = torch.as_tensor(list(rewards), dtype=torch.float32).reshape(-1)
    if float(r.sum()) > 0:
        return r, "rejection"
    return adv, "none"


def episode_reward(
    success: bool,
    steps: int,
    horizon: int | None = None,
    efficiency_bonus: float = 0.0,
) -> float:
    """Scalar reward for one episode: ``1`` on success, ``0`` otherwise.

    ``efficiency_bonus`` optionally rewards finishing early, scaled by the unused
    fraction of the episode: ``bonus * (1 - steps / horizon)``. It is added to
    the success term only -- giving a partial bonus for a failed episode that
    merely terminated early would teach the policy to give up quickly.

    ``steps``/``horizon`` are passed even when ``efficiency_bonus`` is zero so
    that turning the term on does not change the call sites.
    """
    reward = 1.0 if success else 0.0
    if reward and efficiency_bonus and horizon and horizon > 0:
        reward += float(efficiency_bonus) * max(0.0, 1.0 - float(steps) / float(horizon))
    return float(reward)


class FrozenReference:
    """A frozen copy of the policy, used only for the flow-loss-gap penalty.

    Construct lazily (``train_rl`` builds it on the first iteration where
    ``kl_coef > 0``) because the copy doubles the resident model memory: the
    SmolVLA backbone plus experts is several GB on a 20 GB card.

    The penalty is the *difference in supervised flow loss* between the current
    policy and the frozen one on the same sampled batch: if the policy drifts far
    from the reference, its ability to denoise that chunk gets worse and the gap
    grows. It is a proxy for divergence, not a KL -- see the module docstring.
    """

    def __init__(self, policy, device: str | torch.device | None = None):
        self.policy = copy.deepcopy(policy)
        if device is not None:
            self.policy.to(device)
        self.policy.eval()
        self.policy.requires_grad_(False)

    @torch.no_grad()
    def flow_loss(self, batch: dict) -> float:
        """Reference flow loss on ``batch`` (same reduction ``policy.forward`` uses)."""
        _, loss_dict = self.policy.forward(batch)
        return float(loss_dict["losses_after_rm_padding"])

    def __call__(self, batch: dict) -> float:
        return self.flow_loss(batch)


def kl_penalty(current_loss: float, reference_loss: float, coef: float) -> float:
    """One-sided flow-loss-gap penalty ``coef * relu(current - reference)``.

    One-sided because moving the loss *below* the reference is the direction we
    want: only an increase is penalized. This mirrors the truncated-KL style of
    trust-region methods without pretending to compute a KL.
    """
    if not coef:
        return 0.0
    return float(coef) * max(0.0, float(current_loss) - float(reference_loss))
