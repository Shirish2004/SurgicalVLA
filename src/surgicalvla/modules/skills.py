"""Unsupervised skill discovery + the chunk-level routing prior.

Two things live here:

* :class:`SkillClusterer` — fits k-means over hand-crafted kinematic descriptors of
  an action chunk and exposes ``labels(actions)``. These labels are the *only*
  supervision the router gets, so they define what a "skill" means in this repo.
  They are kinematic by construction: the descriptors are computed from the
  action trajectory alone, so the clusters cannot distinguish two visually
  different tasks that happen to move the arm the same way. See the README.
* :class:`PhaseHead` — a small cross-attention head over the vision-language
  prefix. Its output logits are the chunk-level phase prior added to every
  token router inside the action expert.

Nothing here needs LeRobot, so it is cheap to unit-test.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

NUM_FEATURES = 8


def _sync(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    return t


def _bcast(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, 0)
    return t


class SkillClusterer(nn.Module):
    """Online k-means over chunk kinematics, with EMA centroids and dead-cluster repair."""

    def __init__(
        self,
        num_clusters: int,
        action_dim: int,
        gripper_dims=(),
        ema: float = 0.99,
        freeze_steps: int = 2000,
    ):
        super().__init__()
        self.k = num_clusters
        self.action_dim = action_dim
        self.gripper_dims = tuple(gripper_dims)
        self.ema = ema
        self.freeze_steps = freeze_steps
        self.register_buffer("centroids", torch.zeros(num_clusters, NUM_FEATURES))
        self.register_buffer("usage", torch.ones(num_clusters))
        self.register_buffer("feat_mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("feat_var", torch.ones(NUM_FEATURES))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    def features(self, actions, is_pad=None) -> torch.Tensor:
        """(B, T, D) actions -> (B, 8) kinematic descriptors.

        Mean / max / std arm speed, speed trend within the chunk, path
        straightness, jerkiness, and gripper activity / net travel.
        """
        a = actions[..., : self.action_dim].float()
        b, t, d = a.shape
        dev = a.device
        if t < 2:
            return torch.zeros(b, NUM_FEATURES, device=dev, dtype=a.dtype)
        valid = torch.ones(b, t, dtype=torch.bool, device=dev) if is_pad is None else ~is_pad
        g = [i for i in self.gripper_dims if i < d]
        arm = a[..., [i for i in range(d) if i not in g]]
        pair = (valid[:, 1:] & valid[:, :-1]).float()
        n = pair.sum(1).clamp_min(1.0)
        s = (arm[:, 1:] - arm[:, :-1]).pow(2).mean(-1).sqrt()
        mean_s = (s * pair).sum(1) / n
        max_s = (s * pair).max(1).values
        std_s = (((s - mean_s[:, None]).pow(2) * pair).sum(1) / n).sqrt()
        pos = torch.arange(t - 1, device=dev)[None]
        first = (pos < n[:, None] / 2).float() * pair
        second = pair - first
        trend = (s * second).sum(1) / second.sum(1).clamp_min(1.0) - (s * first).sum(1) / first.sum(
            1
        ).clamp_min(1.0)
        pp = pair[:, 1:] * pair[:, :-1]
        jerk = ((s[:, 1:] - s[:, :-1]).abs() * pp).sum(1) / pp.sum(1).clamp_min(1.0)
        last = (valid.sum(1) - 1).clamp_min(0)
        ar = torch.arange(b, device=dev)
        net = (arm[ar, last] - arm[:, 0]).pow(2).mean(-1).sqrt()
        straight = net / ((s * pair).sum(1) + 1e-6)
        if g:
            ga = a[..., g]
            gv = (ga[:, 1:] - ga[:, :-1]).pow(2).mean(-1).sqrt()
            gact = (gv * pair).sum(1) / n
            gnet = (ga[ar, last] - ga[:, 0]).abs().mean(-1)
        else:
            gact = gnet = torch.zeros(b, device=dev)
        return torch.stack([mean_s, max_s, std_s, trend, straight, jerk, gact, gnet], dim=-1)

    def standardize(self, f: torch.Tensor) -> torch.Tensor:
        return (f - self.feat_mean) / self.feat_var.clamp_min(1e-6).sqrt()

    @torch.no_grad()
    def fit(self, f: torch.Tensor, iters: int = 30, warm: bool = False):
        """k-means++ init (or warm start) then Lloyd iterations.

        ``iters=0`` is legal and means "assign with the current centroids".
        """
        self.feat_mean.copy_(f.mean(0))
        self.feat_var.copy_(f.var(0, unbiased=False).clamp_min(1e-6))
        z = self.standardize(f)
        if warm and bool(self.initialized):
            c = self.centroids.clone()
        else:
            c = z[torch.randint(0, z.shape[0], (1,), device=z.device)]
            for _ in range(self.k - 1):
                d = torch.cdist(z, c).min(1).values.pow(2) + 1e-9
                c = torch.cat([c, z[torch.multinomial(d, 1)]])
        cnt = torch.ones(self.k, device=z.device)
        for _ in range(max(0, iters)):
            oh = F.one_hot(torch.cdist(z, c).argmin(1), self.k).float()
            cnt = oh.sum(0)
            c = torch.where((cnt > 0)[:, None], oh.t() @ z / cnt.clamp_min(1.0)[:, None], c)
        oh = F.one_hot(torch.cdist(z, c).argmin(1), self.k).float()
        cnt = oh.sum(0)
        self.centroids.copy_(c)
        self.usage.copy_(cnt / cnt.sum().clamp_min(1.0))
        for t in (self.centroids, self.feat_mean, self.feat_var, self.usage):
            _bcast(t)
        self.initialized.fill_(True)
        self.frozen.fill_(True)

    @torch.no_grad()
    def update(self, f: torch.Tensor):
        if not bool(self.initialized):
            self.fit(f, iters=5)
            self.frozen.fill_(False)
            return
        ema = self.ema
        z = self.standardize(f)
        oh = F.one_hot(torch.cdist(z, self.centroids).argmin(-1), self.k).float()
        sums = _sync(oh.t() @ z)
        counts = _sync(oh.sum(0))
        present = counts > 0
        target = sums / counts.clamp_min(1.0).unsqueeze(-1)
        self.centroids[present] = ema * self.centroids[present] + (1 - ema) * target[present]
        self.usage.mul_(ema).add_((1 - ema) * counts / counts.sum().clamp_min(1.0))
        dead = self.usage < 0.1 / self.k
        if bool(dead.any()):
            repl = z[torch.randint(0, z.shape[0], (int(dead.sum()),), device=z.device)]
            self.centroids[dead] = repl + 0.05 * torch.randn_like(repl)
            self.usage[dead] = 1.0 / self.k
        _bcast(self.centroids)

    @torch.no_grad()
    def labels(self, actions, is_pad=None, train: bool = False):
        """Cluster ids for each sample, or ``None`` before the first fit."""
        f = self.features(actions, is_pad)
        if train:
            if not bool(self.frozen) and int(self.steps) < self.freeze_steps:
                self.update(f)
            self.steps += 1
        if not bool(self.initialized):
            return None
        return torch.cdist(self.standardize(f), self.centroids).argmin(-1)

    def freeze(self):
        self.frozen.fill_(True)

    def soft_counts(self) -> torch.Tensor:
        """Cluster occupancy over the most recent ``update`` (diagnostic only)."""
        return self.usage

    def reset(self):
        """Forget everything — for tests and for re-fitting on a new dataset."""
        self.centroids.zero_()
        self.usage.fill_(1.0)
        self.feat_mean.zero_()
        self.feat_var.fill_(1.0)
        self.initialized.fill_(False)
        self.steps.zero_()
        self.frozen.fill_(False)


class PhaseHead(nn.Module):
    """Vision-language prefix -> chunk-level phase prior over skills.

    Cross-attends a handful of learned queries over the prefix embeddings and
    scores them against learned skill prototypes. Training this head is what
    makes routing scene- and instruction-dependent; the routers alone only see
    action-expert hidden states.
    """

    def __init__(self, in_dim: int, num_experts: int, dim: int = 256, queries: int = 4, heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, dim)
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Sequential(nn.Linear(queries * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.protos = nn.Parameter(torch.randn(num_experts, dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, embs: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """(B, T, D) prefix embeddings + (B, T) pad mask -> (B, E) logits."""
        x = F.layer_norm(embs.to(self.proj.weight.dtype), (self.in_dim,))
        x = self.proj(x)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        h, _ = self.attn(q, x, x, key_padding_mask=~pad)
        h = self.mix(self.norm(h).flatten(1))
        return self.log_scale.exp().clamp(max=30.0) * F.normalize(h, dim=-1) @ F.normalize(self.protos, dim=-1).t()
