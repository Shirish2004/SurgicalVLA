
import torch
from torch import nn


class RouteState:
    def __init__(self):
        self.weights = None


class SlicedExpert(nn.Module):
    def __init__(self, dense: nn.Module, ratio: float):
        super().__init__()
        ref = dense.gate_proj.weight
        inter, hidden = ref.shape
        width = max(16, int(inter * ratio))
        pick = torch.randperm(inter)[:width].to(ref.device)
        kw = dict(bias=False, dtype=ref.dtype, device=ref.device)
        self.gate_proj = nn.Linear(hidden, width, **kw)
        self.up_proj = nn.Linear(hidden, width, **kw)
        self.down_proj = nn.Linear(width, hidden, **kw)
        with torch.no_grad():
            self.gate_proj.weight.copy_(dense.gate_proj.weight[pick])
            self.up_proj.weight.copy_(dense.up_proj.weight[pick])
            self.down_proj.weight.zero_()
        self.act = dense.act_fn

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class PhaseMoEFFN(nn.Module):
    def __init__(self, dense: nn.Module, route: RouteState, num_experts: int, ratio: float):
        super().__init__()
        self.shared = dense
        self.experts = nn.ModuleList(SlicedExpert(dense, ratio) for _ in range(num_experts))
        self.route = route

    def forward(self, x):
        out = self.shared(x)
        w = self.route.weights
        if w is None:
            return out
        w = w.to(out.dtype)
        for e, expert in enumerate(self.experts):
            rows = (w[:, e] > 0).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            out = out.index_add(0, rows, expert(x[rows]) * w[rows, e].view(-1, 1, 1))
        return out


def upcycle_to_moe(vlm_with_expert, route: RouteState, num_experts: int, ratio: float, every_n: int):
    for i, layer in enumerate(vlm_with_expert.lm_expert.layers):
        if i % every_n == every_n - 1:
            layer.mlp = PhaseMoEFFN(layer.mlp, route, num_experts, ratio)
    return vlm_with_expert


def moe_layers(vlm_with_expert):
    return [layer.mlp for layer in vlm_with_expert.lm_expert.layers if isinstance(layer.mlp, PhaseMoEFFN)]