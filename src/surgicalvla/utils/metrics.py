import torch

BUCKETS = ((0, 10), (10, 25), (25, 50))


def _pair(x, m):
    return (x * m).sum(), m.sum()


def chunk_metrics(pred, target, is_pad=None, gripper_dims=()):
    t = min(pred.shape[1], target.shape[1])
    d = min(pred.shape[-1], target.shape[-1])
    pred = pred[:, :t, :d].float()
    target = target[:, :t, :d].float()
    dev = pred.device
    valid = torch.ones(pred.shape[:2], device=dev) if is_pad is None else (~is_pad[:, :t]).float()
    g = [i for i in gripper_dims if i < d]
    keep = torch.ones(d, dtype=torch.bool, device=dev)
    if g:
        keep[g] = False
    diff = pred - target
    arm = diff.abs()[..., keep].mean(-1)
    out = {
        "mae_arm": _pair(arm, valid),
        "mse_arm": _pair(diff[..., keep].pow(2).mean(-1), valid),
        "mae_per_step": ((arm * valid).sum(0), valid.sum(0)),
    }
    for lo, hi in BUCKETS:
        out[f"mae_h{lo}_{hi}"] = _pair(arm[:, lo:hi], valid[:, lo:hi])
    if g:
        out["mae_grip"] = _pair(diff.abs()[..., g].mean(-1), valid)
    if t >= 3:
        v3 = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]

        def sd(x):
            return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2])[..., keep].abs().mean(-1)

        out["jitter_pred"] = _pair(sd(pred), v3)
        out["jitter_gt"] = _pair(sd(target), v3)
    return out


class Accumulator:
    def __init__(self):
        self.s = {}
        self.n = {}

    def add(self, pairs):
        for k, (s, n) in pairs.items():
            s = s.detach().float().cpu()
            n = n.detach().float().cpu()
            self.s[k] = self.s[k] + s if k in self.s else s
            self.n[k] = self.n[k] + n if k in self.n else n

    def count(self, k):
        return float(self.n[k].sum()) if k in self.n else 0.0

    def result(self):
        res = {}
        for k in self.s:
            v = self.s[k] / self.n[k].clamp_min(1.0)
            res[k] = v.tolist() if v.dim() else v.item()
        if "jitter_pred" in res and "jitter_gt" in res:
            res["jitter_ratio"] = res["jitter_pred"] / max(res["jitter_gt"], 1e-8)
        return res