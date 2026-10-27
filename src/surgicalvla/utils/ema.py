from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, module, decay: float = 0.999):
        self.module = module
        self.decay = decay
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in self.module.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @contextmanager
    def apply(self):
        backup = {}
        with torch.no_grad():
            for n, p in self.module.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.module.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def state_dict(self):
        return {"shadow": self.shadow, "updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state.get("updates", 0)
        for n, v in state["shadow"].items():
            if n in self.shadow:
                self.shadow[n].copy_(v.to(self.shadow[n].device))