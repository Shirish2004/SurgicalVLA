"""Small neural-building-block helpers shared across modules."""

import torch
import torch.nn.functional as F
from torch import nn


class FP32Linear(nn.Linear):
    """``nn.Linear`` that always computes in float32, even under bf16 autocast.

    Used for the action output head and the MoE routers: both are small, sit at
    the end of a long bf16 chain, and lose measurable precision when their
    accumulation happens in bf16.
    """

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(
                x.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float(),
            )
