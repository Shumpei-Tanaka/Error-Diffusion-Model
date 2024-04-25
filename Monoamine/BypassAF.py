from typing import Callable

import torch


class BypassAF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        activate: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ):
        return activate(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor):
        return None, grad_out


def bypass_af(
    activate: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
):
    result = BypassAF.apply(activate, x)
    return result


class BypassAfLayer(torch.nn.Module):
    def __init__(self, activate: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.activate = activate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return bypass_af(self.activate, input)
