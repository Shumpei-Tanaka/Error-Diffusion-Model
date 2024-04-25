from typing import Callable

import torch
import torch.nn as nn

from .AgMonoamine import ag_monoamine
from .BypassAF import BypassAfLayer
from .Sign import Sign


class MonoamineLayer(nn.Module):
    def __init__(
        self,
        sign: Sign,
        num_in: int,
        num_newlon: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:
        # assert (
        #     num_in % num_newlon == 0
        # ), f"num_in % num_newlon must to zero. {num_in=},{num_newlon=}"

        super().__init__()

        pws = torch.rand(size=(num_in, num_newlon))
        nws = torch.rand(size=(num_in, num_newlon))

        self.sign = sign

        self.pws = nn.Parameter(pws)
        self.nws = nn.Parameter(nws)
        if bias:
            self.pbs = nn.Parameter(torch.rand(size=(1, num_newlon)))
            self.nbs = nn.Parameter(torch.rand(size=(1, num_newlon)))
        else:
            self.pbs = None
            self.nbs = None

        if not activate is None:
            self.actf = BypassAfLayer(activate)
        else:
            self.actf = None

    def forward(
        self,
        pin: torch.Tensor,
        nin: torch.Tensor,
    ):
        y = ag_monoamine(self.sign, pin, nin, self.pws, self.nws, self.pbs, self.nbs)
        if self.actf:
            y = self.actf(y)
        return y


# -*- coding: utf-8 -*-
def main():
    input_num = 16
    n_num = 16
    batch_size = 1
    sign = Sign.plus

    x = torch.rand((batch_size, input_num))
    t = torch.rand((batch_size, n_num))
    print(x.shape)
    print(x)

    lossf = nn.L1Loss()
    model = MonoamineLayer(sign, input_num, n_num, activate=nn.Softmax(dim=-1))
    model.train()

    y = model(x, x)

    print(y.shape)
    print(y)

    loss = lossf(y, t)
    loss.backward()


if __name__ == "__main__":
    main()
