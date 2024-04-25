from typing import Callable

import torch
import torch.nn as nn

from .AgMonoamineFixedOut import ag_monoamine_fixedout
from .BypassAF import BypassAfLayer
from .Sign import Sign


class MonoamineFixedLayer(nn.Module):
    def __init__(
        self,
        sign: Sign,
        num_in: int,
        num_newlon: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:

        # assert (
        #     num_newlon % num_in == 0
        # ), f"num_newlon % num_in must to zero. {num_newlon=},{num_in=}"

        super().__init__()

        pws = torch.rand(size=(1, num_newlon))
        nws = torch.rand(size=(1, num_newlon))

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

    def forward(self, pin: torch.Tensor, nin: torch.Tensor):
        y = ag_monoamine_fixedout(
            self.sign, pin, nin, self.pws, self.nws, self.pbs, self.nbs
        )
        if self.actf:
            y = self.actf(y)
        return y


# -*- coding: utf-8 -*-
def main():
    input_num = 4
    n_num = 4
    batch_size = 1
    sign = Sign.plus

    x = torch.rand((batch_size, input_num))
    t = torch.rand((batch_size, n_num))
    print(x.shape)
    print(x)

    lossf = nn.L1Loss()
    model = MonoamineFixedLayer(sign, input_num, n_num)
    model.train()

    y = model(x, x)

    print(y.shape)
    print(y)

    loss = lossf(y, t)
    loss.backward()


if __name__ == "__main__":
    main()
