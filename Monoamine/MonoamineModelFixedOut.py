import math

import torch
import torch.nn as nn

from .MonoamineFixedLayer import MonoamineFixedLayer
from .Sign import Sign


class MonoamineModel(nn.Module):
    def __init__(
        self,
        sign: Sign,
        num_in: int,
        num_out: int,
        num_hiddens: list[int],
        sep_hidden: int = 8,
    ) -> None:
        super().__init__()

        assert (
            num_in * num_out
        ) % sep_hidden == 0, f"(num_in * num_out) % sep_hidden must to be 0. {num_in*num_out=},{sep_hidden=}"

        assert sep_hidden % 2 == 0, f"sep_hidden must to be 2^n. {sep_hidden=}"

        hidden = num_in * num_out
        n_hidden_layers = int(math.log2(sep_hidden))

        self.lip = MonoamineFixedLayer(Sign.plus, num_in, hidden)
        self.lin = MonoamineFixedLayer(Sign.minus, num_in, hidden)

        hidden_p = hidden
        for n_hidden_layer in range(n_hidden_layers):
            lhs = []

            hidden = hidden_p / 2

            lhs.append(MonoamineFixedLayer(Sign.plus, hidden, hidden))
            lhs.append(MonoamineFixedLayer(Sign.plus, hidden, hidden))

    def forward(self, x: torch.Tensor):
        p1 = self.lip(x, x)
        n1 = self.lin(x, x)
        p2 = self.l2p(p1, n1)
        pass
