from typing import Callable

import torch
import torch.nn as nn

from .MonoamineFixedLayer import MonoamineFixedLayer
from .MonoamineLayer import MonoamineLayer
from .Sign import Sign


class MonoamineDualLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_newlon: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:

        # assert (
        #     num_in % num_newlon == 0
        # ), f"num_in % num_newlon must to zero. {num_in=},{num_newlon=}"

        super().__init__()

        self.pl = MonoamineLayer(Sign.plus, num_in, num_newlon, bias, activate)
        self.nl = MonoamineLayer(Sign.minus, num_in, num_newlon, bias, activate)

    def forward(self, x: tuple[torch.Tensor]):
        py = self.pl(*x)
        ny = self.nl(*x)
        return py, ny


class MonoamineFixedDualLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_newlon: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:

        assert num_in == num_newlon, f"{num_in=},{num_newlon=}"

        super().__init__()

        self.pl = MonoamineFixedLayer(Sign.plus, num_in, num_newlon, bias, activate)
        self.nl = MonoamineFixedLayer(Sign.minus, num_in, num_newlon, bias, activate)

    def forward(self, x: tuple[torch.Tensor]):
        py = self.pl(*x)
        ny = self.nl(*x)
        return py, ny


class MonoamineFixedDual2MidLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:
        """make dual Monoamin layer for middle layer
        num of outputs is harf of inputs

        Args:
            num_in (int): num of inputs
            bias (bool, optional): enable bias. Defaults to True.
            activate (Callable[[torch.Tensor], torch.Tensor] | None, optional): activate function. Defaults to nn.ReLU().
        """
        assert num_in % 2 == 0, f"{num_in=}"

        super().__init__()

        num_newlon = num_in // 2

        self.pl = MonoamineFixedLayer(Sign.plus, num_newlon, num_newlon, bias, activate)
        self.nl = MonoamineFixedLayer(
            Sign.minus, num_newlon, num_newlon, bias, activate
        )

    def forward(self, x: tuple[torch.Tensor]):
        px, nx = x
        ppx, npx = torch.chunk(px, 2, 1)
        pnx, nnx = torch.chunk(nx, 2, 1)

        py = self.pl(ppx, pnx)
        ny = self.nl(npx, nnx)

        return py, ny


class MonoamineDual2OutLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_newlon: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:

        # assert (
        #     num_in % num_newlon == 0
        # ), f"num_in % num_newlon must to zero. {num_in=},{num_newlon=}"
        super().__init__()
        self.pl = MonoamineLayer(Sign.plus, num_in, num_newlon, bias, activate)

    def forward(self, x: tuple[torch.Tensor]):
        py = self.pl(*x)
        return py


class MonoamineDual2MultiOutLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_outs: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:
        """make dual Monoamine layer for multi output
        num_in % num_outs must to zero

        Args:
            num_in (int): num of input
            num_outs (int): num of output
            bias (bool, optional): enable bias. Defaults to True.
            activate (Callable[[torch.Tensor], torch.Tensor] | None, optional): activate function. Defaults to nn.ReLU().
        """
        assert (
            num_in % num_outs == 0
        ), f"num_in % num_outs must to zero. {num_in=},{num_outs=}"

        super().__init__()

        num_in_seped = num_in // num_outs

        self.pls = nn.ModuleList(
            [
                MonoamineLayer(Sign.plus, num_in_seped, 1, bias, activate=None)
                for i in range(num_outs)
            ]
        )
        self.num_outs = num_outs
        self.activate = activate

    def forward(self, x: tuple[torch.Tensor]):
        px, nx = x
        pxs = px.chunk(self.num_outs, dim=1)
        nxs = nx.chunk(self.num_outs, dim=1)
        pys = [pl(px, nx) for pl, px, nx in zip(self.pls, pxs, nxs)]
        py = torch.cat(pys, dim=-1)
        if self.activate:
            py = self.activate(py)
        return py


class MonoamineFixedDual2RepInputLayer(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_outs: int,
        bias: bool = True,
        activate: Callable[[torch.Tensor], torch.Tensor] | None = nn.ReLU(),
    ) -> None:
        """Duplicate inputs
        num_outs % num_in must to zero.
        Args:
            num_in (int): num of inputs
            num_outs (int): num of outs
            bias (bool, optional): enable bias. Defaults to True.
            activate (Callable[[torch.Tensor], torch.Tensor] | None, optional): activate function. Defaults to nn.ReLU().
        """
        assert (
            num_outs % num_in == 0
        ), f"num_outs % num_in must to zero. {num_outs=},{num_in=}"

        super().__init__()

        self.l = MonoamineFixedDualLayer(num_outs, num_outs, bias, activate)
        self.xrep = num_outs // num_in

    def forward(self, x: tuple[torch.Tensor]):
        px, nx = x
        _px = px.repeat(1, self.xrep)
        _nx = nx.repeat(1, self.xrep)
        y = self.l((_px, _nx))
        return y


# -*- coding: utf-8 -*-
def main():
    input_num = 16
    n_num = 32
    out_num = 4
    batch_size = 5

    x = torch.rand((batch_size, input_num))
    t = torch.rand((batch_size, out_num))
    print(x.shape)
    print(x)

    lossf = nn.L1Loss()
    model = nn.Sequential(
        MonoamineFixedDual2RepInputLayer(input_num, n_num),
        MonoamineFixedDualLayer(n_num, n_num),
        MonoamineDual2MultiOutLayer(n_num, out_num, activate=nn.Softmax(dim=-1)),
    )
    model.train()

    y = model((x, x))

    print(y.shape)
    print(y)

    loss = lossf(y, t)
    loss.backward()


if __name__ == "__main__":
    main()
