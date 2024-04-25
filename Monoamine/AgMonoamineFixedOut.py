import torch

from .Sign import Sign


# [barch,len] -> [batch,n_newlons]
class AgMonoamineFixedOut(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sign: Sign,
        pin: torch.Tensor,
        nin: torch.Tensor,
        pws: torch.Tensor,
        nws: torch.Tensor,
        pbs: torch.Tensor | None,
        nbs: torch.Tensor | None,
    ):
        ctx.sign = sign
        ctx.save_for_backward(pin, nin, pbs, nbs)

        if pbs is None or nbs is None:
            return sign * (pin * pws - nin * nws)
        else:
            return sign * ((pin * pws + pbs) - (nin * nws + nbs))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor):

        # print(f"fixed {grad_out=}")
        pin, nin, pbs, nbs = ctx.saved_tensors
        grad_out = grad_out * ctx.sign

        dp = grad_out
        dn = -grad_out

        dwp = dp.mean(dim=0, keepdim=True) * pin
        dwn = dn.mean(dim=0, keepdim=True) * nin

        if pbs is None or nbs is None:
            dbp = None
            dbn = None
        else:
            dbp = dp * pbs
            dbn = dn * nbs

        return None, dp, dn, dwp, dwn, dbp, dbn


def ag_monoamine_fixedout(
    sign: Sign,
    pin: torch.Tensor,
    nin: torch.Tensor,
    pws: torch.Tensor,
    nws: torch.Tensor,
    pbs: torch.Tensor | None,
    nbs: torch.Tensor | None,
):
    result = AgMonoamineFixedOut.apply(sign, pin, nin, pws, nws, pbs, nbs)
    return result


def main():
    batch_size = 1
    num_in = 3
    num_newlon = 3
    x = torch.rand((batch_size, num_in), requires_grad=True)
    x2 = torch.rand((batch_size, num_in), requires_grad=True)
    t = torch.rand((batch_size, num_newlon))

    pws = torch.rand(size=(1, num_newlon), requires_grad=True)
    cws = torch.rand(size=(1, num_newlon), requires_grad=True)
    pbs = torch.rand(size=(1, num_newlon), requires_grad=True)
    cbs = torch.rand(size=(1, num_newlon), requires_grad=True)

    pws2p = torch.rand(size=(1, num_newlon), requires_grad=True)
    cws2p = torch.rand(size=(1, num_newlon), requires_grad=True)
    pbs2p = torch.rand(size=(1, num_newlon), requires_grad=True)
    cbs2p = torch.rand(size=(1, num_newlon), requires_grad=True)

    pws2n = torch.rand(size=(1, num_newlon), requires_grad=True)
    cws2n = torch.rand(size=(1, num_newlon), requires_grad=True)
    pbs2n = torch.rand(size=(1, num_newlon), requires_grad=True)
    cbs2n = torch.rand(size=(1, num_newlon), requires_grad=True)
    # print(x)
    # print(x.repeat(1, num_newlon // num_in))

    yp = ag_monoamine_fixedout(Sign.plus, x, x2, pws2p, cws2p, pbs2p, cbs2p)
    yn = ag_monoamine_fixedout(Sign.minus, x, x2, pws2n, cws2n, pbs2n, cbs2n)
    y = ag_monoamine_fixedout(Sign.plus, yp, yn, pws, cws, pbs, cbs)

    print(f"{x.shape=}")
    print(x)
    print(f"{pws.shape=}")
    print(pws)
    print(f"{y.shape=}")
    print(y)
    print(f"{t.shape=}")
    print(t)

    loss = torch.nn.L1Loss()
    s = loss(y, t)
    s.backward()
    print(f"{s=}")
    print(f"{pws.grad=}")
    print(f"{cws.grad=}")
    print(f"{pws2p.grad=}")
    print(f"{cws2p.grad=}")
    print(f"{pws2n.grad=}")
    print(f"{cws2n.grad=}")
    print(f"{x.grad=}")
    print(f"{x2.grad=}")
    print()


if __name__ == "__main__":
    main()
