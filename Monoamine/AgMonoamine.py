import torch

from .Sign import Sign


# [barch,len] -> [batch,n_newlons]
class AgMonoamine(torch.autograd.Function):
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
            return sign * (pin @ pws - nin @ nws)
        else:
            return sign * ((pin @ pws + pbs) - (nin @ nws + nbs))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor):
        # print(f"{grad_out.T=}")
        grad_out = grad_out * ctx.sign
        pin, nin, pbs, nbs = ctx.saved_tensors

        dp = grad_out
        dn = -grad_out

        dwp = (pin.T @ dp) / grad_out.shape[0]
        dwn = (nin.T @ dn) / grad_out.shape[0]

        dip = dp.expand((-1, pin.shape[1]))
        din = dn.expand((-1, nin.shape[1]))

        if pbs is None or nbs is None:
            dbp = None
            dbn = None
        else:
            dbp = dp * pbs
            dbn = dn * nbs

        return None, dip, din, dwp, dwn, dbp, dbn


def ag_monoamine(
    sign: Sign,
    pin: torch.Tensor,
    nin: torch.Tensor,
    pws: torch.Tensor,
    nws: torch.Tensor,
    pbs: torch.Tensor | None,
    nbs: torch.Tensor | None,
):
    result = AgMonoamine.apply(sign, pin, nin, pws, nws, pbs, nbs)
    return result


def main():
    batch_size = 2
    num_in = 4
    num_newlon = 1
    x = torch.rand((batch_size, num_in), requires_grad=True)
    t = torch.rand((batch_size, num_newlon))

    pws = torch.rand(size=(num_in, num_newlon), requires_grad=True)
    cws = torch.rand(size=(num_in, num_newlon), requires_grad=True)

    pbs = torch.rand(size=(1, num_newlon), requires_grad=True)
    cbs = torch.rand(size=(1, num_newlon), requires_grad=True)

    # print(x)
    # print(x.repeat(1, num_newlon // num_in))
    _x = x.repeat(1, num_newlon // num_in)
    px = torch.rand((batch_size, num_in), requires_grad=True)
    nx = torch.rand((batch_size, num_in), requires_grad=True)
    sign = Sign.plus
    y = ag_monoamine(sign, px, nx, pws, cws, pbs, cbs)

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
    # print(f"{y.grad=}")
    print(f"{pws.grad=}")
    print(f"{cws.grad=}")
    print(f"{px.grad=}")
    print(f"{nx.grad=}")


if __name__ == "__main__":
    main()
