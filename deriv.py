import torch


def pad(t: torch.Tensor, *ds: int):
    for d in ds:
        t = t.unsqueeze(d)
    return t


def diag(t: torch.Tensor):
    return t.diag_embed()


def rms(t: torch.Tensor, **kwargs):
    return t.square().mean(**kwargs).sqrt()


def main():

    torch.set_default_tensor_type(torch.DoubleTensor)

    b, n = 100, 10
    x = torch.rand(b, n)
    u = torch.rand(b, n)
    v = torch.rand(b, n)
    h = 1e-3
    p = x.softmax(dim=-1)

    def f(i, j=0):
        dx = v*(h*i) + u*(h*j)
        r = torch.expm1(dx)
        m = (p * r).sum(dim=-1)
        return torch.log1p(m)

    d1 = (f(0.5)-f(-0.5))/h
    d2 = (f(1)-f(0)*2+f(-1))/h/h
    d3 = (f(1.5)-f(0.5)*3+f(-0.5)*3-f(-1.5))/h/h/h
    d4 = (f(2)-f(1)*4+f(0)*6-f(-1)*4+f(-2))/h/h/h/h

    m1 = p
    m2 = diag(p)-pad(p, 1)*pad(p, 2)
    m3 = (
        diag(diag(p))
        - pad(diag(p), 1)*pad(p, 2, 3)
        - pad(diag(p), 2)*pad(p, 1, 3)
        - pad(diag(p), 3)*pad(p, 1, 2)
        + 2*pad(p, 2, 3)*pad(p, 1, 3)*pad(p, 1, 2))
    m4 = (
        diag(diag(diag(p)))
        - pad(diag(diag(p)), 4)*pad(p, 1, 2, 3)
        - pad(diag(diag(p)), 3)*pad(p, 1, 2, 4)
        - pad(diag(diag(p)), 2)*pad(p, 1, 3, 4)
        - pad(diag(diag(p)), 1)*pad(p, 2, 3, 4)
        - pad(diag(p), 3, 4)*pad(diag(p), 1, 2)
        - pad(diag(p), 2, 4)*pad(diag(p), 1, 3)
        - pad(diag(p), 2, 3)*pad(diag(p), 1, 4)
        + 2*pad(diag(p), 3, 4)*pad(p, 1, 2, 4)*pad(p, 1, 2, 3)
        + 2*pad(diag(p), 2, 4)*pad(p, 1, 3, 4)*pad(p, 1, 2, 3)
        + 2*pad(diag(p), 2, 3)*pad(p, 1, 3, 4)*pad(p, 1, 2, 4)
        + 2*pad(diag(p), 1, 4)*pad(p, 2, 3, 4)*pad(p, 1, 2, 3)
        + 2*pad(diag(p), 1, 3)*pad(p, 2, 3, 4)*pad(p, 1, 2, 4)
        + 2*pad(diag(p), 1, 2)*pad(p, 2, 3, 4)*pad(p, 1, 3, 4)
        - 6*pad(p, 2, 3, 4)*pad(p, 1, 3, 4)*pad(p, 1, 2, 4)*pad(p, 1, 2, 3))

    v1 = torch.einsum("bi,bi->b", m1, v)
    v2 = torch.einsum("bij,bi,bj->b", m2, v, v)
    v3 = torch.einsum("bijk,bi,bj,bk->b", m3, v, v, v)
    v4 = torch.einsum("bijkl,bi,bj,bk,bl->b", m4, v, v, v, v)

    print(rms(d1-v1))
    print(rms(d2-v2))
    print(rms(d3-v3))
    print(rms(d4-v4))

    d4p = (
        f(1, 1)+f(1, -1)+f(-1, 1)+f(-1, -1)
        + (f(1, 0)+f(0, 1)+f(-1, 0)+f(0, -1))*-2
        + f(0, 0)*4
    )/h/h/h/h
    v4p = torch.einsum("bijkl,bi,bj,bk,bl->b", m4, u, u, v, v)
    print(rms(d4p-v4p))

    p1 = pad(p, 1)
    p2 = pad(p, 2)
    p12 = p1*p2
    m4p = diag(p*(1+p*(-6+p*8)))+p12*(-1+p1*2+p2*2+p12*-6)
    u4p = torch.einsum("bij,bi,bj->b", m4p, u.square(), v.square())
    v4p = torch.einsum("biijj,bi,bj->b", m4, u.square(), v.square())
    print(rms(u4p-v4p))


if __name__ == "__main__":
    main()
