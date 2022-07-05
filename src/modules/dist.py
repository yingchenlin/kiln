import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistFlatten(nn.Flatten):

    def forward(self, input):
        return super().forward(input), None


class DistLinear(nn.Linear):

    def forward(self, input):
        m, s = input
        if s == None:
            return super().forward(m), None
        w, b = self.weight.T, self.bias
        mp = torch.matmul(m, w) + b
        sp = torch.einsum("bij,ik,jl->bkl", s, w, w)
        return mp, sp


class DistReLU(nn.Module):

    def forward(self, input):
        m, s = input
        if s == None:
            return m * (m > 0), None
        sd = s.diagonal(0, 1, 2).sqrt()
        z = m / sd
        g0, g1, g2 = self._gauss(z)
        mp = sd * g2
        sp = s * self._outer(g1) + s.square() * self._outer(g0 / sd) * (np.pi/2-1)
        return mp, sp

    def _outer(self, x):
        return x[:, :, None] * x[:, None, :]

    def _gauss(self, z):
        g0 = (z.square() * -0.5).exp() * np.sqrt(1/(np.pi*2))
        g1 = ((z * np.sqrt(0.5)).erf() + 1) * 0.5
        g2 = z * g1 + g0
        return g0, g1, g2


class DistDropout(nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def extra_repr(self):
        return f"std={self.std}"

    def forward(self, input):
        m, s = input
        return m, self._cov(s, m)

    def _cov(self, s, m):
        if self.std == 0:
            return s
        v = self.std**2
        if s == None:
            return v * m.square().diag_embed()
        sp = s.diagonal(0, 1, 2) + m.square()
        return s + v * sp.diag_embed()


def agi_softmax(m, s):
    c = 1 / (np.pi * 2 * np.square(np.log(2)))
    v = s.diagonal(0, -2, -1)
    dm = m[:, None, :] - m[:, :, None]
    ds = v[:, None, :] + v[:, :, None] - 2 * s
    z = dm / (1 + c * ds).sqrt()
    p = 1 / z.exp().sum(-1)
    return p


class DistCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, s, i):
        ctx.save_for_backward(m, s, i)
        return F.cross_entropy(m, i, reduction="none")

    @staticmethod
    def backward(ctx, grad_output):
        m, s, i = ctx.saved_tensors
        if s == None:
            p = m.softmax(-1)
        else:
            p = agi_softmax(m, s)
        grad = p - F.one_hot(i, m.shape[-1])
        return grad_output[:, None] * grad, None, None


class DistCrossEntropyLoss(nn.Module):

    def forward(self, input, target):
        (m, s), i = input, target
        return DistCrossEntropyFunction.apply(m, s, i)
