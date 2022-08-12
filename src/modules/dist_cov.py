import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mlp import MLP


def outer(x):
    return x.unsqueeze(-1) * x.unsqueeze(-2)

def std_norm_pdf(z):
    return (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))

def std_norm_cdf(z):
    return ((z * np.sqrt(0.5)).erf() + 1) * 0.5

def cross_entropy(x, i, dim=-1):
    return x.logsumexp(dim) - x.gather(dim, i.unsqueeze(dim)).squeeze(dim)


class DistFlatten(nn.Flatten):

    def forward(self, input):
        return super().forward(input), None


class CovLinear(nn.Linear):

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None
        
        # update distribution
        w, b = self.weight.T, self.bias
        mp = m @ w + b
        kp = w.T @ k @ w
        return mp, kp


class CovReLU(nn.ReLU):

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None
        
        # retrieve attributes
        s = k.diagonal(0, 1, 2).sqrt() # standard deviation
        r = k / (outer(s) + 1e-8) # correlation coefficient
        z = m / (s + 1e-8) # inverse coefficient of variation

        # compute probabilities
        g0 = std_norm_pdf(z)
        g1 = std_norm_cdf(z)
        g2 = z * g1 + g0 # anti-devriative of std_norm_cdf

        # update distribution
        mp = s * g2
        kp = k * (outer(g1) + outer(g0) * r * 0.5)
        return mp, kp


class CovDropout(nn.Module):

    def __init__(self, config, std):
        super().__init__()
        self.std = std

    def extra_repr(self):
        return f"std={self.std}"

    def forward(self, input):
        m, k = input
        if self.std == 0:
            return m, k

        # handle deterministic values
        v = self.std**2
        if k == None:
            return m, v * m.square().diag_embed()

        # update distribution
        kd = k.diagonal(0, 1, 2) + m.square()
        return m, k + v * kd.diag_embed()


class CovMLP(MLP):

    Flatten = DistFlatten
    Linear = CovLinear
    ReLU = CovReLU
    Dropout = CovDropout


# approximate gaussian integral of softmax
def agi_softmax(m, k):
    c = 1 / (np.pi * 2 * np.square(np.log(2)))
    v = k.diagonal(0, -2, -1)
    dm = m.unsqueeze(-2) - m.unsqueeze(-1)
    ds = v.unsqueeze(-2) + v.unsqueeze(-1) - 2 * k
    z = dm / (1 + c * ds).sqrt()
    p = 1 / z.exp().sum(-1)
    return p


class CovApproxCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, k, i):
        ctx.save_for_backward(m, k, i)
        return F.cross_entropy(m, i, reduction="none")

    @staticmethod
    def backward(ctx, grad_output):
        m, k, i = ctx.saved_tensors
        if k == None:
            p = m.softmax(-1)
        else:
            p = agi_softmax(m, k)
        grad = p - F.one_hot(i, m.shape[-1])
        return grad_output[:, None] * grad, None, None


class CovApproxCrossEntropyLoss(nn.Module):

    def forward(self, input, target):
        (m, k), i = input, target
        return CovApproxCrossEntropyFunction.apply(m, k, i)


class CovMonteCarloCrossEntropyLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_samples = config["samples"]

    def forward(self, input, target):
        (m, k), i = input, target
        if k == None:
            return cross_entropy(m, i)

        # sample from multivariate normal distribution
        l, _ = torch.linalg.cholesky_ex(k)
        u = torch.randn((self.num_samples, 1, l.size(-1), 1))
        x = m.unsqueeze(0) + (l.unsqueeze(0) @ u).squeeze(-1)

        # expected value of cross entropy loss
        ip = i.unsqueeze(0).expand(x.shape[:-1])
        return cross_entropy(x, ip).mean(0)
