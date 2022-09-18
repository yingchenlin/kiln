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


class CovReLU(nn.ReLU):

    def __init__(self, config):
        super().__init__()
        self.biased = config["biased"]
        self.order = config["order"]

    def extra_repr(self):
        return f"bias={self.bias} order={self.order}"

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None
            
        if self.biased or self.order > 0:
    
            # retrieve attributes
            s = k.diagonal(0, 1, 2).sqrt() # standard deviation
            r = k / (outer(s) + 1e-8) # correlation coefficient
            z = m / (s + 1e-8) # inverse coefficient of variation

            # compute probabilities
            g0 = std_norm_pdf(z)
            g1 = std_norm_cdf(z)
            g2 = z * g1 + g0 # anti-devriative of std_norm_cdf

        # update distribution
        if self.biased:
            mp = s * g2
        else:
            mp = super().forward(m)
        if self.order == 0:
            kp = k * outer(m > 0)
        elif self.order == 1:
            kp = k * outer(g1)
        elif self.order == 2:
            kp = k * (outer(g1) + outer(g0) * r * 0.5)
        else:
            raise Exception(f"unsupported order '{self.order}'")
        return mp, kp


class CovDropout(nn.Linear):

    def __init__(self, config, in_dim, out_dim, std):
        super().__init__(in_dim, out_dim)
        self.std = std
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cross = config["cross"]

    def extra_repr(self):
        return f"in_dim={self.in_dim} out_dim={self.out_dim} std={self.std} cross={self.cross}"

    def forward(self, input):
        m, k = input
        m, k = self._dropout(m, k)
        m, k = self._linear(m, k)
        return m, k

    def _dropout(self, m, k):
        v = self.std ** 2
        if v == 0:
            kp = k
        elif k == None:
            kp = v * m.square().diag_embed()
        elif not self.cross:
            kp = k + v * m.square().diag_embed()
        else:
            d = k.diagonal(0, 1, 2) + m.square()
            kp = k + v * d.diag_embed()
        return m, kp

    def _linear(self, m, k):
        mp = super().forward(m)
        if k == None:
            kp = None
        else:
            w = self.weight
            kp = w @ k @ w.T
        return mp, kp


class CovMLP(MLP):

    Flatten = DistFlatten
    Activation = CovReLU
    Dropout = CovDropout


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


class CovQuadraticCrossEntropyLoss(nn.Module):

    def forward(self, input, target):
        (m, k), i = input, target
        quad_term = 0
        if k != None:
            p = m.softmax(-1)
            h = p.diag_embed(0, -2, -1) - p.unsqueeze(-1) * p.unsqueeze(-2)
            quad_term = torch.einsum("bij,bij->b", k, h) * 0.5
        return cross_entropy(m, i) + quad_term
