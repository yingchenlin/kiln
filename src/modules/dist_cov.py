import torch
import torch.nn as nn
import numpy as np

from .mlp import MLP


def outer(x):
    return x.unsqueeze(-1) * x.unsqueeze(-2)

def gaussian(z):
    g0 = (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))
    g1 = ((z * np.sqrt(0.5)).erf() + 1) * 0.5
    return g0, g1

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
        self.stop_grad = config["stop_grad"]

    def extra_repr(self):
        return f"biased={self.biased} order={self.order} stop_grad={self.stop_grad}"

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None

        # compute attributes
        s = k.diagonal(0, 1, 2).sqrt() + 1e-8 # standard deviation
        g0, g1 = gaussian(m / s)

        mp = self._mean(m, s, g0, g1)
        km = self._cov_mul(m, s, k, g0, g1)
        if self.stop_grad:
            km = km.detach()
        kp = k * km
        return mp, kp

    def _mean(self, m, s, g0, g1) -> torch.Tensor:
        if self.biased:
            return m * g1 + s * g0
        else:
            return super().forward(m)

    def _cov_mul(self, m, s, k, g0, g1) -> torch.Tensor:
        if self.order == 0:
            return outer(m > 0)
        elif self.order == 1:
            return outer(g1)
        elif self.order == 2:
            return (outer(g1) + (k * 0.5) * outer(g0 / s))
        else:
            raise Exception(f"unsupported order '{self.order}'")


class CovDropout(nn.Linear):

    def __init__(self, config, in_dim, out_dim, std):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std = std
        self.dropout_cross = config["dropout_cross"]
        self.dropout_stop_grad = config["dropout_stop_grad"]
        self.linear_stop_grad = config["linear_stop_grad"]

    def extra_repr(self):
        props = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "std": self.std,
            "dropout_cross": self.dropout_cross,
            "dropout_stop_grad": self.dropout_stop_grad,
            "linear_stop_grad": self.linear_stop_grad,
        }
        return " ".join([f"{k}={v}" for k, v in props.items()])

    def forward(self, input):
        m, k = input
        m, k = self._dropout(m, k)
        m, k = self._linear(m, k)
        return m, k

    def _dropout(self, m, k):
        if self.std == 0:
            return m, k

        d = m.square()
        if self.dropout_stop_grad:
            d = d.detach()
        if self.dropout_cross and k != None:
            d = d + k.diagonal(0, 1, 2)

        v = self.std ** 2
        kp = k if k != None else 0
        kp = kp + v * d.diag_embed()

        return m, kp

    def _linear(self, m, k):
        mp = super().forward(m)
        if k == None:
            return mp, None

        w = self.weight
        if self.linear_stop_grad:
            w = w.detach()
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
        r = torch.randn((self.num_samples, 1, l.size(-1), 1), device=l.device)
        x = m.unsqueeze(0) + (l.unsqueeze(0) @ r).squeeze(-1)

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
