import torch
import torch.nn as nn
import numpy as np

from .mlp import MLP, Erf


def outer(x):
    return x.unsqueeze(-1) * x.unsqueeze(-2)

def gaussian(z):
    g0 = (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))
    g1 = ((z * np.sqrt(0.5)).erf() + 1) * 0.5
    return g0, g1

def cross_entropy(x, i, dim=-1):
    return x.logsumexp(dim) - x.gather(dim, i.unsqueeze(dim)).squeeze(dim)


def get_activation(config):
    name = config["name"]
    if name == "relu":
        return CovReLU(config)
    if name in CovErf.TX:
        return CovErf(config)
    raise Exception(f"unknown activation '{name}'")


class DistFlatten(nn.Flatten):

    def forward(self, input):
        return super().forward(input), None


class CovErf(Erf):

    def __init__(self, config):
        super().__init__(config)
        self.biased = config["biased"]
        self.order = config["order"]
        self.stop_grad = config["stop_grad"]

    def extra_repr(self):
        return f"name={self.name} biased={self.biased} order={self.order} stop_grad={self.stop_grad}"

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None

        z, r, g0, g1 = self._compute(m, k)

        if self.biased:
            mp = g1
        else:
            mp = super().forward(m)

        if self.order == 1:
            km = r * outer(g0)
        elif self.order == 2:
            km = r * (1 + k * r * outer(z) / 2) * outer(g0)
        else:
            raise Exception(f"unsupported order '{self.order}'")

        if self.stop_grad:
            km = km.detach()
        kp = k * km

        return mp, kp

    def _compute(self, m, k):
        s = k.diagonal(0, -2, -1).sqrt() # standard deviation
        sx, tx, sy, ty = Erf.TX[self.name]
        sx, tx, sy, ty = sx * np.sqrt(2), tx * np.sqrt(2), sy * 2, ty - sy
        n = 1 / ((s * sx).square() + 1).sqrt()
        r = (sx ** 2) * outer(n)
        z = (m * sx + tx) * n
        g0, g1 = gaussian(z)
        g0 = g0 * sy
        g1 = g1 * sy + ty
        return z, r, g0, g1


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
        s = k.diagonal(0, -2, -1).sqrt() + 1e-8 # standard deviation
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
        self.target = config["target"]
        self.dropout_cross = config["dropout_cross"]
        self.state_stop_grad = config["state_stop_grad"]
        self.weight_stop_grad = config["weight_stop_grad"]
        self.propagate_stop_grad = config["propagate_stop_grad"]

    def extra_repr(self):
        props = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "std": self.std,
            "dropout_cross": self.dropout_cross,
            "state_stop_grad": self.state_stop_grad,
            "weight_stop_grad": self.weight_stop_grad,
            "propagate_stop_grad": self.propagate_stop_grad,
        }
        return " ".join([f"{k}={v}" for k, v in props.items()])

    def forward(self, input):
        m, k = input
        mp = super().forward(m)
        k0 = self._cov_propagate(k)
        k1 = self._cov_dropout(m, k)
        kp = k0 if k1 == None else k1 if k0 == None else k0 + k1
        return mp, kp

    def _cov_dropout(self, m, k):
        if self.std == 0:
            return None

        w = self.weight
        if self.weight_stop_grad:
            w = w.detach()
        if self.state_stop_grad:
            m = m.detach()

        d = m.square()
        if self.dropout_cross and k != None:
            d = d + k.diagonal(0, -2, -1)
        d = d * self.std ** 2

        kp = w @ d.diag_embed() @ w.T
        if self.target == "weight":
            kp = kp.diagonal(0, -2, -1).diag_embed()

        return kp

    def _cov_propagate(self, k):
        if k == None:
            return None

        w = self.weight
        if self.propagate_stop_grad:
            w = w.detach()

        return w @ k @ w.T


class CovMLP(MLP):

    Flatten = DistFlatten
    Activation = lambda _, *args: get_activation(*args)
    Dropout = CovDropout


class CovMonteCarloCrossEntropyLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_samples = config["num_samples"]
        self.stop_grad = config["stop_grad"]

    def forward(self, input, target):
        (m, k), i = input, target
        if k == None:
            return cross_entropy(m, i)

        # sample from multivariate normal distribution
        l, _ = torch.linalg.cholesky_ex(k)
        r = torch.randn((self.num_samples, l.size(-1)), device=l.device)
        while r.dim() < l.dim(): r = r.unsqueeze(-2)
        d = (l.unsqueeze(0) @ r.unsqueeze(-1)).squeeze(-1)

        # expected value of cross entropy loss
        mp = m.unsqueeze(0)
        if self.stop_grad: mp = mp.detach()
        ip = i.unsqueeze(0).expand(d.shape[:-1])
        L = cross_entropy(mp + d, ip).mean(0)
        if self.stop_grad:
            L0 = cross_entropy(m, i)
            L = L0 + (L - L0.detach())
        return L


class CovQuadraticCrossEntropyLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.stop_grad = config["stop_grad"]

    def forward(self, input, target):
        (m, k), i = input, target
        quad_term = 0
        if k != None:
            p = m.softmax(-1)
            h = p.diag_embed(0, -2, -1) - p.unsqueeze(-2) * p.unsqueeze(-1)
            if self.stop_grad: h = h.detach()
            quad_term = torch.einsum("sbij,sbij->sb", k, h) * 0.5
        return cross_entropy(m, i) + quad_term
