import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dropout(config, in_dim, out_dim, std):
    name = config["name"]
    if name == "bernoulli":
        return BernoulliDropout(in_dim, out_dim, std, config)
    if name == "uniform":
        return UniformDropout(in_dim, out_dim, std, config)
    if name == "normal":
        return NormalDropout(in_dim, out_dim, std, config)
    if name == "reg":
        return ExpRegularization(in_dim, out_dim, std, config)
    if name == "l2":
        return L2Regularization(in_dim, out_dim, std, config)
    raise Exception(f"unknown dropout '{name}'")


class DropoutLayer(nn.Linear):

    def __init__(self, in_dim, out_dim, std, config):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std = std
        self.on = config["on"]
        self._setup()

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, std={self.std}, on={self.on}"

    def forward(self, x: torch.Tensor):
        self.state = x
        w, b = self.weight, self.bias
        if self.training and self.std != 0:
            w, x = self._dropout(w, x)
        return F.linear(x, w, b)

    def _dropout(self, w, x):
        if self.on == "state":
            d = self._sample(x.shape)
            x = x * d
        if self.on == "weight":
            d = self._sample(w.shape)
            w = w * d
        return w, x

    def _setup(self):
        pass

    def _sample(self, shape) -> torch.Tensor:
        pass


class BernoulliDropout(DropoutLayer):

    def _setup(self):
        self._prob = 1 / (self.std * self.std + 1)

    def _sample(self, shape):
        p = torch.full(shape, self._prob)
        return torch.bernoulli(p) / self._prob


class UniformDropout(DropoutLayer):

    def _setup(self):
        r = self.std * np.sqrt(3)
        self._scale = r * 2
        self._shift = 1 - r

    def _sample(self, shape):
        return torch.rand(shape) * self._scale + self._shift


class NormalDropout(DropoutLayer):

    def _sample(self, shape):
        m = torch.full(shape, 1.0)
        s = torch.full(shape, self.std)
        return torch.normal(m, s)


class Regularization(DropoutLayer):

    def _sample(self, shape):
        return 1

    def _init(_, output, target):
        return None

    def _next(m, ctx):
        return None

    def _reg_loss(m, ctx):
        return 0


class L2Regularization(Regularization):

    def _reg_loss(m, ctx):
        if m.on == "state":
            assert m.state.dim() == 2
            return (m.std**2 / 2) * m.state.square().mean(1)
        if m.on == "weight":
            return (m.std**2 / 2) * m.weight.square().mean(1).sum()
        raise f"unrecognized on='{m.on}'"


class ExpRegularization(Regularization):

    def _init(_, output, target):
        prob = F.softmax(output, dim=1)
        jacob = torch.ones_like(prob).diag_embed()
        return prob, jacob

    def _next(m, ctx):
        prob, jacob = ctx
        jacob = torch.matmul(jacob, m.weight) * (m.state > 0).unsqueeze(1)
        return prob, jacob

    def _reg_loss(m, ctx):

        prob, jacob = ctx
        weight = m.weight
        state = m.state

        if m.on == "state":
            jacob = torch.matmul(jacob, weight)
            state_2 = state.square()
            return m._approx(prob, jacob, state_2)
        if m.on == "weight":
            state_2 = torch.einsum(
                "ij,bj->bi", weight.square(), state.square())
            return m._approx(prob, jacob, state_2)
        raise Exception(f"unrecognized on='{m.on}'")

    def _approx(m, prob, jacob, state_2):
        m2 = torch.einsum("bi,bij->bj", prob, jacob.square())
        m1 = torch.einsum("bi,bij->bj", prob, jacob)
        c2 = m2 - m1.square()
        return (m.std**2 / 2) * torch.einsum("bi,bi->b", c2, state_2)


'''
m2 = torch.einsum("bi,bij->bj", prob, jacob.square())
m1 = torch.einsum("bi,bij->bj", prob, jacob)
c2 = m2 - m1.square()
jacob_2 = jacob.square()
m22 = torch.einsum("bi,bij,bik->bjk", prob, jacob_2, jacob_2)
m21 = torch.einsum("bi,bij,bik->bjk", prob, jacob_2, jacob)
m11 = torch.einsum("bi,bij,bik->bjk", prob, jacob, jacob)
m20 = m2.unsqueeze(2)
m01 = m1.unsqueeze(1)
m01_2 = m01.square()
c4 = (m22 - 2*addsym(m21*m01) - 2*m11.square() + 8*m11*mulsym(m01)
        - mulsym(m20) + 2*addsym(m20*m01_2) - 6*mulsym(m01_2))
return (
    (m.std**2 / 2) * torch.einsum("bi,bi->b", c2, state_2) +
    (m.std**4 / 8) * torch.einsum("bij,bi,bj->b", c4, state_2, state_2))

def addsym(x):
    return x + x.swapaxes(1, 2)

def mulsym(x):
    return x * x.swapaxes(1, 2)
'''
