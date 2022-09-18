import numpy as np
import torch
import torch.nn as nn


def get_dropout(config, input_dim, output_dim, std):
    name = config["name"]
    if name == "bernoulli":
        return BernoulliDropout(config, input_dim, output_dim, std)
    if name == "uniform":
        return UniformDropout(config, input_dim, output_dim, std)
    if name == "normal":
        return NormalDropout(config, input_dim, output_dim, std)
    if name == "lin-apx":
        return LinearApproxDropout(config, input_dim, output_dim, std)
    if name == "l2":
        return L2Regularization(config, input_dim, output_dim, std)
    raise Exception(f"unknown dropout '{name}'")


class DropoutBase(nn.Linear):

    def __init__(self, config, input_dim, output_dim, std):
        super().__init__(input_dim, output_dim)
        self.std = std
        self.on = config["on"]
        self._setup()

    def extra_repr(self):
        return f"std={self.std}, on={self.on}, in={self.in_features}, out={self.out_features}"

    def forward(self, x):
        w, b = self.weight, self.bias
        if self.training:
            if self.on == "state":
                x = x * self._sample(x.shape)
            if self.on == "weight":
                w = w * self._sample(w.shape)
        return x @ w.T + b

    def _setup(self):
        pass

    def _sample(self, shape):
        raise Exception("unimplemented method")


class BernoulliDropout(DropoutBase):

    def _setup(self):
        self._prob = 1 / (self.std * self.std + 1)

    def _sample(self, shape):
        p = torch.full(shape, self._prob, device=self.device)
        return torch.bernoulli(p) / self._prob


class UniformDropout(DropoutBase):

    def _setup(self):
        r = self.std * np.sqrt(3)
        self._scale = r * 2
        self._shift = 1 - r

    def _sample(self, shape):
        return torch.rand(shape, device=self.device) * self._scale + self._shift


class NormalDropout(DropoutBase):

    def _sample(self, shape):
        return torch.randn(shape, device=self.device) * self.std + 1


class Regularization(DropoutBase):

    def forward(self, x):
        w, b = self.weight, self.bias
        self.state = x
        return x @ w.T + b

    def reg_loss(self, ctx):
        raise Exception("unimplemented method")


class LinearApproxDropout(Regularization):

    def reg_loss(self, ctx):

        reg_loss = 0

        if len(ctx) == 0:
            probs = self.forward(self.state).softmax(-1)
            hess = probs.diag_embed(0, -2, -1) - probs.unsqueeze(-1) * probs.unsqueeze(-2)
            jacob = torch.ones_like(probs).diag_embed(0, -2, -1)
        else:
            hess = ctx["hess"]
            jacob = ctx["jacob"]

        if self.std > 0 and self.on == "weight":
            metric = torch.einsum("bij,bik,bjk->bk", hess, jacob, jacob)
            term = torch.einsum("bk,bkl,bl->b", metric, self.weight.square(), self.state.square())
            reg_loss = term * (self.std ** 2 / 2)

        jacob = jacob @ self.weight

        if self.std > 0 and self.on == "state":
            metric = torch.einsum("bij,bik,bjk->bk", hess, jacob, jacob)
            term = torch.einsum("bk,bk->b", metric, self.state.square())
            reg_loss = term * (self.std ** 2 / 2)

        jacob = jacob * (self.state > 0).unsqueeze(-2)

        ctx["hess"] = hess
        ctx["jacob"] = jacob

        return reg_loss


class L2Regularization(Regularization):

    def reg_loss(self, ctx):
        if self.std == 0:
            return 0
        if self.on == "weight":
            return self.weight.square().sum() * (self.std / 2)
        if self.on == "state":
            return self.state.square().sum(-1) * (self.std / 2)
