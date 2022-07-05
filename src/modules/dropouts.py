import numpy as np
import torch
import torch.nn as nn


def get_dropout(config, std):
    name = config["name"]
    if name == "bernoulli":
        return BernoulliDropout(config, std)
    if name == "uniform":
        return UniformDropout(config, std)
    if name == "normal":
        return NormalDropout(config, std)
    raise Exception(f"unknown dropout '{name}'")


class DropoutBase(nn.Module):

    def __init__(self, config, std):
        super().__init__()
        self.std = std
        self._setup()

    def extra_repr(self):
        return f"std={self.std}"

    def forward(self, x):
        if self.training and self.std != 0:
            d = self._sample(x.shape)
            x = x * d
        return x

    def _setup(self):
        pass

    def _sample(self, shape):
        pass


class BernoulliDropout(DropoutBase):

    def _setup(self):
        self._prob = 1 / (self.std * self.std + 1)

    def _sample(self, shape):
        p = torch.full(shape, self._prob)
        return torch.bernoulli(p) / self._prob


class UniformDropout(DropoutBase):

    def _setup(self):
        r = self.std * np.sqrt(3)
        self._scale = r * 2
        self._shift = 1 - r

    def _sample(self, shape):
        return torch.rand(shape) * self._scale + self._shift


class NormalDropout(DropoutBase):

    def _sample(self, shape):
        return torch.randn(shape) * self.std + 1
