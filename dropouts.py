import numpy as np
import torch
import torch.nn as nn


def get_dropout(name, std):
    if name == "bernoulli":
        return BernoulliDropout(std)
    if name == "uniform":
        return UniformDropout(std)
    if name == "normal":
        return NormalDropout(std)
    if name == "log-normal":
        return LogNormalDropout(std)
    raise Exception(f"unknown dropout '{name}'")


class VariationalLayer(nn.Module):

    def __init__(self, in_dim, out_dim, reg):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.reg = reg

    def forward(self, x):

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        self.state = (mu, logvar)

        y = mu
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(x)
            y = y + std * eps
        return y

    def reg_loss(self):
        if self.reg == 0:
            return 0
        mu, logvar = self.state
        reg_loss = -0.5 * torch.sum(
            1 + logvar - logvar.exp() - mu.square(), dim=1)
        return reg_loss * self.reg


class StatefulLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.state = x
        return x


class LinearLayer(nn.Linear):

    def __init__(self, in_dim, out_dim, reg, lock):
        super().__init__(in_dim, out_dim)
        self.reg = reg
        if lock:
            for param in self.parameters():
                param.requires_grad = False

    def extra_repr(self):
        return f"{super().extra_repr()} reg={self.reg}"

    def reg_loss(self):
        if self.reg == 0:
            return 0
        return self.weight.square().sum() * self.reg


class DropoutLayer(nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std
        self._setup()

    def extra_repr(self):
        return f"std={self.std}"

    def forward(self, x):
        if self.training and self.std != 0:
            d = self._sample_like(x)
            x = x * d
        return x

    def _setup(self):
        pass

    def _sample_like(self, x):
        pass


class BernoulliDropout(DropoutLayer):

    def _setup(self):
        self._prob = 1 / (self.std * self.std + 1)

    def _sample_like(self, x):
        p = torch.full_like(x, self._prob)
        return torch.bernoulli(p) / self._prob


class UniformDropout(DropoutLayer):

    def _setup(self):
        r = self.std * np.sqrt(3)
        self._scale = r * 2
        self._shift = 1 - r

    def _sample_like(self, x):
        return torch.rand_like(x) * self._scale + self._shift


class NormalDropout(DropoutLayer):

    def _sample_like(self, x):
        m = torch.full_like(x, 1.0)
        s = torch.full_like(x, self.std)
        return torch.normal(m, s)


class LogNormalDropout(DropoutLayer):

    def _setup(self):
        l = np.log(self.std * self.std + 1)
        self._mean = -l / 2
        self._std = np.sqrt(l)

    def _sample_like(self, x):
        m = torch.full_like(x, self._mean)
        s = torch.full_like(x, self._std)
        return torch.normal(m, s).exp()
