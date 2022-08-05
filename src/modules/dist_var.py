import torch
import torch.nn as nn
import torch.nn.functional as F

from .dist_cov import *
from .mlp import MLP


class VarLinear(nn.Linear):

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None

        # update distribution
        w, b = self.weight.T, self.bias
        mp = m @ w + b
        vp = k @ w.square()
        return mp, vp


class VarReLU(nn.ReLU):

    def forward(self, input):
        m, k = input
        if k == None:
            return super().forward(m), None

        # retrieve attributes
        s = k.sqrt() # standard deviation
        z = m / (s + 1e-8) # inverse coefficient of variation

        # compute probabilities
        g0 = std_norm_pdf(z)
        g1 = std_norm_cdf(z)
        g2 = z * g1 + g0 # anti-devriative of std_norm_cdf

        # update distribution
        mp = s * g2
        kp = k * ((z - g2) * g2 + g1)
        return mp, kp


class VarDropout(nn.Module):

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
            return m, v * m.square()

        # update distribution
        return m, k + v * (m.square() + k)


class VarMLP(MLP):

    Flatten = DistFlatten
    Linear = VarLinear
    ReLU = VarReLU
    Dropout = VarDropout


class VarMonteCarloCrossEntropyLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_samples = config["samples"]

    def forward(self, input, target):
        (m, k), i = input, target
        if k == None:
            return cross_entropy(m, i)
        
        # sample from independent normal distributions
        u = torch.randn((self.num_samples, 1, k.size(-1)))
        x = m.unsqueeze(0) + k.unsqueeze(0).sqrt() * u

        # expected value of cross entropy loss
        ip = i.unsqueeze(0).expand(x.shape[:-1])
        return cross_entropy(x, ip).mean(0)
