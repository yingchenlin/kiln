import torch
import torch.nn as nn

from .dist_cov import *
from .mlp import MLP


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


class VarDropout(nn.Linear):

    def __init__(self, config, in_dim, out_dim, std):
        super().__init__(in_dim, out_dim)
        self.std = std
        self.in_dim = in_dim
        self.out_dim = out_dim

    def extra_repr(self):
        return f"in_dim={self.in_dim} out_dim={self.out_dim} std={self.std}"

    def forward(self, input):
        m, k = input
        m, k = self._dropout(m, k)
        m, k = self._linear(m, k)
        return m, k

    def _dropout(self, m, k):
        if self.std == 0:
            return m, k
        
        # handle deterministic values
        v = self.std**2
        if k == None:
            return m, v * m.square()

        # update distribution
        return m, k + v * (m.square() + k)
    
    def _linear(self, m, k):
        if k == None:
            return super().forward(m), None

        # update distribution
        w, b = self.weight.T, self.bias
        mp = m @ w + b
        vp = k @ w.square()
        return mp, vp

    def _linear(self, m, k):
        mp = super().forward(m)
        if k == None:
            kp = None
        else:
            w = self.weight
            kp = k @ w.square()
        return mp, kp


class VarMLP(MLP):

    Flatten = DistFlatten
    Activation = VarReLU
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
        u = torch.randn((self.num_samples, 1, k.size(-1)), device=k.device)
        x = m.unsqueeze(0) + k.unsqueeze(0).sqrt() * u

        # expected value of cross entropy loss
        ip = i.unsqueeze(0).expand(x.shape[:-1])
        return cross_entropy(x, ip).mean(0)


class VarQuadraticCrossEntropyLoss(nn.Module):

    def forward(self, input, target):
        (m, k), i = input, target
        quad_term = 0
        if k != None:
            p = m.softmax(-1)
            quad_term = (p * (1 - p) * k).sum(-1) * 0.5
        return cross_entropy(m, i) + quad_term
