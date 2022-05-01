import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Linear(nn.Linear):

    def forward(self, input):
        m, s = input
        w, b = self.weight.T, self.bias
        mp = torch.matmul(m, w) + b
        sp = self._cov(s, w)
        return mp, sp

    def _cov(self, s, w):
        if s == None:
            return None
        return torch.einsum("bij,ik,jl->bkl", s, w, w)


class ReLU2(nn.Module):

    def forward(self, input):
        m, s = input
        b = (m > 0)
        mp = m * b
        sp = self._cov(s, b)
        return mp, sp

    def _cov(self, s, b):
        if s == None:
            return None
        return s * b.unsqueeze(1) * b.unsqueeze(2)


class ReLU(nn.Module):

    def forward(self, input):
        m, s = input
        sd = s.diagonal(0, 1, 2).sqrt()
        z = m / sd
        zp = self._gauss_ad2(z)
        mp = sd * zp
        sp = self._cov(s, z, zp)
        return mp, sp

    def _cov(self, s, z, zp):
        if s == None:
            return None
        z1 = z.unsqueeze(1)
        z2 = z.unsqueeze(2)
        z_min = torch.min(z1, z2)
        z_max = torch.max(z1, z2)
        zp2 = (z1 * z2 + 1) * self._gauss_ad(z_min) + \
            z_max * self._gauss(z_min)
        return s * (zp2 - zp.unsqueeze(1) * zp.unsqueeze(2))

    def _gauss(self, x):
        return (x.square() / -2).exp() / np.sqrt(np.pi*2)

    def _gauss_ad(self, x):
        return ((x / np.sqrt(2)).erf() + 1) / 2

    def _gauss_ad2(self, x):
        return x * self._gauss_ad(x) + self._gauss(x)


class Dropout(nn.Module):

    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, input):
        m, s = input
        return m, self._cov(s, m)

    def _cov(self, s, m):
        if self.std == 0:
            return s
        if s == None:
            return self.std**2 * m.square().diag_embed()
        sm2 = s.diagonal(0, 1, 2) + m.square()
        return s + self.std**2 * sm2.diag_embed()


class CrossEntropyLoss(nn.Module):

    def forward(self, input, target):
        m, s = input
        loss = F.cross_entropy(m, target)
        if s != None:
            pass
        return loss


if __name__ == "__main__":

    model = nn.Sequential(
        Linear(30, 20),
        ReLU2(),
        Dropout(0.5),
        Linear(20, 10),
        ReLU2(),
        Dropout(0.5),
        Linear(10, 5))

    m = torch.rand(100, 30)
    s = torch.rand(100, 30, 30)
    mp, sp = model((m, s))
