import unittest

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from dist import *


class Generator:

    def __init__(self, batch, dim):
        self.mean = torch.randn(batch, dim) * 3
        self.scale = torch.randn(batch, dim, dim) / np.power(dim, 1/2) * 3
        self.cov = torch.einsum("bik,bjk->bij", self.scale, self.scale)

    def __call__(self):
        eps = torch.randn_like(self.mean)
        return self.mean + torch.einsum("bij,bj->bi", self.scale, eps)


class Estimator:

    def __init__(self, times, fn):
        s1 = 0
        s2 = 0
        for _ in trange(times, leave=False):
            x = fn().detach()
            s1 += x
            s2 += self._outer(x)
        self.mean = s1 / times
        self.cov = s2 / times - self._outer(self.mean)

    def _outer(self, x):
        return x[:, None, :] * x[:, :, None]


class TestDist(unittest.TestCase):

    batch = 100
    dim = 100
    times = 10000

    def test_random(self):
        self._test_output("random", nn.Identity(), nn.Identity())

    def test_linear(self):
        rand_model = nn.Linear(self.dim, self.dim)
        dist_model = DistLinear(self.dim, self.dim)
        dist_model.load_state_dict(rand_model.state_dict())
        self._test_output("linear", rand_model, dist_model)
        self._test_output_det("linear", rand_model, dist_model)

    def test_relu(self):
        rand_model = nn.ReLU()
        dist_model = DistReLU()
        self._test_output("relu", rand_model, dist_model)
        self._test_output_det("relu", rand_model, dist_model)

    def test_dropout(self):
        rate = 0.5
        std = np.sqrt(rate / (1 - rate))
        rand_model = nn.Dropout(rate)
        dist_model = DistDropout(std)
        self._test_output("dropout", rand_model, dist_model)

    def test_cross_entropy(self):
        rand_loss_fn = nn.CrossEntropyLoss()
        dist_loss_fn = DistCrossEntropyLoss()
        self._test_grad("cross_entropy", rand_loss_fn, dist_loss_fn)
        self._test_grad_det("cross_entropy", rand_loss_fn, dist_loss_fn)

    def _test_grad(self, label, rand_loss_fn, dist_loss_fn):

        size = (self.batch, self.dim)
        rand_variable = torch.zeros(size, requires_grad=True)
        dist_variable = torch.zeros(size, requires_grad=True)

        gen = Generator(self.batch, self.dim)
        target = torch.randint(self.dim, size=(self.batch,))

        for _ in trange(self.times, leave=False):
            rand_input = gen() + rand_variable
            rand_loss = rand_loss_fn(rand_input, target)
            rand_loss.mean().backward()
        rand_grad = rand_variable.grad / self.times

        dist_input = (gen.mean + dist_variable, gen.cov)
        dist_loss = dist_loss_fn(dist_input, target)
        dist_loss.mean().backward()
        dist_grad = dist_variable.grad

        grad_error = self._error(rand_grad, dist_grad)
        print(grad_error, label)

    def _test_grad_det(self, label, rand_loss_fn, dist_loss_fn):

        size = (self.batch, self.dim)
        rand_variable = torch.zeros(size, requires_grad=True)
        dist_variable = torch.zeros(size, requires_grad=True)

        input = torch.randn(self.batch, self.dim)
        target = torch.randint(self.dim, size=(self.batch,))

        rand_input = input + rand_variable
        rand_loss = rand_loss_fn(rand_input, target)
        rand_loss.mean().backward()
        rand_grad = rand_variable.grad

        dist_input = (input + dist_variable, None)
        dist_loss = dist_loss_fn(dist_input, target)
        dist_loss.mean().backward()
        dist_grad = dist_variable.grad

        grad_error = self._error(rand_grad, dist_grad)
        print(grad_error, label)

    def _test_output(self, label, rand_model, dist_model):

        rand_model.train()
        gen = Generator(self.batch, self.dim)
        est = Estimator(self.times, lambda: rand_model(gen()))
        rand_output = est.mean, est.cov

        dist_input = (gen.mean, gen.cov)
        dist_output = dist_model(dist_input)

        output_mean_error = self._error(rand_output[0], dist_output[0])
        output_cov_error = self._error(rand_output[1], dist_output[1])
        print(output_mean_error, output_cov_error, label)

    def _test_output_det(self, label, rand_model, dist_model):

        input = torch.randn(self.batch, self.dim)

        rand_model.train()
        rand_output = rand_model(input)

        dist_output = dist_model((input, None))[0]

        output_mean_error = self._error(rand_output, dist_output)
        print(output_mean_error, label)

    def _error(self, expect, actual):
        expect = expect.detach()
        actual = actual.detach()
        dist = (expect - actual).square().mean().sqrt()
        norm = expect.square().mean().sqrt()
        return dist / norm


if __name__ == "__main__":
    unittest.main()