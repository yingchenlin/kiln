import torch
import numpy as np

from dropouts import DropoutLayer


class Metrics:

    def __init__(self, config, weights, eps=1e-10):
        self.topk = config["topk"]
        self.weights = weights
        self.eps = eps
        self.aggs = {}

    def reset(self):
        self.aggs = {}

    def get(self):
        return {k: v.mean() for k, v in self.aggs.items()}

    def add_loss(self, loss):
        self._add("loss", loss, p="mean")

    def add_rank(self, output, target):
        rank = self._rank(output, target)
        for name, get_weight in self.weights.items():
            value = (rank <= self.topk) * get_weight(rank)
            self._add(name, value, p="mean")

    def add_state(self, model, output):
        for i, m in enumerate(model.modules()):
            if isinstance(m, DropoutLayer):
                self._add(f"model_{i}.state.l1", m.state, p="l1")
                self._add(f"model_{i}.state.l2", m.state, p="l2")
        self._add(f"model.output.l1", output, p="l1")
        self._add(f"model.output.l2", output, p="l2")

    def add_param(self, model):
        for name, value in model.state_dict().items():
            self._add(f"model_{name}.l1", value, p="l1")
            self._add(f"model_{name}.l2", value, p="l2")

    def _rank(self, x, i):
        x = x + torch.rand_like(x) * self.eps  # random tie-breaker
        x_i = torch.gather(x, 1, i[:, None])
        return (x >= x_i).sum(1)

    def _add(self, key, value, p):
        if key not in self.aggs:
            self.aggs[key] = Aggregator(p)
        self.aggs[key].add(value)


def noop(x):
    return x


class Aggregator:

    def __init__(self, p):
        self.reset()
        if p == "mean":
            self.fwd = noop
            self.bwd = noop
        elif p == "l1":
            self.fwd = np.abs
            self.bwd = noop
        elif p == "l2":
            self.fwd = np.square
            self.bwd = np.sqrt
        else:
            raise Exception(f"unsupported p={p}")

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, x):
        x = x.flatten().detach().cpu().numpy()
        x = self.fwd(x)
        self.sum += x.sum()
        self.num += len(x)

    def mean(self):
        mean = self.sum / self.num
        return self.bwd(mean)
