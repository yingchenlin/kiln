import torch


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
        self._add("loss", loss)

    def add_rank(self, output, target):
        rank = self._rank(output, target)
        for name, get_weight in self.weights.items():
            value = (rank <= self.topk) * get_weight(rank)
            self._add(name, value)

    def _rank(self, x, i):
        x = x + torch.rand_like(x) * self.eps  # random tie-breaker
        x_i = torch.gather(x, 1, i[:, None])
        return (x >= x_i).sum(1)

    def _add(self, key, value):
        if key not in self.aggs:
            self.aggs[key] = Aggregator()
        self.aggs[key].add(value)


class Aggregator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, x):
        x = x.detach().cpu().numpy()
        self.sum += x.sum()
        self.num += len(x)

    def mean(self):
        return self.sum / self.num
