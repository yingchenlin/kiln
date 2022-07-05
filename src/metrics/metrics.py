from torch import nn

from .aggregator import Aggregator
from .ranking import Ranking


class CaptureLayer(nn.Module):

    def forward(self, input):
        self.state = input
        return input


class Metrics:

    def __init__(self, config):
        self.ranking = Ranking(config["ranking"])
        self.aggs = {}

    def reset(self):
        self.aggs = {}

    def get(self):
        return {k: v.mean() for k, v in self.aggs.items()}

    def add_losses(self, losses):
        self._add("loss", losses, p="mean")

    def add_ranks(self, scores, targets):
        for key, values in self.ranking(scores, targets):
            self._add(key, values, p="mean")

    def add_states(self, model):
        for i, m in enumerate(model.modules()):
            if isinstance(m, CaptureLayer):
                self._add(f"model_{i}.state.l1", m.state, p="l1")
                self._add(f"model_{i}.state.l2", m.state, p="l2")

    def add_param(self, model):
        for name, value in model.state_dict().items():
            self._add(f"model_{name}.l1", value, p="l1")
            self._add(f"model_{name}.l2", value, p="l2")

    def _add(self, key, value, p):
        if key not in self.aggs:
            self.aggs[key] = Aggregator(p)
        self.aggs[key].add(value)
