from typing import Dict
import torch.nn as nn

from .aggregators import *
from .ranking import Ranking


def ce(x, i):
    if isinstance(x, tuple):
        x = x[0]
    return x.logsumexp(-1) - x.gather(-1, i.unsqueeze(-1)).squeeze(-1)


class CaptureLayer(nn.Module):

    def forward(self, input):
        self.state = input
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        self.state.retain_grad()
        return input


class Metrics:

    def __init__(self, config):
        self.ranking = Ranking(config["ranking"])
        self.reset()

    def reset(self) -> None:
        self.aggs = {}
        self.aggs: dict[str, torch.Tensor]

    def get(self) -> Dict[str, float]:
        return {k: v.get() for k, v in self.aggs.items()}

    def add_states(self, targets: torch.Tensor, model: torch.nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, CaptureLayer):
                self._add(f"${name}.state.l2", "l2", module.state)

    def add_losses(self, outputs: torch.Tensor, targets: torch.Tensor, losses: torch.Tensor):
        self._add("loss", "mean", losses)
        self._add("ce", "mean", ce(outputs, targets))

    def add_ranks(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor):
        for key, values in self.ranking(inputs, outputs, targets):
            self._add(key, "mean", values)

    def add_params(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self._add(f"${name}.l2", "l2", param)

    def _add(self, key: str, agg: str, value: torch.Tensor):
        if key not in self.aggs:
            self.aggs[key] = get_aggregator(agg)
        self.aggs[key].add(value)
