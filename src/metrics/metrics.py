from typing import Dict

from .aggregators import *
from .ranking import Ranking
from .modules import CaptureLayer


class Metrics:

    def __init__(self, config):
        self.ranking = Ranking(config["ranking"])
        self.reset()

    def reset(self) -> None:
        self.aggs = {}
        self.aggs: dict[str, torch.Tensor]

    def get(self) -> Dict[str, float]:
        return {k: v.get() for k, v in self.aggs.items()}

    def add_losses(self, losses: torch.Tensor):
        self._add("loss", losses, "mean")

    def add_ranks(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor):
        for key, values in self.ranking(inputs, outputs, targets):
            self._add(key, values, "mean")

    def add_states(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, CaptureLayer):
                self._add_vec(f"{name}.state", module.state)

    def add_params(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self._add_vec(name, param)

    def _add_vec(self, name: str, value: torch.Tensor):
        self._add(f"${name}.l0", value, "l0")
        self._add(f"${name}.l1", value, "l1")
        self._add(f"${name}.l2", value, "l2")

    def _add(self, key: str, value: torch.Tensor, agg: str):
        if key not in self.aggs:
            self.aggs[key] = get_aggregator(agg)
        self.aggs[key].add(value)
