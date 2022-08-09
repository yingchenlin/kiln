from typing import Dict
from torch import nn

from .aggregators import *
from .ranking import Ranking


class CaptureLayer(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("train_num", torch.zeros(1))
        self.register_buffer("train_sum", torch.zeros(num_features))
        self.register_buffer("train_sum_sq", torch.zeros(num_features, num_features))
        self.register_buffer("eval_num", torch.zeros(1))
        self.register_buffer("eval_sum", torch.zeros(num_features))
        self.register_buffer("eval_sum_sq", torch.zeros(num_features, num_features))
        self.train_num: torch.Tensor
        self.train_sum: torch.Tensor
        self.train_sum_sq: torch.Tensor
        self.eval_num: torch.Tensor
        self.eval_sum: torch.Tensor
        self.eval_sum_sq: torch.Tensor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        x = input
        if isinstance(x, tuple):
            x = x[0]
        x = x.detach()
        self.state = x

        num, sum, sum_sq = self._get_state(self.training)
        num.add_(len(x))
        sum.add_(x.sum(0))
        sum_sq.add_(torch.einsum("bi,bj->ij", x, x))

        return input

    def train(self, mode: bool = True):
        num, sum, sum_sq = self._get_state(mode)
        num.zero_()
        sum.zero_()
        sum_sq.zero_()
        return super().train(mode)

    def _get_state(self, mode):
        num = self.train_num if mode else self.eval_num
        sum = self.train_sum if mode else self.eval_sum
        sum_sq = self.train_sum_sq if mode else self.eval_sum_sq
        return num, sum, sum_sq


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
        self._add(f"${name}.l1", value, "l1")
        self._add(f"${name}.l2", value, "l2")

    def _add(self, key: str, value: torch.Tensor, agg: str):
        if key not in self.aggs:
            self.aggs[key] = get_aggregator(agg)
        self.aggs[key].add(value)
