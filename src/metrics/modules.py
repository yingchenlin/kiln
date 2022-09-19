import torch
from torch import nn


class AggregateLayer(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("s0", torch.zeros(1))
        self.register_buffer("s1", torch.zeros(num_features))
        self.register_buffer("s2", torch.zeros(num_features, num_features))
        self.register_buffer("s1p", torch.zeros(num_features))
        self.register_buffer("s2p", torch.zeros(num_features, num_features))
        self.s0: torch.Tensor
        self.s1: torch.Tensor
        self.s2: torch.Tensor
        self.s1p: torch.Tensor
        self.s2p: torch.Tensor

    def add(self, x: torch.Tensor) -> None:
        self.s0.add_(len(x))
        self.s1.add_(x.sum(0))
        self.s2.add_(x.T @ x)
        xp = (x > 0).float()
        self.s1p.add_(xp.sum(0))
        self.s2p.add_(xp.T @ xp)

    def reset(self) -> None:
        self.s0.zero_()
        self.s1.zero_()
        self.s2.zero_()
        self.s1p.zero_()
        self.s2p.zero_()


class CaptureLayer(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.train_agg = AggregateLayer(num_features)
        self.test_agg = AggregateLayer(num_features)

    def _get_agg(self):
        return self.train_agg if self.training else self.test_agg

    def _get_state(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, tuple):
            x = x[0]
        return x.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.state = self._get_state(input)
        self._get_agg().add(self.state)
        return input

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._get_agg().reset()
        return result
