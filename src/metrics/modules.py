import torch
from torch import nn


class AggregateLayer(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.register_buffer("s0", torch.zeros(1))
        self.register_buffer("s1", torch.zeros(self.num_features))
        self.register_buffer("s2", torch.zeros(self.num_features, self.num_features))
        self.register_buffer("n1", torch.zeros(self.num_features))
        self.register_buffer("n2", torch.zeros(self.num_features, self.num_features))
        self.s0: torch.Tensor
        self.s1: torch.Tensor
        self.s2: torch.Tensor
        self.n1: torch.Tensor
        self.n2: torch.Tensor

    def add(self, x: torch.Tensor) -> None:
        x = x.detach() # prevent memory leak
        x = x.flatten(end_dim=-2)
        self.s0.add_(len(x))
        self.s1.add_(x.sum(0))
        self.s2.add_(x.T @ x)
        n = (x > 0).float()
        self.n1.add_(n.sum(0))
        self.n2.add_(n.T @ n)

    def reset(self) -> None:
        self.s0.zero_()
        self.s1.zero_()
        self.s2.zero_()
        self.n1.zero_()
        self.n2.zero_()


class CaptureLayer(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.train_agg = AggregateLayer(num_features, num_classes)
        self.test_agg = AggregateLayer(num_features, num_classes)

    def get_agg(self):
        return self.train_agg if self.training else self.test_agg

    def forward(self, input):
        self.state = input
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        self.get_agg().add(self.state)
        return input

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.get_agg().reset()
        return result
