import torch


class Aggregator:

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum = 0
        self.num = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().cpu().flatten()
        self.num += len(x)
        self.sum += self._fwd(x)

    def get(self) -> torch.Tensor:
        m = self.sum / self.num
        return self._inv(m)

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        raise Exception("unimplemented aggregator")

    def _inv(self, m: torch.Tensor) -> torch.Tensor:
        return m.item()


class MeanAggregator(Aggregator):

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class L0Aggregator(Aggregator):

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).sum()


class L1Aggregator(Aggregator):

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum()


class L2Aggregator(Aggregator):

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        return x.square().sum()

    def _inv(self, m: torch.Tensor) -> torch.Tensor:
        return m.sqrt().item()


class SetAggregator(Aggregator):

    def reset(self) -> None:
        self.value = torch.nan

    def add(self, x: torch.Tensor) -> None:
        self.value = x.item()

    def get(self) -> torch.Tensor:
        return self.value


def get_aggregator(name: str) -> Aggregator:
    if name == "mean":
        return MeanAggregator()
    if name == "l0":
        return L0Aggregator()
    if name == "l1":
        return L1Aggregator()
    if name == "l2":
        return L2Aggregator()
    if name == "set":
        return SetAggregator()
    raise Exception(f"unsupported aggregator name={name}")