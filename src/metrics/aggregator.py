import numpy as np


def noop(x):
    return x


class Aggregator:

    TRANSFORMS = {
        "mean": (noop, noop),
        "l1": (np.abs, noop),
        "l2": (np.square, np.sqrt),
    }

    def __init__(self, p):
        if p not in self.TRANSFORMS:
            raise Exception(f"unsupported p={p}")
        self.fwd, self.bwd = self.TRANSFORMS[p]
        self.reset()

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

    def __repr__(self):
        return f"sum={self.sum} num={self.num}"
