import numpy as np
from torch import nn

from .dropouts import DropoutBase, Regularization, get_dropout

if __package__ == "modules":
    from metrics import CaptureLayer
else:
    from ..metrics import CaptureLayer


def get_activation(config):
    name = config["name"]
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in Erf.TX:
        return Erf(config)
    raise Exception(f"unknown activation '{name}'")


class Erf(nn.Module):

    TX = {
        "erf": (1, 0, 1, 0),
        "erf-g": (1/np.sqrt(2), 0, 0.5, 0.5),
        "erf-s": (np.sqrt(np.pi)/4, 0, 0.5, 0.5),
        "erf-t": (np.sqrt(np.pi)/2, 0, 1, 0),
    }

    def __init__(self, config):
        super().__init__()
        self.name = config["name"]
        assert(self.name in Erf.TX)

    def extra_repr(self):
        return f"name={self.name}"

    def forward(self, x):
        sx, tx, sy, ty = Erf.TX[self.name]
        return (x * sx + tx).erf() * sy + ty


class MLP(nn.Sequential):

    Flatten = nn.Flatten
    Activation = lambda _, *args: get_activation(*args)
    Dropout = lambda _, *args: get_dropout(*args)

    def __init__(self, config, input_shape, num_classes):

        self.config = config

        layers = []
        layers.append(self.Flatten(start_dim=2))
        output_dim = np.prod(input_shape)
        for i in range(config["num_layers"]):
            input_dim, output_dim = output_dim, config["hidden_dim"]
            layers.append(self._get_dropout(i, input_dim, output_dim))
            layers.append(CaptureLayer(output_dim, num_classes))
            layers.append(self._get_activation(i))
            layers.append(CaptureLayer(output_dim, num_classes))
        input_dim, output_dim = output_dim, num_classes
        layers.append(self._get_dropout(config["num_layers"], input_dim, output_dim))
        layers.append(CaptureLayer(output_dim, num_classes))

        super().__init__(*layers)

    def _get_dropout(self, layer, input_dim, output_dim):
        config = self.config["dropout"]
        std = config["std"] if layer in config["layers"] else 0
        return self.Dropout(config, input_dim, output_dim, std)

    def _get_activation(self, layer):
        config = self.config["activation"]
        if layer not in config["layers"]:
            return nn.Identity()
        return self.Activation(config)

    def set_epoch(self, epoch):
        for m in self.modules():
            if isinstance(m, DropoutBase):
                m.set_epoch(epoch)

    def reg_loss(self):
        ctx = {}
        reg_loss = 0
        for i, m in enumerate(self.children()):
            if isinstance(m, Regularization) and m.std > 0:
                break
        for m in reversed(list(self.children())[i:]):
            if isinstance(m, Regularization):
                reg_loss = reg_loss + m.reg_loss(ctx)
        return reg_loss


class CrossEntropyLoss(nn.Module):

    def forward(self, logit, index):
        assert(logit.shape[:-1] == index.shape)
        return logit.logsumexp(-1) - logit.gather(-1, index.unsqueeze(-1)).squeeze(-1)


class MultiCrossEntropyLoss(nn.Module):

    def forward(self, logit, weight):
        assert(logit.shape == weight.shape)
        return logit.logsumexp(-1) * weight.sum(-1) - (logit * weight).sum(-1)
