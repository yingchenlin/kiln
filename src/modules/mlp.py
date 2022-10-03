import numpy as np
from torch import nn

from .dropouts import Regularization, get_dropout
from metrics import CaptureLayer


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

        input_dim = np.prod(input_shape)
        output_dim = num_classes
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        layers = []
        layers.append(self.Flatten(start_dim=2))
        for i in range(num_layers):
            layers.append(self._get_dropout(config["dropout"], i, input_dim, hidden_dim))
            layers.append(CaptureLayer(hidden_dim, num_classes))
            layers.append(self._get_activation(config["activation"], i))
            layers.append(CaptureLayer(hidden_dim, num_classes))
            input_dim = hidden_dim
        layers.append(self._get_dropout(config["dropout"], num_layers, input_dim, output_dim))
        layers.append(CaptureLayer(output_dim, num_classes))

        super().__init__(*layers)

    def _get_dropout(self, config, layer, input_dim, output_dim):
        std = config["std"] if layer in config["layers"] else 0
        return self.Dropout(config, input_dim, output_dim, std)

    def _get_activation(self, config, layer):
        if layer not in config["layers"]:
            return nn.Identity()
        return self.Activation(config)

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
