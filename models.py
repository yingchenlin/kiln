import numpy as np
import torch.nn as nn

from dropouts import Regularization, get_dropout


def get_model(dataset, config):
    name = config["name"]
    if name == "mlp":
        return MLP(dataset, config)
    raise Exception(f"unknown model '{name}'")


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    raise Exception(f"unknown activation '{name}'")


class MLP(nn.Sequential):

    def __init__(self, dataset, config):

        input_dim = np.prod(dataset.input_shape)
        output_dim = dataset.num_classes
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]
        activation = config["activation"]

        layers = []
        layers.append(nn.Flatten())
        for i in range(num_layers):
            layers.append(self._get_dropout(
                input_dim, hidden_dim, config["dropout"], i))
            input_dim = hidden_dim
            layers.append(get_activation(activation))
        layers.append(self._get_dropout(
            input_dim, output_dim, config["dropout"], num_layers))

        super().__init__(*layers)

    def _get_dropout(self, in_dim, out_dim, config, layer):
        std = config["std"] if layer in config["layers"] else 0
        return get_dropout(config, in_dim, out_dim, std)

    def reg_loss(self, output, target):
        reg_loss = 0
        modules = [m for m in self.modules() if isinstance(m, Regularization)]
        first_index = _find(modules, lambda m: m.std != 0)
        if first_index != -1:
            ctx = modules[first_index]._init(output, target)
            for m in reversed(modules[first_index:]):
                if m.std != 0:
                    reg_loss += m._reg_loss(ctx)
                ctx = m._next(ctx)
        return reg_loss


def _find(arr, predicate):
    return next((i for i, x in enumerate(arr) if predicate(x)), -1)
