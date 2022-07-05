import numpy as np
from torch import nn

from .dropouts import get_dropout
from .dist import *


class MLP(nn.Sequential):

    def __init__(self, config, input_shape, num_classes):

        input_dim = np.prod(input_shape)
        output_dim = num_classes
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        layers = []
        layers.append(nn.Flatten())
        for i in range(num_layers):
            layers.append(self._get_dropout(config["dropout"], i))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(self._get_dropout(config["dropout"], num_layers))
        layers.append(nn.Linear(input_dim, output_dim))

        super().__init__(*layers)

    def _get_dropout(self, config, layer):
        std = config["std"] if layer in config["layers"] else 0
        return get_dropout(config, std)


class DistMLP(nn.Sequential):

    def __init__(self, config, input_shape, num_classes):

        input_dim = np.prod(input_shape)
        output_dim = num_classes
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        layers = []
        layers.append(DistFlatten())
        for i in range(num_layers):
            layers.append(self._get_dropout(config["dropout"], i))
            layers.append(DistLinear(input_dim, hidden_dim))
            layers.append(DistReLU())
            input_dim = hidden_dim
        layers.append(self._get_dropout(config["dropout"], num_layers))
        layers.append(DistLinear(input_dim, output_dim))

        super().__init__(*layers)

    def _get_dropout(self, config, layer):
        std = config["std"] if layer in config["layers"] else 0
        return DistDropout(std)
