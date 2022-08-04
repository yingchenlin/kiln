import numpy as np
from torch import nn

from .dropouts import get_dropout


class MLP(nn.Sequential):

    Flatten = nn.Flatten
    Linear = nn.Linear
    ReLU = nn.ReLU
    Dropout = get_dropout

    def __init__(self, config, input_shape, num_classes):

        input_dim = np.prod(input_shape)
        output_dim = num_classes
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        layers = []
        layers.append(self.Flatten())
        for i in range(num_layers):
            layers.append(self._get_dropout(config["dropout"], i))
            layers.append(self.Linear(input_dim, hidden_dim))
            layers.append(self.ReLU())
            input_dim = hidden_dim
        layers.append(self._get_dropout(config["dropout"], num_layers))
        layers.append(self.Linear(input_dim, output_dim))

        super().__init__(*layers)

    def _get_dropout(self, config, layer):
        std = config["std"] if layer in config["layers"] else 0
        return self.Dropout(config, std)
