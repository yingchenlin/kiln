import numpy as np
import torch
import torch.nn as nn

from dropouts import StatefulLayer, LinearLayer, get_dropout


def get_model(dataset, config):
    name = config["name"]
    if name == "mlp":
        return MLP(dataset, config)
    raise Exception(f"unknown model '{name}'")


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
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
            layers.extend(self.get_layers(
                input_dim, hidden_dim, config["dropout"], i))
            layers.append(get_activation(activation))
            input_dim = hidden_dim
        layers.extend(self.get_layers(
            input_dim, output_dim, config["dropout"], num_layers))
        layers.append(StatefulLayer())

        super().__init__(*layers)

    def get_layers(self, in_dim, out_dim, config, layer):
        std = config["std"] if layer in config["layers"] else 0
        reg = config["reg"] if layer in config["layers"] else 0
        lock = layer in config["lock"]
        return [
            StatefulLayer(),
            get_dropout(config["name"], std),
            LinearLayer(in_dim, out_dim, reg, lock),
        ]

    def reg_loss(self):
        reg_sum = 0
        for m in self.modules():
            if isinstance(m, LinearLayer):
                reg_sum = reg_sum + m.reg_loss()
        return reg_sum
