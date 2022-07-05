from .ae import *
from .mlp import *
from .rnn import *
from .dropouts import *
from .dist import *
from .losses import *

from torch import optim


def get_loss_fn(config):
    name = config["name"]
    if name == "ce":
        return CrossEntropyLoss()
    if name == "multi-ce":
        return MultiCrossEntropyLoss()
    if name == "dist-ce":
        return DistCrossEntropyLoss()
    raise Exception(f"unknown loss function '{name}'")


def get_optimizer(config, params):
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    name = config["name"]
    if name == "sgd":
        momentum = config.get("momentum", 0)
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    if name == "adam":
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=(beta1, beta2))
    raise Exception(f"unknown optimizer '{name}'")


def get_model(config, input_shape, num_classes):
    name = config["name"]
    if name == "ae":
        return Autoencoder(config, input_shape, num_classes, get_model)
    if name == "mlp":
        return MLP(config, input_shape, num_classes)
    if name == "dist-mlp":
        return DistMLP(config, input_shape, num_classes)
    if name == "lstm":
        return LSTM(config, input_shape, num_classes)
    raise Exception(f"unknown model '{name}'")

