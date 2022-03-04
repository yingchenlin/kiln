from torch import nn
from torch import optim


def get_loss_fn(config):
    name = config["name"]
    reduction = config["reduction"]
    if name == "hinge":
        return nn.MultiMarginLoss(reduction=reduction)
    if name == "mse":
        return nn.MultiMarginLoss(2, reduction=reduction)
    if name == "ce":
        return nn.CrossEntropyLoss(reduction=reduction)
    if name == "nll":
        return nn.NLLLoss(reduction=reduction)
    if name == "kl":
        return nn.KLDivLoss(reduction=reduction)
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
