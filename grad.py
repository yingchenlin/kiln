import os
import json
import torch
from tqdm import tqdm

from src.datasets import get_dataset
from src.modules import get_model, get_loss_fn, get_optimizer

class Snapshot:

    def __init__(self, dir_path, label, epoch, seed):
        self.dir_path = dir_path
        self.label = label
        self.epoch = epoch
        self.seed = seed
        self.path = f"{self.dir_path}/grad.{self.label}.{self.epoch}.pt"

    def __call__(self):
        if os.path.exists(self.path):
            return
        self._setup()
        s0 = self._snapshot()
        self._init_grad()
        self.optimizer.step()
        self._epoch()
        s1 = self._snapshot()
        self._save(s0, s1)

    def _init_grad(self):
        inputs = torch.rand(1, 1, *self.dataset.input_shape)
        targets = torch.zeros((1, 1), dtype=torch.long)
        outputs = self.model(inputs)
        losses = self.loss_fn(outputs, targets)
        (losses * 0).backward()

    def _setup(self):

        with open(f"{self.dir_path}/{self.label}/config.json") as f:
            config = json.load(f)

        self.dataset = get_dataset(config["dataset"], "datasets")
        self.model = get_model(config["model"], self.dataset.input_shape, self.dataset.num_classes)
        self.optimizer = get_optimizer(config["fit"]["optimizer"], self.model.parameters())
        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])
        self.num_samples = config["fit"]["num_samples"]
        self.learning_rate = config["fit"]["optimizer"]["learning_rate"]

        path = f"{self.dir_path}/{self.label}/checkpoint-{self.epoch}.pt"
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _epoch(self):
        self.model.eval()
        self.model.train()
        self.optimizer.zero_grad()
        torch.manual_seed(self.seed)
        desc = f"{self.label}@{self.epoch}"
        for inputs, targets in tqdm(self.dataset.train_loader, desc=desc):
            inputs = inputs.unsqueeze(0).expand(self.num_samples, *inputs.shape)
            targets = targets.unsqueeze(0).expand(self.num_samples, *targets.shape)
            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets)
            sub_loss = losses.mean() / len(self.dataset.train_loader)
            sub_loss.backward()

    def _snapshot(self):
        snapshot = {}
        self._weight(snapshot)
        self._state(snapshot)
        return snapshot

    def _state(self, snapshot):
        for i in range(2, 14, 2):
            s0 = self.model.get_buffer(f"{i}.train_agg.s0").detach().clone()
            s1 = self.model.get_buffer(f"{i}.train_agg.s1").detach().clone()
            s2 = self.model.get_buffer(f"{i}.train_agg.s2").detach().clone()
            snapshot[f"{i}.state.m1"] = s1 / s0
            snapshot[f"{i}.state.m2"] = s2 / s0

    def _weight(self, snapshot):
        for i in range(1, 14, 4):
            b = self.model.get_parameter(f"{i}.bias").detach().clone()
            w = self.model.get_parameter(f"{i}.weight").detach().clone()
            if i < 14-1:
                snapshot[f"{i}.bias"] = b
                snapshot[f"{i}.bias.1t"] = b[:, None] * b[None, :]
                snapshot[f"{i}.weight.1t"] = w @ w.T
            if i > 0:
                snapshot[f"{i}.weight.t1"] = w.T @ w

    def _save(self, s0, s1):
        lr = self.learning_rate
        sg = {k: (s1[k]-v)/lr for k, v in s0.items()}
        torch.save({
            "value": s0,
            "grad": sg,
        }, self.path)

'''
class Snapshot:

    def __init__(self, dir_path, config_label, state_label, epoch, seed, learning_rate=1e-3):
        self.dir_path = dir_path
        self.config_label = config_label
        self.state_label = state_label
        self.epoch = epoch
        self.seed = seed
        self.learning_rate = learning_rate
        self.path = f"{self.dir_path}/grad.{self.config_label}.{self.state_label}.{self.epoch}.pt"

    def __call__(self):
        if os.path.exists(self.path):
            return
        self._setup()
        self._epoch()
        s0 = self._snapshot()
        self.optimizer.step()
        self._epoch()
        s1 = self._snapshot()
        self._save(s0, s1)

    def _setup(self):

        with open(f"{self.dir_path}/{self.config_label}/config.json") as f:
            config = json.load(f)

        self.dataset = get_dataset(config["dataset"], "datasets")
        self.model = get_model(config["model"], self.dataset.input_shape, self.dataset.num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])
        self.num_samples = config["fit"]["num_samples"]

        path = f"{self.dir_path}/{self.state_label}/checkpoint-{self.epoch}.pt"
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])

    def _epoch(self):
        self.model.eval()
        self.model.train()
        self.optimizer.zero_grad()
        torch.manual_seed(self.seed)
        desc = f"{self.config_label}:{self.state_label}@{self.epoch}"
        for inputs, targets in tqdm(self.dataset.train_loader, desc=desc):
            inputs = inputs.unsqueeze(0).expand(self.num_samples, *inputs.shape)
            targets = targets.unsqueeze(0).expand(self.num_samples, *targets.shape)
            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets)
            sub_loss = losses.mean() / len(self.dataset.train_loader)
            sub_loss.backward()

    def _snapshot(self):
        snapshot = {}
        self._weight(snapshot)
        self._state(snapshot)
        return snapshot

    def _state(self, snapshot):
        for i in range(2, 14, 2):
            s0 = self.model.get_buffer(f"{i}.train_agg.s0").detach().clone()
            s1 = self.model.get_buffer(f"{i}.train_agg.s1").detach().clone()
            s2 = self.model.get_buffer(f"{i}.train_agg.s2").detach().clone()
            snapshot[f"{i}.state.m1"] = s1 / s0
            snapshot[f"{i}.state.m2"] = s2 / s0

    def _weight(self, snapshot):
        for i in range(1, 14, 4):
            b = self.model.get_parameter(f"{i}.bias").detach().clone()
            w = self.model.get_parameter(f"{i}.weight").detach().clone()
            if i < 14-1:
                snapshot[f"{i}.bias"] = b
                snapshot[f"{i}.bias.1t"] = b[:, None] * b[None, :]
                snapshot[f"{i}.weight.1t"] = w @ w.T
            if i > 0:
                snapshot[f"{i}.weight.t1"] = w.T @ w

    def _save(self, s0, s1):
        lr = self.learning_rate
        sg = {k: (s1[k]-v)/lr for k, v in s0.items()}
        torch.save({
            "value": s0,
            "grad": sg,
        }, self.path)
'''

seed = 2020883518546164801
labels = [f"mean_tanh_{l}_0" for l in ("l", "l1", "l2", "l3", "l123")]
for epoch in (10, 20, 30, 40, 50):
    for label in labels:
        #Snapshot("outputs/lyr", label, label, epoch, seed)()
        #Snapshot("outputs/lyr", label, labels[0], epoch, seed)()
        #Snapshot("outputs/lyr", labels[0], label, epoch, seed)()
        Snapshot("outputs/lyr", label, epoch, seed)()
