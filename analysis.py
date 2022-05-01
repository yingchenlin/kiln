import argparse
import logging
import os
import traceback
import json
import random
import numpy as np
import torch
import torch.multiprocessing as mp

from datasets import get_dataset
from models import get_model
from dropouts import DropoutLayer
from tools import get_loss_fn


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", "-b", type=str, default="outputs")
    parser.add_argument("--threads", "-t", type=int, default=4)
    args = parser.parse_args()

    # build tasks
    tasks = []
    for label in sorted(os.listdir(args.basepath)):
        root = f"{args.basepath}/{label}"
        seed = random.getrandbits(31)
        task = Task(label, root, seed)
        tasks.append(task)

    # spawn workers
    mp.set_start_method("spawn")
    queue = mp.Queue()
    workers = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devices = [torch.device(i) for i in range(num_devices)]
    for device in devices:
        for _ in range(args.threads):
            worker_args = (queue, device)
            worker = mp.Process(target=worker_fn, args=worker_args)
            workers.append(worker)

    # execute tasks
    for worker in workers:
        worker.start()
    for task in tasks:
        queue.put(task)
    for worker in workers:
        queue.put(None)
    for worker in workers:
        worker.join()


def worker_fn(queue, device):

    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO)

    while True:
        task = queue.get()
        if task is None:
            break
        try:
            task(device)
        except Exception:
            print(traceback.format_exc())


class Task:

    def __init__(self, label, root, seed):
        self.label = label
        self.root = root
        self.seed = seed

    def __call__(self, device):

        if os.path.exists(f"{self.root}/analysis.pt"):
            return
        if not os.path.exists(f"{self.root}/checkpoint.pt"):
            return

        with open(f"{self.root}/config.json") as f:
            config = json.load(f)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.dataset = get_dataset(config["dataset"])
        self.dataloader = self.dataset.train_loader
        self.model = get_model(self.dataset, config["model"])
        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])

        checkpoint = torch.load(f"{self.root}/checkpoint.pt")
        self.model.load_state_dict(checkpoint["model"])

        output = {
            "ml": self._compute(device),
            # "mc": self._compute(device, samples=8),
        }
        torch.save(output, f"{self.root}/analysis.pt")

    def _compute(self, device, samples=None):

        dropouts = list(get_layers(self.model, DropoutLayer))

        count = 0
        data = {
            "m1": [0 for _ in dropouts],
            "m2": [0 for _ in dropouts],
            "w": [0 for _ in dropouts],
            "w_d": [0 for _ in dropouts],
            "b": [0 for _ in dropouts],
            "b_d": [0 for _ in dropouts],
        }

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        for progress in self._iterate(device, samples):
            for input, target in progress:
                input = input.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss_fn(output, target)
                loss.mean().backward()

                count += 1

                for i, m in enumerate(dropouts[1:]):
                    add_i(data, i, get_stat(m.state))
                add_i(data, -1, get_stat(output))

                for i, m in enumerate(dropouts):
                    value, grad = safe_get(m.weight)
                    add_i(data, i, {"w": value, "w_d": grad})
                    value, grad = safe_get(m.bias)
                    add_i(data, i, {"b": value, "b_d": grad})

        data = {k: [(e/count).float() for e in v] for k, v in data.items()}

        return data

    def _iterate(self, device, samples=None):

        if samples == None:
            self.model.to(device)
            self.model.eval()
            logging.info(self.label)
            yield self.dataloader
            return

        self.model.to(device)
        self.model.train()
        for sample in range(samples):
            desc = f"{self.label} n={sample}"
            logging.info(desc)
            yield self.dataloader


def add_i(dst, i, src):
    for k, v in src.items():
        dst[k][i] = dst[k][i] + v


def get_stat(value):
    value = value.detach().cpu().double()
    m1 = value.mean(0)
    m2 = torch.matmul(value.T, value) / value.shape[0]
    return {"m1": m1, "m2": m2}


def safe_get(x):
    value = x.detach().cpu().double()
    grad = x.grad.detach().cpu().double()
    return value, grad


def get_layers(model, cls):
    for m in model.modules():
        if isinstance(m, cls):
            yield m


if __name__ == "__main__":
    run()
