import torch

from tasks import run
from datasets import get_dataset
from models import get_model
from tools import get_loss_fn, get_optimizer
from metrics import Metrics


class Engine:

    def setup(self, config, device, verbose):

        self.device = device

        self.dataset = get_dataset(config["dataset"])
        self.model = get_model(self.dataset, config["model"])
        self.model.to(self.device)

        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])
        self.optimizer = get_optimizer(
            config["fit"]["optimizer"], self.model.parameters())
        self.metrics = Metrics({"topk": 1}, {"top1": lambda _: 1})

        self.num_samples = config["fit"]["samples"]

        if verbose:
            print(config)
            print(self.model)

    def get_dataloader(self, test):
        return self.dataset.test_loader if test else self.dataset.train_loader

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save_state(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def _train_loss(self, input, target):
        samples = []
        for _ in range(self.num_samples):
            output = self.model(input)
            sample = self.loss_fn(output, target) + self.model.reg_loss()
            samples.append(sample)
        return torch.stack(samples).mean(0)

    def train(self, dataloader):

        self.model.train()
        self.metrics.reset()

        for input, target in dataloader:
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            loss = self._train_loss(input, target)
            loss.mean().backward()
            self.optimizer.step()

            self.metrics.add_loss(loss)
            yield self.metrics.get()

    def eval(self, dataloader):

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for input, target in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                loss = self.loss_fn(output, target)

                self.metrics.add_loss(loss)
                self.metrics.add_rank(output, target)
                yield self.metrics.get()


if __name__ == "__main__":
    run(Engine)
