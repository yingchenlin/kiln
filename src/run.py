import torch

from tasks import run
from datasets import get_dataset
from modules import get_model, get_loss_fn, get_optimizer
from metrics import Metrics


class Engine:

    def setup(self, config, path, device, verbose):

        self.device = device

        self.dataset = get_dataset(config["dataset"], path)
        self.model = get_model(
            config["model"], 
            self.dataset.input_shape, 
            self.dataset.num_classes)
        self.model.to(self.device)

        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])
        self.optimizer = get_optimizer(
            config["fit"]["optimizer"], self.model.parameters())
        self.metrics = Metrics(config["metrics"])

        self.num_samples = config["fit"]["num_samples"]

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

    def train(self, dataloader, epoch):

        self.model.set_epoch(epoch)
        self.model.train()
        self.metrics.reset()

        for inputs, targets in dataloader:
            inputs = self._pack(inputs, self.num_samples)
            targets = self._pack(targets, self.num_samples)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets)
            losses = losses + self.model.reg_loss()

            self.metrics.add_states(targets, self.model)
            self.metrics.add_losses(outputs, targets, losses)

            losses.mean().backward()
            self.optimizer.step()

            yield self.metrics.get()

    def eval(self, dataloader, epoch):

        self.model.set_epoch(epoch)
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            self.metrics.add_params(self.model)

            for inputs, targets in dataloader:
                inputs = self._pack(inputs)
                targets = self._pack(targets)

                outputs = self.model(inputs)
                losses = self.loss_fn(outputs, targets)
                losses = losses + self.model.reg_loss()

                self.metrics.add_states(targets, self.model)
                self.metrics.add_losses(outputs, targets, losses)
                self.metrics.add_ranks(inputs, outputs, targets)

                yield self.metrics.get()

    def _pack(self, data: torch.Tensor, num_copies = 1):
        data = data.to(self.device)
        data = data.unsqueeze(0).expand(num_copies, *data.shape)
        return data


if __name__ == "__main__":
    run(Engine)
