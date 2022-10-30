import torch

from tasks import run
from datasets import get_dataset
from modules import get_model, get_loss_fn, get_optimizer
from metrics import Metrics


class Engine:

    def setup(self, config, dataset_path, device, verbose):

        self.device = device
        self.epoch = 0

        self.dataset = get_dataset(config["dataset"], dataset_path)
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

    def get_dataloader(self, eval):
        return self.dataset.test_loader if eval else self.dataset.train_loader

    def load_state(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save_state(self, path):
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def next_epoch(self):
        self.epoch += 1
        self.model.set_epoch(self.epoch)
        return self.epoch

    def train(self, dataloader):

        self.model.train()
        self.metrics.reset()

        for inputs, targets in dataloader:
            inputs = self._pack(inputs, self.num_samples)
            targets = self._pack(targets, self.num_samples)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets)
            losses = losses + self.model.reg_loss()
            losses.mean().backward()

            self.metrics.add_states(targets, self.model)
            self.metrics.add_losses(outputs, targets, losses)

            self.optimizer.step()

            yield self.metrics.get()

    def eval(self, dataloader):

        self.model.eval()
        self.metrics.reset()

        self.metrics.add_params(self.model)

        for inputs, targets in dataloader:
            inputs = self._pack(inputs)
            targets = self._pack(targets)

            outputs = self.model(inputs)
            losses = self.loss_fn(outputs, targets)
            losses = losses + self.model.reg_loss()
            losses.mean().backward()

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
