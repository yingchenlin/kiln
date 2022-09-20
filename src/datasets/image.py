import torch
import torchvision


class ImageDataset:

    def __init__(self, config, path):
        input_shape, num_classes, train_set, test_set = \
            ImageDataset._load_dataset(config["name"], path)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train_loader = self._get_dataloader(config["train"], train_set)
        self.test_loader = self._get_dataloader(config["test"], test_set)

    def _get_dataloader(self, config, dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])

    @staticmethod
    def _get_transform(input_shape):
        input_ch = input_shape[0]
        mean = tuple([0.5 for _ in range(input_ch)])
        std = tuple([0.25 for _ in range(input_ch)])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])
        return transform

    @staticmethod
    def _load_dataset(name, path):
        root = f"{path}/{name}"

        if name == "mnist":
            input_shape = (1, 28, 28)
            num_classes = 10
            transform = ImageDataset._get_transform(input_shape)
            train_set = torchvision.datasets.MNIST(
                root=root, train=True, transform=transform, download=True)
            test_set = torchvision.datasets.MNIST(
                root=root, train=False, transform=transform, download=True)
            return (input_shape, num_classes, train_set, test_set)

        if name == "cifar-10":
            input_shape = (3, 32, 32)
            num_classes = 10
            transform = ImageDataset._get_transform(input_shape)
            train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, transform=transform, download=True)
            test_set = torchvision.datasets.CIFAR10(
                root=root, train=False, transform=transform, download=True)
            return (input_shape, num_classes, train_set, test_set)

        if name == "cifar-100":
            input_shape = (3, 32, 32)
            num_classes = 100
            transform = ImageDataset._get_transform(input_shape)
            train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, transform=transform, download=True)
            test_set = torchvision.datasets.CIFAR100(
                root=root, train=False, transform=transform, download=True)
            return (input_shape, num_classes, train_set, test_set)

        if name == "svhn":
            input_shape = (3, 32, 32)
            num_classes = 10
            transform = ImageDataset._get_transform(input_shape)
            train_set = torchvision.datasets.SVHN(
                root=root, split="train", transform=transform, download=True)
            test_set = torchvision.datasets.SVHN(
                root=root, split="test", transform=transform, download=True)
            return (input_shape, num_classes, train_set, test_set)

        raise Exception(f"unknown image dataset '{name}'")
