import torch
import torchvision
#import torchtext

from functools import lru_cache


def get_dataset(config):
    return ImageDataset(config)


'''
class LanguageDataset:

    UNKNOWN = "<unk>"

    def __init__(self, config):

        train_iter, valid_iter, test_iter = \
            self._load_dataset(config["name"], config["path"])
        tokenizer = self._build_tokenizer(config["tokenizer"])
        vocab = self._build_vocab(train_iter, tokenizer)

        train_data = self._index_data(train_iter, tokenizer, vocab)
        valid_data = self._index_data(valid_iter, tokenizer, vocab)
        test_data = self._index_data(test_iter, tokenizer, vocab)

    def _build_tokenizer(self, name):
        return torchtext.data.utils.get_tokenizer(name)

    def _build_vocab(self, raw_text_iter, tokenizer):
        vocab = torchtext.vocab.build_vocab_from_iterator(
            map(tokenizer, raw_text_iter), specials=[LanguageDataset.UNKNOWN])
        vocab.set_default_index(vocab[LanguageDataset.UNKNOWN])
        return vocab

    def _index_data(self, raw_text_iter, tokenizer, vocab):
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
                for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    @lru_cache
    def _load_dataset(self, name, path):
        root = f"{path}/{name}"

        if name == "penn":
            return torchtext.datasets.PennTreebank(root)

        if name == "wiki-2":
            return torchtext.datasets.WikiText2(root)

        if name == "wiki-103":
            return torchtext.datasets.WikiText103(root)

        raise Exception(f"unknown dataset '{name}'")
'''


class ImageDataset:

    def __init__(self, config):
        input_shape, num_classes, train_set, test_set = \
            ImageDataset._load_dataset(config["name"], config["path"])
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
    @lru_cache
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

        raise Exception(f"unknown dataset '{name}'")
