import torch

from .sources import *
from functools import lru_cache

class EventDataset:

    def __init__(self, config, path):

        state_path = self._get_state_path(config, path)
        if os.path.exists(state_path):
            self._load(state_path)
        else:
            self._setup(config)
            self._save(state_path)

        num_users = len(self.user_vocab)
        num_items = len(self.item_vocab)
        user_split =  int(num_users * config["split_ratio"])
        
        self.train_loader = self._get_dataloader(config["train"], 0, user_split)
        self.test_loader = self._get_dataloader(config["test"], user_split, num_users)
        self.input_shape = (num_items,)
        self.num_classes = num_items

    def _get_state_path(self, config, path):
        name = config["name"]
        min_user = config["min_user"]
        min_item = config["min_item"]
        return f"{path}/{name}/cache-mu_{min_user}-mi_{min_item}.pt"

    def _load(self, path):
        state = torch.load(path)
        self.user_vocab = state["user_vocab"]
        self.item_vocab = state["item_vocab"]
        self.users = state["users"]
        self.items = state["items"]

    def _save(self, path):
        torch.save({
            "user_vocab": self.user_vocab,
            "item_vocab": self.item_vocab,
            "users": self.users,
            "items": self.items,
        }, path)

    def _setup(self, config):
        
        # load data
        df = self._load_dataset(config["name"], config["path"])
        df = df.sort_values(["user", "ts"]).reset_index(drop=True)

        # build vocab
        user_vocab = self._get_vocab(df["user"], config["min_user"])
        item_vocab = self._get_vocab(df["item"], config["min_item"])

        # index and filter oov
        df["user"] = user_vocab.get_indexer(df["user"])
        df["item"] = item_vocab.get_indexer(df["item"])
        df = df[(df["user"] != -1)&(df["item"] != -1)]

        self.user_vocab = torch.tensor(user_vocab)
        self.item_vocab = torch.tensor(item_vocab)
        self.users = torch.tensor(df["user"].values)
        self.items = torch.tensor(df["item"].values)

    def _get_vocab(self, values, min_count):
        counts = values.value_counts(sort=False)
        counts = counts[counts >= min_count]
        counts = counts.sample(frac=1) # shuffle
        return counts.index

    def _get_dataloader(self, config, min_user, max_user):
        mask = (self.users >= min_user)&(self.users < max_user)
        users = self.users[mask] - min_user
        items = self.items[mask]
        num_users = max_user - min_user
        num_items = len(self.item_vocab)
        dataset = BagOfWordDataset(config, users, items, num_users, num_items)
        dataloader = torch.utils.data.DataLoader(
            dataset,batch_size=config["batch_size"], shuffle=config["shuffle"])
        return dataloader

    @lru_cache
    def _load_dataset(self, name, path):
        root = f"{path}/{name}"
        if name == "ml-100k":
            df = MovieLens100K(root)
            df = df[df["rating"] >= 4].drop(["rating"], axis=1)
            return df
        if name == "ml-1m":
            df = MovieLens1M(root)
            df = df[df["rating"] >= 4].drop(["rating"], axis=1)
            return df
        raise Exception(f"unknown text dataset '{name}'")


class BagOfWordDataset(torch.utils.data.Dataset):

    def __init__(self, config, labels, values, num_labels, num_values):
        self.split_ratio = config["split_ratio"]
        self.data = self._process(labels, values, num_labels, num_values)

    def _process(self, labels, values, num_labels, num_values):
        data = torch.zeros((num_labels, num_values), dtype=torch.bool)
        data[labels, values] = True
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        masks = torch.rand(data.shape) < self.split_ratio
        inputs = data * masks
        targets = data * ~masks
        return inputs, targets
