import torch
import torchtext
import os


class TextDataset:

    UNKNOWN = "<unk>"

    def __init__(self, config, path):

        state_path = self._get_state_path(config, path)
        if os.path.exists(state_path):
            self._load(state_path)
        else:
            self._setup(config)
            self._save(state_path)

        self.input_size = ()
        self.num_classes = len(self.vocab)
        self.train_loader = SequenceDataset(config["train"], self.train_data)
        self.valid_loader = SequenceDataset(config["test"], self.valid_data)
        self.test_loader = SequenceDataset(config["test"], self.test_data)

    def _get_state_path(self, config, path):
        name = config["name"]
        tokenizer = config["tokenizer"]
        return f"{path}/{name}/cache-{tokenizer}.pt"

    def _load(self, path):
        state = torch.load(path)

        self.vocab = torchtext.vocab.Vocab(
            torchtext._torchtext.Vocab(state["tokens"], None))
        self.vocab.set_default_index(self.vocab[self.UNKNOWN])

        self.train_data = state["train_data"]
        self.valid_data = state["valid_data"]
        self.test_data = state["test_data"]

    def _save(self, path):
        torch.save({
            "tokens": self.vocab.get_itos(),
            "train_data": self.train_data,
            "valid_data": self.valid_data,
            "test_data": self.test_data,
        }, path)

    def _setup(self, config):

        # load corpus
        train_iter, valid_iter, test_iter = \
            TextDataset._load_dataset(config["name"], config["path"])

        # build tokenizer
        tokenizer = torchtext.data.utils.get_tokenizer(config["tokenizer"])

        # build vocab
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            map(tokenizer, train_iter), specials=[self.UNKNOWN])
        self.vocab.set_default_index(self.vocab[self.UNKNOWN])

        # convert tokens to indices
        self.train_data = self._index(train_iter, tokenizer)
        self.valid_data = self._index(valid_iter, tokenizer)
        self.test_data = self._index(test_iter, tokenizer)

    def _index(self, raw_text_iter, tokenizer):
        all_tokens = []
        for raw_text in raw_text_iter:
            tokens = self.vocab(tokenizer(raw_text))
            all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)

    @staticmethod
    def _load_dataset(name, path):
        root = f"{path}/{name}"
        if name == "penn":
            return torchtext.datasets.PennTreebank(root)
        if name == "wiki-2":
            return torchtext.datasets.WikiText2(root)
        if name == "wiki-103":
            return torchtext.datasets.WikiText103(root)
        raise Exception(f"unknown text dataset '{name}'")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, config, data):
        self.batch_size = config["batch_size"]
        self.bptt = config["bptt"]
        self.data = self._process(data)

    def _process(self, data):
        seq_len = data.size(0) // self.batch_size
        data = data[:seq_len * self.batch_size]
        data = data.view(self.batch_size, seq_len).t().contiguous()
        return data

    def __len__(self):
        return (len(self.data) - 1) // self.bptt

    def __getitem__(self, idx):
        if idx >= len(self): # support for loop
            raise IndexError()
        offset = idx * self.bptt
        length = min(self.bptt, len(self.data)-1-offset)
        inputs = self.data[offset:offset+length]
        targets = self.data[offset+1:offset+1+length]
        return inputs, targets
