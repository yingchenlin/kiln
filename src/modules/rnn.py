import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, config, input_shape, num_classes):
        super().__init__()

        assert(input_shape == ())
        embedding_dim = config["embedding_dim"]
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, embedding_dim)

    def train(self, mode: bool = True):
        self.hidden = None
        return super().train(mode)

    def forward(self, input):
        rnn_input = self.embedding(input)
        rnn_output, self.hidden = self.lstm(rnn_input, self.hidden)
        self.hidden = tuple(h.detach() for h in self.hidden)
        output = self.linear(rnn_output)
        logit = torch.einsum("tbi,ji->tbj", output, self.embedding.weight)
        return logit
