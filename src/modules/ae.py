import torch
from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, config, input_shape, num_classes, get_model):
        super().__init__()

        assert(input_shape == (num_classes,))
        embedding_dim = config["embedding_dim"]
        self.fc = nn.Linear(num_classes, embedding_dim, bias=False)
        self.inner = get_model(config["inner"], (embedding_dim,), embedding_dim)

    def forward(self, input):
        embedding = self.fc(input + 1)
        hidden = self.inner(embedding)
        output = torch.matmul(hidden, self.fc.weight)
        return output
