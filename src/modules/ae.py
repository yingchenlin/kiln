import torch
from torch import nn


class Autoencoder(nn.Module):

    NEG_INF = torch.tensor([-1e10])

    def __init__(self, config, input_shape, num_classes, get_model):
        super().__init__()

        assert(input_shape == (num_classes,))
        self.emb_dim = config["emb_dim"]
        self.is_excl = config["is_excl"]
        self.fc = nn.Linear(num_classes, self.emb_dim, bias=False)
        self.inner = get_model(config["inner"], (self.emb_dim,), self.emb_dim)

    def forward(self, inputs):
        embeddings = self.fc(inputs.float())
        hiddens = self.inner(embeddings)
        outputs = torch.matmul(hiddens, self.fc.weight)
        if self.is_excl:
            outputs = torch.where(inputs, self.NEG_INF, outputs)
        return outputs
