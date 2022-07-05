from torch import nn


class CrossEntropyLoss(nn.Module):

    def forward(self, logit, index):
        assert(logit.shape[:-1] == index.shape)
        return logit.logsumexp(-1) - logit.gather(-1, index.unsqueeze(-1)).squeeze(-1)


class MultiCrossEntropyLoss(nn.Module):

    def forward(self, logit, weight):
        assert(logit.shape == weight.shape)
        return logit.logsumexp(-1) * weight.sum(-1) - (logit * weight).sum(-1)
