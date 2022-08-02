import torch


class Ranking:

    # assert fn(1) == 1
    WEIGHTS = {
        "recall": lambda i: torch.ones_like(i) + 0.,
        "ndcg": lambda i: 1 / torch.log2(i + 1),
        "mrr": lambda i: 1 / i,
    }

    def __init__(self, config):
        self.topks = config["topks"]
        self.weights = config["weights"]
        self.is_dist = config["is_dist"]
        self.is_excl = config["is_excl"]
        self.is_multi = config["is_multi"]
        self.eps = config["eps"]

    def __call__(self, inputs, outputs, targets):
        if self.is_dist:
            outputs = outputs[0]
        if self.is_excl:
            neg_inf = torch.tensor([-torch.inf])
            outputs = torch.where(inputs, neg_inf, outputs)
        if self.is_multi:
            return self._get_matches(outputs, targets)
        else:
            return self._get_places(outputs, targets)

    def _get_matches(self, outputs, targets):
        assert(outputs.shape == targets.shape)
        assert(targets.dtype == torch.bool)

        outputs = outputs + torch.rand_like(outputs) * self.eps
        _, indices = torch.topk(outputs, max(self.topks))
        matches = targets.gather(-1, indices)
        target_counts = targets.sum(-1)

        for name in self.weights:
            places = (torch.arange(max(self.topks)) + 1).float()
            weights = self.WEIGHTS[name](places)
            weight_sums = torch.cat([torch.ones(1), weights.cumsum(-1)])

            for topk in self.topks:
                gains = (matches[..., :topk] * weights[:topk]).sum(-1)
                ideal_gains = weight_sums[target_counts.clip(max=topk)]
                values = gains / ideal_gains
                yield (f"{name}{topk}", values)

    def _get_places(self, outputs, targets):
        assert(outputs.shape[:-1] == targets.shape)
        assert(targets.dtype == torch.int64)
        
        outputs = outputs + torch.rand_like(outputs) * self.eps
        target_outputs = torch.gather(outputs, 1, targets.unsqueeze(-1))
        places = (outputs >= target_outputs).sum(-1)

        for name in self.weights:
            weights = self.WEIGHTS[name](places)
            for topk in self.topks:
                values = (places <= topk) * weights
                yield (f"{name}{topk}", values)
