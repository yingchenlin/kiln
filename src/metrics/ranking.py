import torch


class Ranking:

    # assert fn(1) == 1
    WEIGHTS = {
        "recall": lambda i: torch.ones_like(i),
        "ndcg": lambda i: 1 / torch.log2(i + 1),
        "mrr": lambda i: 1 / i,
    }

    def __init__(self, config):
        self.topks = config["topks"]
        self.weights = config["weights"]
        self.is_dist = config["is_dist"]
        self.is_multi = config["is_multi"]
        self.eps = config["eps"]

    def __call__(self, scores, targets):
        if self.is_dist:
            scores = scores[0]
        if self.is_multi:
            return self._get_matches(scores, targets)
        else:
            return self._get_places(scores, targets)

    def _get_matches(self, scores, targets):
        assert(scores.shape == targets.shape)
        assert(targets.dtype == torch.bool)

        scores = scores + torch.rand_like(scores) * self.eps
        _, indices = torch.topk(scores, max(self.topks))
        matches = targets.gather(-1, indices)
        match_counts = matches.sum(-1)

        for name in self.weights:
            places = (torch.arange(max(self.topks)) + 1).float()
            weights = self.WEIGHTS[name](places)
            weight_sums = torch.cat([torch.ones(1), weights.cumsum(-1)])

            for topk in self.topks:
                gains = (matches[..., :topk] * weights[:topk]).sum(-1)
                ideal_gains = weight_sums[match_counts.clip(max=topk)]
                values = gains / ideal_gains
                yield (f"{name}{topk}", values)

    def _get_places(self, scores, targets):
        assert(scores.shape[:-1] == targets.shape)
        assert(targets.dtype == torch.int64)
        
        scores = scores + torch.rand_like(scores) * self.eps
        target_scores = torch.gather(scores, 1, targets.unsqueeze(-1))
        places = (scores >= target_scores).sum(-1)

        for name in self.weights:
            weights = self.WEIGHTS[name](places)
            for topk in self.topks:
                values = (places <= topk) * weights
                yield (f"{name}{topk}", values)
