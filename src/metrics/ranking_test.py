import unittest
import torch

from ranking import Ranking


class TestRanking(unittest.TestCase):

    def test_ranking_single(self):
        ranking = Ranking({
            "topks": [3],
            "weights": ["recall", "ndcg", "mrr"],
            "is_dist": False,
            "is_excl": False,
            "is_multi": False,
            "eps": 1e-5,
        })
        inputs = None
        outputs = torch.tensor([
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
        ], dtype=torch.float)
        targets = torch.tensor([0, 1, 2, 3])
        for actual, expect in zip(ranking(inputs, outputs, targets), [
            ("recall3", torch.tensor([1., 1., 1., 0.])),
            ("ndcg3", torch.tensor([1., 0.6309, 0.5, 0.])),
            ("mrr3", torch.tensor([1., 0.5, 0.3333, 0.])),
        ]):
            self.assertEqual(actual[0], expect[0])
            self.assertTrue((actual[1] - expect[1]).square().mean().sqrt() <= 1e-4)

    def test_ranking_multi(self):
        topk = 2
        ranking = Ranking({
            "topks": [topk],
            "weights": ["recall", "ndcg", "mrr"],
            "is_dist": False,
            "is_excl": False,
            "is_multi": True,
            "eps": 1e-5,
        })
        inputs = None
        outputs = torch.arange(topk+1).neg().float()\
            .unsqueeze(0).expand(1<<(topk+1), topk+1)
        targets = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=torch.bool)
        for actual, expect in zip(ranking(inputs, outputs, targets), [
            ("recall2", torch.tensor([0., 0., 1., 0.5, 1., 0.5, 1., 1.])),
            ("ndcg2", torch.tensor([0., 0., 0.6309, 0.3869, 1., 0.6131, 1., 1.])),
            ("mrr2", torch.tensor([0., 0., 0.5, 0.3333, 1., 0.6667, 1., 1.])),
        ]):
            self.assertEqual(actual[0], expect[0])
            self.assertTrue((actual[1] - expect[1]).square().mean().sqrt() <= 1e-4)


if __name__ == '__main__':
    unittest.main()