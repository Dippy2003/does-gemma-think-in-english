import torch

from src.hooks import resid_post_stack


class FakeCache(dict):
    def __getitem__(self, key):
        name, layer = key
        assert name == "resid_post"
        return torch.zeros(2, 5, 8)  # batch, pos, d_model


def test_resid_post_stack_shape():
    cache = FakeCache()
    n_layers = 4
    stacked = resid_post_stack(cache, n_layers)
    assert stacked.shape == (n_layers, 2, 5, 8)
