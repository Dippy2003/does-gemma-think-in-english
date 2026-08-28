import torch

from src.logit_lens import decode_layer


class FakeModel:
    """Minimal stand-in exposing just what decode_layer needs."""

    def __init__(self, vocab):
        self.vocab = vocab  # list[str], index = token id

    def ln_final(self, resid):
        return resid  # identity: skip real layernorm for this unit test

    def unembed(self, resid):
        # resid: [1, seq, d_model] -> one-hot-ish logits picking vocab index
        # equal to the residual's own first dim value, for a deterministic test
        batch, seq, d_model = resid.shape
        logits = torch.zeros(batch, seq, len(self.vocab))
        for s in range(seq):
            idx = int(resid[0, s, 0].item())
            logits[0, s, idx] = 10.0
        return logits

    def to_single_str_token(self, token_id):
        return self.vocab[token_id]


def test_decode_layer_picks_argmax_token():
    model = FakeModel(vocab=["the", "cat", "sat"])
    resid = torch.zeros(1, 1, 4)
    resid[0, 0, 0] = 1  # points at "cat"
    assert decode_layer(model, resid, position=-1) == "cat"
