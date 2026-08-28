import torch

from src.patching import logit_diff


def test_logit_diff_sign():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 2] = 3.0  # correct
    logits[0, 0, 1] = 1.0  # incorrect
    assert logit_diff(logits, position=0, correct_token_id=2, incorrect_token_id=1) == 2.0


def test_logit_diff_identity_when_equal():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 0] = 5.0
    logits[0, 0, 1] = 5.0
    assert logit_diff(logits, position=0, correct_token_id=0, incorrect_token_id=1) == 0.0
