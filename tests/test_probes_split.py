import numpy as np

from src.probes import standardize, train_test_split_by_prompt


def test_split_no_leakage_between_folds():
    prompt_ids = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    train_idx, test_idx = train_test_split_by_prompt(prompt_ids, labels, test_frac=0.4, seed=0)
    train_ids = {prompt_ids[i] for i in train_idx}
    test_ids = {prompt_ids[i] for i in test_idx}
    assert train_ids.isdisjoint(test_ids)
    assert set(train_idx) | set(test_idx) == set(range(len(prompt_ids)))


def test_standardize_uses_train_stats_only():
    train = np.array([[0.0], [2.0]])
    test = np.array([[10.0]])
    train_std, test_std = standardize(train, test)
    assert np.isclose(train_std.mean(), 0.0, atol=1e-6)
    assert test_std[0, 0] > 1.0  # test point is far outside train distribution
