import numpy as np

from src.probe_model import fit_probe, random_baseline_accuracy, train_all_layers


def test_fit_probe_separates_linearly_separable_classes():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    y = np.array([0] * 20 + [1] * 20)
    clf = fit_probe(X, y)
    assert clf.score(X, y) > 0.95


def test_random_baseline_is_majority_class_rate():
    y = [0, 0, 0, 1]
    assert random_baseline_accuracy(y) == 0.75


def test_train_all_layers_returns_per_layer_accuracy():
    rng = np.random.default_rng(0)
    n_layers, n_examples, d_model = 3, 20, 4
    X = rng.normal(size=(n_layers, n_examples, d_model))
    y = [0] * 10 + [1] * 10
    train_idx = list(range(15))
    test_idx = list(range(15, 20))
    results = train_all_layers(X, y, train_idx, test_idx)
    assert set(results.keys()) == {0, 1, 2}
    for r in results.values():
        assert 0.0 <= r["test_accuracy"] <= 1.0
