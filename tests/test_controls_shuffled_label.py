import numpy as np

from src.controls import shuffled_label_probe_control
from src.probe_model import fit_probe


def test_shuffled_labels_probe_near_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 8))
    y = [0] * 30 + [1] * 30
    y_shuffled = shuffled_label_probe_control(y, seed=1)

    clf = fit_probe(X, y_shuffled)
    acc = clf.score(X, y_shuffled)
    # random noise features can't reach much above chance even after fitting
    assert acc < 0.85
