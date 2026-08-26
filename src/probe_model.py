from sklearn.linear_model import LogisticRegression


def fit_probe(X_train, y_train, max_iter: int = 1000) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf


def train_all_layers(X: "np.ndarray", y, train_idx, test_idx) -> dict:
    """Train one probe per layer. X: [n_layers, n_examples, d_model]."""
    import numpy as np

    results = {}
    for layer in range(X.shape[0]):
        clf = fit_probe(X[layer][train_idx], np.array(y)[train_idx])
        acc = clf.score(X[layer][test_idx], np.array(y)[test_idx])
        results[layer] = {"clf": clf, "test_accuracy": acc}
    return results
