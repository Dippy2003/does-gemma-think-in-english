import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def fit_probe(X_train, y_train, max_iter: int = 1000) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf


def cross_validated_accuracy(X_layer: np.ndarray, y, cv: int = 5) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000)
    return cross_val_score(clf, X_layer, y, cv=cv)


def train_all_layers(X: np.ndarray, y, train_idx, test_idx) -> dict:
    """Train one probe per layer. X: [n_layers, n_examples, d_model]."""
    results = {}
    for layer in range(X.shape[0]):
        clf = fit_probe(X[layer][train_idx], np.array(y)[train_idx])
        acc = clf.score(X[layer][test_idx], np.array(y)[test_idx])
        results[layer] = {"clf": clf, "test_accuracy": acc}
    return results


def probe_accuracy_by_layer(results: dict) -> dict:
    """Flatten train_all_layers() output to {layer: test_accuracy}."""
    return {layer: r["test_accuracy"] for layer, r in results.items()}


def random_baseline_accuracy(y) -> float:
    """Accuracy of predicting the majority class, ignoring activations entirely.

    Any layer's probe accuracy must clear this, not 1/n_classes, since
    real-world label distributions are rarely uniform.
    """
    y = np.array(y)
    values, counts = np.unique(y, return_counts=True)
    majority = values[np.argmax(counts)]
    return float(np.mean(y == majority))


def probe_results_to_dataframe(results: dict, model_name: str = "gemma-2") -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(
        [
            {"model": model_name, "layer": layer, "test_accuracy": r["test_accuracy"]}
            for layer, r in results.items()
        ]
    )


def weight_norms(results: dict) -> dict:
    """L2 norm of each layer's probe weight vector.

    A layer where the probe relies on a small number of large-magnitude
    directions vs. many small ones is a coarse hint about how distributed the
    language signal is at that layer — not a causal claim.
    """
    return {layer: float(np.linalg.norm(r["clf"].coef_)) for layer, r in results.items()}


def layer_confusion_matrix(clf, X_layer_test, y_test):
    y_pred = clf.predict(X_layer_test)
    return confusion_matrix(y_test, y_pred, labels=clf.classes_)
