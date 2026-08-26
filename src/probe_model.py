from sklearn.linear_model import LogisticRegression


def fit_probe(X_train, y_train, max_iter: int = 1000) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf
