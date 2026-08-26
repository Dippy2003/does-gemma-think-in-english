"""Linear probing: is language linearly decodable from the residual stream?

Linear decodability answers a narrower question than the logit lens: not
"what would the output head produce," but "does a simple linear function of
this layer's activations separate languages at all." A probe finding high
accuracy is still not a causal claim about what the model relies on — see
src/patching.py.
"""

import numpy as np

from src.hooks import resid_post_stack


def extract_activations(model, cache, position: int = -1) -> np.ndarray:
    """Extract the last-position resid_post vector at every layer.

    Returns an array of shape [n_layers, d_model].
    """
    stacked = resid_post_stack(cache, model.cfg.n_layers)  # [n_layers, batch, pos, d_model]
    return stacked[:, 0, position, :].detach().cpu().float().numpy()


def train_test_split_by_prompt(
    prompt_ids: list, labels: list, test_frac: float = 0.2, seed: int = 0
) -> tuple:
    """Split indices by unique prompt id, not by row — prevents the same
    prompt's activations from appearing in both train and test folds, which
    would silently leak label information through prompt-specific quirks."""
    rng = np.random.default_rng(seed)
    unique_ids = sorted(set(prompt_ids))
    rng.shuffle(unique_ids)
    n_test = max(1, int(len(unique_ids) * test_frac))
    test_ids = set(unique_ids[:n_test])
    train_idx = [i for i, pid in enumerate(prompt_ids) if pid not in test_ids]
    test_idx = [i for i, pid in enumerate(prompt_ids) if pid in test_ids]
    return train_idx, test_idx


def standardize(train: np.ndarray, test: np.ndarray) -> tuple:
    """Fit mean/std on train only, apply to both — avoids test-set leakage
    into the standardization statistics."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std
