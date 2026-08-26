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
