"""Logit lens: decode intermediate residual-stream states through the output head.

This is a readout, not a mechanism. Logit lens shows what the unembedding
matrix would produce from an intermediate layer's residual stream — it is not
evidence that the model is "thinking" in whatever language that decode
happens to be. Only activation patching (src/patching.py) supports a causal
claim about what the model actually relies on.
"""

from src.hooks import apply_final_ln, resid_post_stack, unembed
from src.scripts import script_of


def decode_layer(model, resid_layer, position: int = -1):
    """Decode one layer's residual stream at `position` to its top token."""
    normed = apply_final_ln(model, resid_layer)
    logits = unembed(model, normed)
    token_id = logits[0, position].argmax().item()
    return model.to_single_str_token(token_id)


def sweep_layers(model, cache, position: int = -1) -> list[str]:
    """Decode every layer's resid_post at `position`, layer 0 to n_layers-1."""
    stacked = resid_post_stack(cache, model.cfg.n_layers)
    return [decode_layer(model, stacked[layer], position) for layer in range(model.cfg.n_layers)]


def decode_layer_topk(model, resid_layer, position: int = -1, k: int = 5):
    normed = apply_final_ln(model, resid_layer)
    logits = unembed(model, normed)
    top = logits[0, position].topk(k)
    return [
        (model.to_single_str_token(idx.item()), score.item())
        for idx, score in zip(top.indices, top.values)
    ]
