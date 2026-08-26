import torch


def run_with_cache(model, prompt: str):
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def resid_post_stack(cache, n_layers: int) -> torch.Tensor:
    """Stack resid_post for every layer: [n_layers, batch, pos, d_model]."""
    return torch.stack([cache["resid_post", layer] for layer in range(n_layers)])
