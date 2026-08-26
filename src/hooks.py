import torch


def run_with_cache(model, prompt: str):
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def resid_post_stack(cache, n_layers: int) -> torch.Tensor:
    """Stack resid_post for every layer: [n_layers, batch, pos, d_model]."""
    return torch.stack([cache["resid_post", layer] for layer in range(n_layers)])


def apply_final_ln(model, resid: torch.Tensor) -> torch.Tensor:
    """Apply the model's final layernorm to a residual-stream tensor."""
    return model.ln_final(resid)


def unembed(model, normed_resid: torch.Tensor) -> torch.Tensor:
    """Project a (post-final-LN) residual-stream tensor through the unembed matrix."""
    return model.unembed(normed_resid)
