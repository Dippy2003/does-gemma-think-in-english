import torch


def run_with_cache(model, prompt: str):
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache
