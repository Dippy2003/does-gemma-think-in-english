"""Activation patching: the only method in this repo that supports a causal claim.

Everything in src/logit_lens.py and src/probes.py reads activations without
touching them. Patching does the opposite: run the model on a "clean" prompt
and a "corrupted" prompt, transplant one run's activation at a chosen layer
and position into the other run, and measure whether the output changes. If
patching layer 14 flips the answer and patching layer 3 does not, layer 14 is
where the decision actually lives — a claim logit lens or probing alone
cannot support.
"""

from src.hooks import run_with_cache


def clean_corrupt_pair(model, clean_prompt: str, corrupt_prompt: str):
    """Run both prompts, returning (clean_logits, clean_cache, corrupt_logits, corrupt_cache)."""
    clean_tokens, clean_logits, clean_cache = run_with_cache(model, clean_prompt)
    corrupt_tokens, corrupt_logits, corrupt_cache = run_with_cache(model, corrupt_prompt)
    return clean_logits, clean_cache, corrupt_logits, corrupt_cache
