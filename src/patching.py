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


def english_twin_pair(sinhala_prompt: str, english_prompt: str) -> dict:
    """The clean/corrupt pairing used throughout this repo: a Sinhala prompt
    and its literal English translation, asking the same question."""
    return {"clean": sinhala_prompt, "corrupt": english_prompt}


def patch_layer_at_position(model, corrupt_tokens, clean_cache, layer: int, position: int):
    """Run the model on `corrupt_tokens`, but with `resid_post` at `layer`,
    `position` overwritten with the value from a clean run's cache. Returns
    the patched-run logits."""

    def patch_hook(tensor, hook):
        tensor[:, position, :] = clean_cache["resid_post", layer][:, position, :]
        return tensor

    return model.run_with_hooks(
        corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
    )


def logit_diff(logits, position: int, correct_token_id: int, incorrect_token_id: int) -> float:
    """logit(correct) - logit(incorrect) at `position` — the standard patching
    metric: positive means the model favors the correct answer, and the
    magnitude of the *change* in this value from patching is the patch effect."""
    return (
        logits[0, position, correct_token_id] - logits[0, position, incorrect_token_id]
    ).item()
