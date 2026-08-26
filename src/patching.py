"""Activation patching: the only method in this repo that supports a causal claim.

Everything in src/logit_lens.py and src/probes.py reads activations without
touching them. Patching does the opposite: run the model on a "clean" prompt
and a "corrupted" prompt, transplant one run's activation at a chosen layer
and position into the other run, and measure whether the output changes. If
patching layer 14 flips the answer and patching layer 3 does not, layer 14 is
where the decision actually lives — a claim logit lens or probing alone
cannot support.

This is the only module in the repo where a result can be reported as
"the model relies on this representation," rather than "the readout at this
layer looks like this." Every other module's output should be read as
suggestive of where to patch, not as a finding on its own.
"""

from dataclasses import dataclass

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


def patch_positions_at_layer(
    model, corrupt_tokens, clean_cache, layer: int, positions: list[int]
):
    """Patch several positions at once, at a single layer."""

    def patch_hook(tensor, hook):
        for pos in positions:
            tensor[:, pos, :] = clean_cache["resid_post", layer][:, pos, :]
        return tensor

    return model.run_with_hooks(
        corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
    )


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


@dataclass
class PatchResult:
    layer: int
    position: int
    clean_logit_diff: float
    corrupt_logit_diff: float
    patched_logit_diff: float

    @property
    def effect(self) -> float:
        """Normalized recovery: 0 = no effect (still like corrupt run), 1 =
        fully restores the clean run's logit diff."""
        span = self.clean_logit_diff - self.corrupt_logit_diff
        if span == 0:
            return 0.0
        return (self.patched_logit_diff - self.corrupt_logit_diff) / span


def layer_sweep(
    model, clean_tokens, corrupt_tokens, clean_cache, position, correct_id, incorrect_id
) -> list[PatchResult]:
    """Patch every layer at a fixed position, one at a time.

    Runs under torch.no_grad(): none of this needs gradients, and without it
    every one of the n_layers forward passes below builds and retains an
    autograd graph, multiplying peak memory by roughly n_layers for no
    benefit.
    """
    import torch

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        corrupt_logits = model(corrupt_tokens)
        clean_ld = logit_diff(clean_logits, position, correct_id, incorrect_id)
        corrupt_ld = logit_diff(corrupt_logits, position, correct_id, incorrect_id)

        results = []
        for layer in range(model.cfg.n_layers):
            patched_logits = patch_layer_at_position(
                model, corrupt_tokens, clean_cache, layer, position
            )
            patched_ld = logit_diff(patched_logits, position, correct_id, incorrect_id)
            results.append(
                PatchResult(
                    layer=layer,
                    position=position,
                    clean_logit_diff=clean_ld,
                    corrupt_logit_diff=corrupt_ld,
                    patched_logit_diff=patched_ld,
                )
            )
    return results


def layer_by_position_grid(
    model, clean_tokens, corrupt_tokens, clean_cache, n_positions, correct_id, incorrect_id
) -> list[PatchResult]:
    """Patch every (layer, position) combination — the full causal map."""
    clean_logits = model(clean_tokens)
    corrupt_logits = model(corrupt_tokens)
    clean_ld = logit_diff(clean_logits, n_positions - 1, correct_id, incorrect_id)
    corrupt_ld = logit_diff(corrupt_logits, n_positions - 1, correct_id, incorrect_id)

    results = []
    for layer in range(model.cfg.n_layers):
        for position in range(n_positions):
            patched_logits = patch_layer_at_position(
                model, corrupt_tokens, clean_cache, layer, position
            )
            patched_ld = logit_diff(patched_logits, n_positions - 1, correct_id, incorrect_id)
            results.append(
                PatchResult(
                    layer=layer,
                    position=position,
                    clean_logit_diff=clean_ld,
                    corrupt_logit_diff=corrupt_ld,
                    patched_logit_diff=patched_ld,
                )
            )
    return results
