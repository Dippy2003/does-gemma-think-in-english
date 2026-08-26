"""Logit lens: decode intermediate residual-stream states through the output head.

This is a readout, not a mechanism. At each layer we take the residual
stream, apply the model's final layernorm, and project through the unembed
matrix — the same computation the model performs to produce its actual
output, just run early. A layer's argmax token tells you what the output
head *would* say if the network stopped there. It does not tell you that the
network is internally "thinking" in that language, that this layer computed
that token, or that the model relies on this representation causally — the
residual stream at every layer is later read, transformed, and possibly
overwritten by everything downstream. The only method in this repo that
supports a causal claim is activation patching (src/patching.py): does
transplanting this layer's activation into a different run change the
output? Logit lens results should be reported as "the readout at layer N is
X," never as "the model is doing X at layer N."
"""

from dataclasses import dataclass, field

import torch

from src.hooks import apply_final_ln, resid_post_stack, unembed
from src.scripts import script_of


@dataclass
class LayerTrace:
    layer: int
    token: str
    script: str
    prob: float


@dataclass
class Trace:
    prompt: str
    layers: list[LayerTrace] = field(default_factory=list)


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


def layer_probabilities(model, resid_layer, position: int = -1) -> torch.Tensor:
    """Softmax probability distribution over the vocabulary at one layer."""
    normed = apply_final_ln(model, resid_layer)
    logits = unembed(model, normed)
    return torch.softmax(logits[0, position], dim=-1)


def build_trace(model, prompt: str, cache, position: int = -1) -> Trace:
    stacked = resid_post_stack(cache, model.cfg.n_layers)
    trace = Trace(prompt=prompt)
    for layer in range(model.cfg.n_layers):
        token = decode_layer(model, stacked[layer], position)
        probs = layer_probabilities(model, stacked[layer], position)
        trace.layers.append(
            LayerTrace(layer=layer, token=token, script=script_of(token), prob=probs.max().item())
        )
    return trace


def print_trace_table(trace: Trace) -> None:
    print(f"prompt: {trace.prompt!r}")
    print(f"{'layer':>5}  {'token':<20}  {'script':<12}  {'prob':>6}")
    for lt in trace.layers:
        print(f"{lt.layer:>5}  {lt.token[:20]:<20}  {lt.script:<12}  {lt.prob:>6.3f}")
