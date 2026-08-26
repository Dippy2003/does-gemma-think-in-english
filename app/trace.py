from src.hooks import run_with_cache
from src.logit_lens import Trace, build_trace
from src.pivot import detect_pivot


def trace_prompt(model, prompt: str) -> dict:
    """Run the full trace pipeline for one prompt, returning a plain dict
    (JSON-serializable) for the Gradio UI layer."""
    tokens, logits, cache = run_with_cache(model, prompt)
    trace: Trace = build_trace(model, prompt, cache)
    pivot_layer = detect_pivot(trace, target_script="sinhala", source_script="latin")
    return {
        "prompt": prompt,
        "n_tokens": tokens.shape[-1],
        "layers": [
            {"layer": lt.layer, "token": lt.token, "script": lt.script, "prob": lt.prob}
            for lt in trace.layers
        ],
        "pivot_layer": pivot_layer,
    }
