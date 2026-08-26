"""Pivot detection from logit-lens traces.

Like everything downstream of `src/logit_lens.py`, a detected "pivot" is a
statement about what the readout looks like layer to layer, not a claim about
the model's internal computation. Treat it as descriptive statistics on
script labels, not as evidence of translation.
"""

import pandas as pd

from src.logit_lens import Trace


def per_layer_script_distribution(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of prompts whose decoded token is each script, per layer."""
    return (
        trace_df.groupby(["layer", "script"]).size().unstack(fill_value=0).apply(
            lambda row: row / row.sum(), axis=1
        )
    )


def detect_pivot(trace: Trace, target_script: str, source_script: str = "latin") -> int | None:
    """First layer at which the decoded token's script switches from
    `source_script` to `target_script` and stays `target_script` for the rest
    of the trace. Returns None if no such stable switch occurs."""
    layers = trace.layers
    for i, lt in enumerate(layers):
        if lt.script == target_script:
            rest = layers[i:]
            if all(r.script == target_script for r in rest):
                return lt.layer
    return None


def pivot_confidence(trace: Trace, pivot_layer: int) -> float:
    """Mean top-token probability across layers from the pivot to the end.

    A pivot detected from a run of low-confidence argmax tokens is weaker
    evidence than one where the model's readout is consistently confident.
    """
    tail = [lt for lt in trace.layers if lt.layer >= pivot_layer]
    if not tail:
        return 0.0
    return sum(lt.prob for lt in tail) / len(tail)


def english_likeness_score(trace: Trace) -> list[float]:
    """Per-layer fraction-style score: 1.0 if decoded token is latin script, else 0.0.

    A smoothed/aggregated version of this over many prompts is what Wendler
    et al.'s "English pivot" curve actually looks like.
    """
    return [1.0 if lt.script == "latin" else 0.0 for lt in trace.layers]


def aggregate_pivot_stats(traces: list[Trace], target_script: str) -> dict:
    """Across many traces: how often a stable pivot occurs, and at what layer."""
    pivots = [detect_pivot(t, target_script) for t in traces]
    found = [p for p in pivots if p is not None]
    return {
        "n_traces": len(traces),
        "n_with_pivot": len(found),
        "pivot_rate": len(found) / len(traces) if traces else 0.0,
        "mean_pivot_layer": sum(found) / len(found) if found else None,
    }
