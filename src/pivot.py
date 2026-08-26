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
