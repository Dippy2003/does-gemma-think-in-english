"""Pivot detection from logit-lens traces.

Like everything downstream of `src/logit_lens.py`, a detected "pivot" is a
statement about what the readout looks like layer to layer, not a claim about
the model's internal computation. Treat it as descriptive statistics on
script labels, not as evidence of translation.
"""

import pandas as pd


def per_layer_script_distribution(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of prompts whose decoded token is each script, per layer."""
    return (
        trace_df.groupby(["layer", "script"]).size().unstack(fill_value=0).apply(
            lambda row: row / row.sum(), axis=1
        )
    )
