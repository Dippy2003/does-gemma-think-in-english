"""Unified loading of every results artifact this repo produces."""

from pathlib import Path

from src.io import read_parquet

RESULT_FILES = {
    "fertility_gemma-2": "results/fertility_gemma-2.parquet",
    "fertility_qwen2.5": "results/fertility_qwen2.5.parquet",
    "fertility_gpt-oss": "results/fertility_gpt-oss.parquet",
    "traces": "results/traces.parquet",
    "probe_accuracy": "results/probe_accuracy_gemma-2.parquet",
    "patching": "results/patching.parquet",
}


def load_all_results() -> dict:
    """Load every results artifact that currently exists on disk."""
    out = {}
    for name, path in RESULT_FILES.items():
        if Path(path).exists():
            out[name] = read_parquet(path)
    return out


def summary_table(results: dict) -> "pd.DataFrame":
    """One row per results artifact: name, row count, column list."""
    import pandas as pd

    rows = []
    for name, df in results.items():
        rows.append({"artifact": name, "n_rows": len(df), "columns": ", ".join(df.columns)})
    return pd.DataFrame(rows)
