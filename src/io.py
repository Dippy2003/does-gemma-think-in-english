from pathlib import Path

import pandas as pd

PROBE_COLUMNS = [
    "id",
    "sinhala",
    "tamil",
    "english",
    "answer_si",
    "answer_ta",
    "answer_en",
    "category",
    "verified",
    "verifier_note",
]

CATEGORIES = ["factual", "spatial", "temporal", "kinship", "numeric"]


def load_probes(path: str | Path = "data/parallel_probes.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = set(PROBE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"probe set missing columns: {missing}")
    df["verified"] = df["verified"].fillna("false").str.lower() == "true"
    return df
