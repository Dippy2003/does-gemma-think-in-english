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


def single_token_report(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """For each answer column, report which rows tokenize to more than one token.

    Multi-token answers make logit-lens argmax comparisons ambiguous, so probes
    with multi-token answers should be flagged (not silently dropped).
    """
    out = df[["id"]].copy()
    for col in ["answer_si", "answer_ta", "answer_en"]:
        out[f"{col}_n_tokens"] = df[col].fillna("").apply(
            lambda s: len(tokenizer.encode(s, add_special_tokens=False))
        )
    return out


def category_counts(df: pd.DataFrame) -> dict:
    """Row count per probe category, for coverage reporting."""
    return df["category"].value_counts().to_dict()


def verification_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows with verified == True."""
    if len(df) == 0:
        return 0.0
    return df["verified"].sum() / len(df)
