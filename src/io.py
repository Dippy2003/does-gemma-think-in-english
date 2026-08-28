from pathlib import Path

import pandas as pd


def read_done_ids(jsonl_path: str | Path) -> set:
    """IDs already present in a checkpointed .jsonl batch-run output file."""
    import json

    path = Path(jsonl_path)
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def patching_jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> pd.DataFrame:
    import json

    rows = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for r in rec["results"]:
            rows.append({"id": rec["id"], **r})
    df = pd.DataFrame(rows)
    write_parquet(df, parquet_path)
    return df


def traces_jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> pd.DataFrame:
    """Flatten a batch-trace .jsonl file (one row per prompt, layers nested) into
    a long-format parquet table (one row per prompt x layer)."""
    import json

    rows = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for layer in rec["layers"]:
            rows.append({"id": rec["id"], "prompt": rec["prompt"], **layer})
    df = pd.DataFrame(rows)
    write_parquet(df, parquet_path)
    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

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
