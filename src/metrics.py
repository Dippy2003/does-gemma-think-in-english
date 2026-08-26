import pandas as pd


def fertility_comparison_table(fertility_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-tokenizer fertility results into one tokenizer x language table."""
    rows = []
    for tag, df in fertility_dfs.items():
        summary = df.groupby("language")[["tokens_per_word", "tokens_per_char"]].mean()
        for lang, row in summary.iterrows():
            rows.append(
                {
                    "tokenizer": tag,
                    "language": lang,
                    "tokens_per_word": row["tokens_per_word"],
                    "tokens_per_char": row["tokens_per_char"],
                }
            )
    return pd.DataFrame(rows)


def per_language_distribution(df: pd.DataFrame, column: str = "tokens_per_word") -> pd.DataFrame:
    """Mean, std, min, max of a fertility column, grouped by language."""
    return df.groupby("language")[column].agg(["mean", "std", "min", "max"])


def cross_model_comparison(pivot_stats_by_model: dict) -> pd.DataFrame:
    """Combine aggregate_pivot_stats() results from multiple models into one table."""
    rows = []
    for model_name, stats in pivot_stats_by_model.items():
        rows.append({"model": model_name, **stats})
    return pd.DataFrame(rows)
