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
