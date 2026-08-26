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


def accuracy_curve(probe_results_df: pd.DataFrame) -> pd.DataFrame:
    """Layer -> test accuracy, sorted, ready for plotting."""
    return probe_results_df.sort_values("layer")[["layer", "test_accuracy"]]


def per_language_distribution(df: pd.DataFrame, column: str = "tokens_per_word") -> pd.DataFrame:
    """Mean, std, min, max of a fertility column, grouped by language."""
    return df.groupby("language")[column].agg(["mean", "std", "min", "max"])


def summarize_scores(scores) -> dict:
    """Mean/std/min/max of a 1D array of scores — shared by fertility,
    probe-accuracy, and patching-effect summaries so each module doesn't
    reimplement the same five-line aggregation."""
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def aggregate_patch_effects(patching_df: pd.DataFrame) -> pd.DataFrame:
    """Mean patch effect per layer, aggregated across every prompt in the sweep."""
    return patching_df.groupby("layer")["effect"].agg(["mean", "std", "count"]).reset_index()


def causal_bottleneck_layer(agg_effects: pd.DataFrame) -> int:
    """The layer with the single largest mean patch effect — where the
    output decision most concentrates, per the patching sweep."""
    return int(agg_effects.loc[agg_effects["mean"].idxmax(), "layer"])


def compare_to_logit_lens_pivot(causal_layer: int, logit_lens_pivot_layer) -> dict:
    """How far apart the causal bottleneck (patching) and the correlational
    pivot (logit lens) are — the single most important sanity check in this
    repo. A large gap means the readout and the actual causal mechanism
    disagree about where the "decision" happens."""
    if logit_lens_pivot_layer is None:
        return {"causal_layer": causal_layer, "logit_lens_pivot_layer": None, "gap": None}
    return {
        "causal_layer": causal_layer,
        "logit_lens_pivot_layer": logit_lens_pivot_layer,
        "gap": causal_layer - logit_lens_pivot_layer,
    }


def cross_model_comparison(pivot_stats_by_model: dict) -> pd.DataFrame:
    """Combine aggregate_pivot_stats() results from multiple models into one table."""
    rows = []
    for model_name, stats in pivot_stats_by_model.items():
        rows.append({"model": model_name, **stats})
    return pd.DataFrame(rows)
