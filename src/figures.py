from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_COLORS = {
    "sinhala": "#d95f02",
    "tamil": "#1b9e77",
    "english": "#7570b3",
    "latin": "#7570b3",
    "devanagari": "#e7298a",
    "other": "#999999",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 11


def fertility_bar_chart(table, value: str = "tokens_per_word"):
    """Grouped bar chart: tokenizer x language, for `value`."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = table.pivot(index="tokenizer", columns="language", values=value)
    pivot.plot.bar(ax=ax)
    ax.set_ylabel(value.replace("_", " "))
    ax.set_xlabel("tokenizer")
    ax.legend(title="language")
    fig.tight_layout()
    return fig


def caption(text: str, n: int) -> str:
    """Standard figure-caption prefix, e.g. 'Figure 3. <text>'."""
    return f"Figure {n}. {text}"


def save_figure(fig, name: str, formats=("png", "pdf")) -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(f"figures/{name}.{fmt}", bbox_inches="tight")


def layer_wise_language_plot(script_dist_df, ax=None):
    """Stacked area plot: fraction of readouts per script, by layer."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    colors = [SCRIPT_COLORS.get(c, "#999999") for c in script_dist_df.columns]
    ax.stackplot(
        script_dist_df.index, script_dist_df.T.values, labels=script_dist_df.columns, colors=colors
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("fraction of readouts")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def multi_language_overlay_plot(curves: dict):
    """Overlay english-likeness (or similar) curves for multiple prompt languages."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, series in curves.items():
        ax.plot(range(len(series)), series, marker="o", markersize=3, label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    return fig


def pivot_histogram(pivot_layers: list):
    """Histogram of detected pivot layers across the probe set."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([p for p in pivot_layers if p is not None], bins=20, color="#7570b3")
    ax.set_xlabel("pivot layer")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def script_by_layer_heatmap(script_dist_df):
    """Heatmap: rows = script, columns = layer, values = readout fraction."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(script_dist_df.T, cmap="viridis", ax=ax, cbar_kws={"label": "fraction"})
    ax.set_xlabel("layer")
    ax.set_ylabel("script")
    fig.tight_layout()
    return fig


def probe_accuracy_figure(accuracy_curve_df, baseline: float | None = None):
    """Line plot of probe test accuracy per layer, with an optional baseline line."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(accuracy_curve_df["layer"], accuracy_curve_df["test_accuracy"], marker="o")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="gray", label=f"baseline ({baseline:.2f})")
        ax.legend()
    ax.set_xlabel("layer")
    ax.set_ylabel("probe test accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def patch_heatmap(grid_df):
    """Heatmap of patch effect, layer x position."""
    set_style()
    pivot = grid_df.pivot(index="layer", columns="position", values="effect")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "patch effect"})
    ax.set_xlabel("position")
    ax.set_ylabel("layer")
    fig.tight_layout()
    return fig


def fertility_distribution_violin(df, value: str = "tokens_per_word"):
    """Violin plot of per-line fertility, split by language."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="language", y=value, ax=ax)
    ax.set_ylabel(value.replace("_", " "))
    fig.tight_layout()
    return fig
