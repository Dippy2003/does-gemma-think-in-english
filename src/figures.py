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


def fertility_distribution_violin(df, value: str = "tokens_per_word"):
    """Violin plot of per-line fertility, split by language."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="language", y=value, ax=ax)
    ax.set_ylabel(value.replace("_", " "))
    fig.tight_layout()
    return fig
