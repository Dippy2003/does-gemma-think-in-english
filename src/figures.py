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
