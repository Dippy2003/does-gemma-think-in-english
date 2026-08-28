#!/usr/bin/env python3
"""Generate pivot figures from a traces parquet file."""
import argparse

from src.figures import (
    layer_wise_language_plot,
    pivot_histogram,
    save_figure,
    script_by_layer_heatmap,
)
from src.io import read_parquet
from src.pivot import per_layer_script_distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="results/traces.parquet")
    args = parser.parse_args()

    df = read_parquet(args.traces)
    dist = per_layer_script_distribution(df)

    save_figure(layer_wise_language_plot(dist), "pivot_layer_language")
    save_figure(script_by_layer_heatmap(dist), "pivot_script_heatmap")
    print("wrote figures/pivot_layer_language.{png,pdf}, figures/pivot_script_heatmap.{png,pdf}")


if __name__ == "__main__":
    main()
