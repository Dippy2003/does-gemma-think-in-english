#!/usr/bin/env python3
"""Print master statistics across every results artifact currently on disk."""
from src.results import load_all_results, summary_table


def main() -> None:
    results = load_all_results()
    if not results:
        print("no results artifacts found on disk yet")
        return
    print(summary_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
