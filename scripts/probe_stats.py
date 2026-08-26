#!/usr/bin/env python3
"""Print probe set statistics: row count, category breakdown, verification coverage."""
from src.io import category_counts, load_probes, verification_coverage


def main() -> None:
    df = load_probes()
    print(f"rows: {len(df)}")
    print(f"categories: {category_counts(df)}")
    print(f"verified: {verification_coverage(df):.1%}")


if __name__ == "__main__":
    main()
