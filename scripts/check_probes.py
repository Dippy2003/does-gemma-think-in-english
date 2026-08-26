#!/usr/bin/env python3
"""Gate script: fail the run if fewer than 80% of probe rows are verified."""
import sys

from src.io import load_probes, verification_coverage

MIN_COVERAGE = 0.80


def main() -> int:
    df = load_probes()
    coverage = verification_coverage(df)
    print(f"probe verification coverage: {coverage:.1%} ({df['verified'].sum()}/{len(df)})")
    if coverage < MIN_COVERAGE:
        print(f"FAIL: coverage below {MIN_COVERAGE:.0%} gate", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
