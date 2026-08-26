#!/usr/bin/env python3
"""Run the batch trace pipeline against the replication model (Qwen2.5-1.5B)."""
import subprocess
import sys

from src.model import MODEL_REGISTRY


def main() -> None:
    model = MODEL_REGISTRY["replication"]
    out = "results/traces_qwen2.5.jsonl"
    subprocess.run(
        [sys.executable, "scripts/run_traces.py", "--model", model, "--out", out],
        check=True,
    )


if __name__ == "__main__":
    main()
