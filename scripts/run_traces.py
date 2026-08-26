#!/usr/bin/env python3
"""Batch logit-lens traces over the probe set, checkpointed and OOM-guarded."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.hooks import run_with_cache
from src.io import load_probes
from src.logit_lens import build_trace, trace_to_json
from src.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2-2b")
    parser.add_argument("--out", default="results/traces.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    df = load_probes()
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            done_ids.add(json.loads(line)["id"])

    with out_path.open("a", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if row["id"] in done_ids:
                continue
            _, _, cache = run_with_cache(model, row["sinhala"])
            trace = build_trace(model, row["sinhala"], cache)
            f.write(json.dumps({"id": row["id"], **json.loads(trace_to_json(trace))}) + "\n")
            f.flush()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{i + 1}/{len(df)} done")


if __name__ == "__main__":
    main()
