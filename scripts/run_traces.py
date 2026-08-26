#!/usr/bin/env python3
"""Batch logit-lens traces over the probe set, checkpointed and OOM-guarded."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.hooks import run_with_cache
from src.io import load_probes, read_done_ids
from src.logit_lens import build_trace, trace_to_json
from src.model import load_model
from src.progress import ProgressLogger


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
    done_ids = read_done_ids(out_path)
    if done_ids:
        print(f"resuming: {len(done_ids)} rows already done")

    progress = ProgressLogger(total=len(df), label="traces")
    with out_path.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if row["id"] in done_ids:
                progress.step()
                continue
            cache = None
            try:
                _, _, cache = run_with_cache(model, row["sinhala"])
                trace = build_trace(model, row["sinhala"], cache)
                f.write(json.dumps({"id": row["id"], **json.loads(trace_to_json(trace))}) + "\n")
                f.flush()
            finally:
                # clear the reference before collecting, or the cache tensor
                # stays reachable through this frame and gc.collect() is a no-op
                del cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            progress.step()


if __name__ == "__main__":
    main()
