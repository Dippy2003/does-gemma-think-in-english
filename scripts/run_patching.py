#!/usr/bin/env python3
"""Checkpointed batch activation-patching sweep over the probe set."""
import argparse
import gc
import json
from pathlib import Path

import torch

from src.io import load_probes, read_done_ids
from src.model import load_model
from src.patching import layer_sweep
from src.progress import ProgressLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2-2b")
    parser.add_argument("--out", default="results/patching.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    df = load_probes()
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(out_path)

    progress = ProgressLogger(total=len(df), label="patching")
    with out_path.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if row["id"] in done_ids:
                progress.step()
                continue
            clean_tokens = model.to_tokens(row["sinhala"])
            corrupt_tokens = model.to_tokens(row["english"])
            correct_id = model.to_single_token(f" {row['answer_si']}")
            incorrect_id = model.to_single_token(f" {row['answer_en']}")
            _, clean_cache = model.run_with_cache(clean_tokens)
            results = layer_sweep(
                model,
                clean_tokens,
                corrupt_tokens,
                clean_cache,
                position=-1,
                correct_id=correct_id,
                incorrect_id=incorrect_id,
            )
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "results": [
                            {"layer": r.layer, "effect": r.effect} for r in results
                        ],
                    }
                )
                + "\n"
            )
            f.flush()
            del clean_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress.step()


if __name__ == "__main__":
    main()
