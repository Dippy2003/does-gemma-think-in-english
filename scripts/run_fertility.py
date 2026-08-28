#!/usr/bin/env python3
"""CLI: measure tokenizer fertility for one tokenizer across si/ta/en corpora."""
import argparse
from pathlib import Path

from src.fertility import batch_fertility, load_tokenizer
from src.io import write_parquet

CORPORA = {
    "si": "data/fertility_corpus_si.txt",
    "ta": "data/fertility_corpus_ta.txt",
    "en": "data/fertility_corpus_en.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer repo id")
    parser.add_argument("--tag", required=True, help="short name for output files, e.g. gemma-2")
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    frames = []
    for lang, path in CORPORA.items():
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        frames.append(batch_fertility(lines, tok, language=lang, tokenizer_name=args.tag))
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    out_path = f"results/fertility_{args.tag}.parquet"
    write_parquet(df, out_path)
    print(df.groupby("language")[["tokens_per_word", "tokens_per_char"]].mean())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
