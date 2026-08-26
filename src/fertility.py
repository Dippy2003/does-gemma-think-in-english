import os

import pandas as pd


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))


def tokens_per_word(text: str, tokenizer) -> float:
    """Mean number of tokens per whitespace-delimited word."""
    words = text.split()
    if not words:
        return 0.0
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens / len(words)


def tokens_per_char(text: str, tokenizer) -> float:
    """Mean number of tokens per character (whitespace excluded)."""
    n_chars = len(text.replace(" ", ""))
    if n_chars == 0:
        return 0.0
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens / n_chars


def batch_fertility(lines: list[str], tokenizer, language: str, tokenizer_name: str) -> pd.DataFrame:
    """Per-line tokens-per-word and tokens-per-char for a corpus."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rows.append(
            {
                "language": language,
                "tokenizer": tokenizer_name,
                "text": line,
                "tokens_per_word": tokens_per_word(line, tokenizer),
                "tokens_per_char": tokens_per_char(line, tokenizer),
            }
        )
    return pd.DataFrame(rows)
