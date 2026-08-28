import os

import pandas as pd


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    try:
        return AutoTokenizer.from_pretrained(model_name, token=token)
    except ValueError:
        # some tokenizers (e.g. plain sentencepiece checkpoints) have no
        # fast (Rust) implementation available
        return AutoTokenizer.from_pretrained(model_name, token=token, use_fast=False)


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


def fragmentation_ratio(text: str, tokenizer, baseline_tokens_per_word: float = 1.3) -> float:
    """How many times more fragmented this text is than a rough English baseline."""
    tpw = tokens_per_word(text, tokenizer)
    if baseline_tokens_per_word == 0:
        return float("nan")
    return tpw / baseline_tokens_per_word


def byte_fallback_rate(text: str, tokenizer) -> float:
    """Fraction of tokens that are raw byte-fallback tokens (e.g. `<0x0D>`).

    A high rate on Sinhala/Tamil text means the tokenizer's vocabulary lacks
    dedicated subword tokens for that script and is falling back to encoding
    UTF-8 bytes one at a time — the most extreme form of fragmentation.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return 0.0
    pieces = tokenizer.convert_ids_to_tokens(ids)
    n_byte_fallback = sum(1 for p in pieces if p.startswith("<0x") and p.endswith(">"))
    return n_byte_fallback / len(pieces)
