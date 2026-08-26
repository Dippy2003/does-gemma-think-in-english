import os


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
