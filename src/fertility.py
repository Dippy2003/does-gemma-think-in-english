import os


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
