import os

from transformer_lens import HookedTransformer


def load_model(name: str = "gemma-2-2b", dtype: str = "float16") -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        name, dtype=dtype, hf_token=os.environ.get("HF_TOKEN")
    )
