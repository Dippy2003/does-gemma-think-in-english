import os

import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(name: str = "gemma-2-2b", dtype: str = "float16") -> HookedTransformer:
    device = get_device()
    if device == "cpu" and dtype == "float16":
        dtype = "float32"  # fp16 matmul is unsupported/slow on CPU
    return HookedTransformer.from_pretrained(
        name, dtype=dtype, device=device, hf_token=os.environ.get("HF_TOKEN")
    )
