import os

import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(device: str, vram_gb: float | None = None) -> str:
    if device == "cpu":
        return "float32"
    if vram_gb is not None and vram_gb < 10:
        return "float16"
    return "bfloat16"


def load_model(name: str = "gemma-2-2b", dtype: str | None = None) -> HookedTransformer:
    device = get_device()
    if dtype is None:
        vram_gb = None
        if device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        dtype = select_dtype(device, vram_gb)
    return HookedTransformer.from_pretrained(
        name, dtype=dtype, device=device, hf_token=os.environ.get("HF_TOKEN")
    )
