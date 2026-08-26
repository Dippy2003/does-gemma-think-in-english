import os

import torch
from transformer_lens import HookedTransformer

MODEL_REGISTRY = {
    "primary": "gemma-2-2b",
    "replication": "Qwen/Qwen2.5-1.5B",
    "fallback": "meta-llama/Llama-3.2-1B",
}


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


def load_with_fallback(role: str = "primary", dtype: str | None = None) -> HookedTransformer:
    """Try MODEL_REGISTRY[role]; on failure to load, try the 'fallback' entry.

    Falls back loudly: prints which model actually loaded so a caller never
    silently ends up analyzing the wrong model.
    """
    name = MODEL_REGISTRY[role]
    try:
        model = load_model(name, dtype=dtype)
        print(f"loaded {name}")
        return model
    except Exception as e:  # noqa: BLE001
        if role == "fallback":
            raise
        print(f"failed to load {name} ({e}); falling back to {MODEL_REGISTRY['fallback']}")
        return load_with_fallback("fallback", dtype=dtype)


def report_config(model: HookedTransformer) -> dict:
    return {
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "n_ctx": model.cfg.n_ctx,
        "device": str(model.cfg.device),
        "dtype": str(model.cfg.dtype),
    }
