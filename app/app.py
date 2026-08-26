import gradio as gr

from src.model import get_device, load_model

_MODEL = None
_MODEL_LOAD_FAILED = False


def get_model():
    """Load the model once at process startup, never per request.

    If loading fails (e.g. no GPU and no way to load in time), the app falls
    back to cache-only mode and shows an offline badge rather than crashing
    on every request.
    """
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is None and not _MODEL_LOAD_FAILED:
        try:
            _MODEL = load_model("gemma-2-2b")
        except Exception:  # noqa: BLE001
            _MODEL_LOAD_FAILED = True
    return _MODEL


def is_offline() -> bool:
    return _MODEL is None


def offline_badge_text() -> str:
    device = get_device()
    return f"⚠️ offline — serving cached results only (device: {device})"


with gr.Blocks(title="Does Gemma think in English?") as demo:
    gr.Markdown("# Does a multilingual LLM pivot through English when prompted in Sinhala?")

if __name__ == "__main__":
    demo.launch()
