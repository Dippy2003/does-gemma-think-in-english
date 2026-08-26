import gradio as gr

from src.model import load_model

_MODEL = None


def get_model():
    """Load the model once at process startup, never per request."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model("gemma-2-2b")
    return _MODEL


with gr.Blocks(title="Does Gemma think in English?") as demo:
    gr.Markdown("# Does a multilingual LLM pivot through English when prompted in Sinhala?")

if __name__ == "__main__":
    demo.launch()
