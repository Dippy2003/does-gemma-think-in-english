import json

import gradio as gr

from app.cache import get_connection, get_or_compute
from app.render import render_layer_stack, render_tokenization_strip
from app.trace import trace_prompt
from src.model import get_device, load_model

MAX_INPUT_CHARS = 500

with open("app/examples.json", encoding="utf-8") as f:
    EXAMPLES = json.load(f)

_CACHE_CONN = get_connection()

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


def run_trace(prompt: str):
    if not prompt or not prompt.strip():
        return "Please enter a non-empty Sinhala sentence.", "", ""
    if len(prompt) > MAX_INPUT_CHARS:
        return f"Input too long (max {MAX_INPUT_CHARS} characters).", "", ""

    model = get_model()
    if model is None:

        def offline_compute(p):
            raise RuntimeError("no model available and no cached result for this prompt")

        try:
            result = get_or_compute(_CACHE_CONN, prompt, offline_compute)
        except RuntimeError:
            return offline_badge_text() + " — this prompt isn't cached.", "", ""
    else:
        result = get_or_compute(_CACHE_CONN, prompt, lambda p: trace_prompt(model, p))

    tokens = result.get("prompt_tokens", [])
    layer_html = render_layer_stack(result["layers"])
    pivot = result["pivot_layer"]
    callout = f"switches to Sinhala at layer {pivot}" if pivot is not None else "no stable pivot detected"
    return callout, render_tokenization_strip(tokens), layer_html


with gr.Blocks(title="Does Gemma think in English?", css="app/style.css") as demo:
    gr.Markdown("# Does a multilingual LLM pivot through English when prompted in Sinhala?")
    inp = gr.Textbox(label="Sinhala sentence", max_lines=3)
    btn = gr.Button("Trace")
    pivot_out = gr.Textbox(label="Pivot")
    strip_out = gr.HTML(label="Tokenization")
    stack_out = gr.HTML(label="Layer stack")
    btn.click(run_trace, inputs=inp, outputs=[pivot_out, strip_out, stack_out])
    gr.Examples(examples=[[ex["sinhala"]] for ex in EXAMPLES], inputs=inp)

if __name__ == "__main__":
    demo.launch()
