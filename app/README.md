# app/

Gradio demo. `app.py` loads the model once at process startup
(`get_model()`, a module-level singleton) — never per request, since a
2B-parameter model load takes minutes on CPU. `trace.py` wraps the
`src/logit_lens.py` pipeline into a plain JSON-serializable dict for the UI
layer. `render.py` turns that dict into the tokenization strip and
layer-stack HTML. `cache.py` (Phase 27) keys every result by a SHA-256 hash
of the prompt so repeated demo inputs, and the prewarmed example set, never
re-run the model.
