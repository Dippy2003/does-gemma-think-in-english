# Deployment instructions

## HuggingFace Spaces

1. Create a Space with SDK "Gradio", pointing at `app/huggingface_space.py`
   (see `app/README_SPACE.md` for the front matter to use as the Space's
   README).
2. Add `HF_TOKEN` as a Space secret (Settings -> Repository secrets) — the
   Space's own account must separately accept the Gemma-2 license at
   https://huggingface.co/google/gemma-2-2b.
3. Commit the prewarmed `app/cache.sqlite3` (Phase 27) so the Space demos
   instantly on cold start even on CPU-only hardware, before the model
   finishes loading.
4. Free Spaces default to CPU-only. `src/model.get_device` and
   `HookedTransformer.from_pretrained_no_processing` (see
   `docs/COMPUTE.md`) are what make CPU inference actually complete instead
   of OOMing or segfaulting — verified in this repo's own dev environment,
   also CPU-only.

## Local

```
pip install -r requirements.txt
export HF_TOKEN=...
python app/app.py
```
