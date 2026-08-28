# Llama-3.2 fertility: not generated

`meta-llama/Llama-3.2-1B` is gated on HuggingFace and the token used for this
project's other model access has not been granted access to it (403
`GatedRepoError`, confirmed 2026-08-26). Per `MASTER_PROMPT.md` section 3,
Llama-3.2 is the **fallback model**, used only if `google/gemma-2-2b` fails to
load via TransformerLens — and Gemma-2 loaded successfully (see
`results/fertility_gemma-2.parquet` and the model-loading verification in
Phase 9), so this is not a blocker for the experiment pipeline.

This file documents the gap explicitly rather than fabricating fertility
numbers or silently omitting the fourth tokenizer, per the project's honesty
rule (`MASTER_PROMPT.md`, "The honesty rule that governs the whole repo").
If access is granted later, run:

```
python scripts/run_fertility.py --tokenizer meta-llama/Llama-3.2-1B --tag llama-3.2
```
