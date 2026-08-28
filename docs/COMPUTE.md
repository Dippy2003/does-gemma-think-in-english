# Compute requirements

Everything in this repo is designed to run on a free Colab T4 (16GB VRAM):

- `google/gemma-2-2b` loads in ~5GB at fp16/bf16.
- Activations are cached to disk as parquet (`src/io.write_parquet`), never
  held for all layers x all prompts in RAM at once.
- `src/model.get_device` degrades to CPU automatically if no GPU is present;
  `src/model.select_dtype` forces float32 on CPU since fp16 matmul is
  unsupported/slow there.
- Batch runners clear the CUDA cache per prompt during large sweeps (see
  Phase 13, `scripts/run_traces.py`) to avoid OOM.

## Colab setup

```bash
!pip install -q -r requirements.txt
import os
os.environ["HF_TOKEN"] = "..."  # accept the Gemma-2 license first
```

Gemma-2 is gated — accept the license at
https://huggingface.co/google/gemma-2-2b before running anything, or every
script in this repo will fail with a `GatedRepoError`.
