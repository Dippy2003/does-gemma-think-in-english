# Demo usage

```
python app/app.py
```

Type or paste a Sinhala sentence, or click one of the 50 preloaded examples
(`app/examples.json`) — examiners click those the large majority of the
time, and they're guaranteed to work even offline since
`scripts/prewarm_cache.sh` populates `app/cache.sqlite3` ahead of time.

Three panels:

1. **Tokenization strip** — the sentence split into actual tokenizer pieces.
2. **Layer stack** — per-layer logit-lens decode, colour-coded by script.
3. **Pivot callout** — the detected pivot layer, or "no stable pivot
   detected" if none is found (see `src/pivot.py`).

Input is capped at 500 characters and rejected if empty. If no GPU is
available and the model fails to load, the app serves cached results only
and shows an offline badge (`offline_badge_text()`).
