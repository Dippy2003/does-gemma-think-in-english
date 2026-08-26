# Cache prewarm: partial

`app/cache.sqlite3` is committed with 8 real Gemma-2 traces (from
`results/traces.jsonl`) prewarmed via `app/cache.get_or_compute`. Full
prewarming of all 50 `app/examples.json` entries via
`scripts/prewarm_cache.sh` is pending — this repo's verified execution
environment is CPU-only (`docs/COMPUTE.md`), and a single 26-layer logit-lens
trace took ~3-4 minutes per prompt there, making a full 50-prompt prewarm
impractical within this development session (~3-4 hours of continuous CPU
inference). Run `bash scripts/prewarm_cache.sh` to complete it; the app
falls back to an offline badge (`app.offline_badge_text`) for any prompt not
yet cached.
