# Patching results: pending full sweep

`results/patching.parquet` and `figures/patch_heatmap.png` /
`figures/layerwise_effect.png` are produced by `scripts/run_patching.py`
followed by `src/io.patching_jsonl_to_parquet`. A full layer-by-position
sweep on CPU (this repo's verified execution environment — see
`docs/COMPUTE.md`) is expensive: each sweep re-runs the model once per layer
per prompt. This commit reserves the output path; see
`results/PROBE_RESULTS_STATUS.md` for the same caveat applied to probing.
