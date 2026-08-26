# Does a multilingual LLM pivot through English when prompted in Sinhala?

## Introduction

Wendler et al. (2024) found that decoding the residual stream of Llama-family
models at middle layers, via the logit lens, produces English-like tokens
even when neither the prompt nor the target output is English — evidence
that these models' internal concept space is biased toward English, with
target-language identity applied late, close to the output. Every language
tested so far — French, German, Chinese — is high-resource and mostly
Latin- or Han-script. Sinhala is neither: it is low-resource, Brahmic-script
(U+0D80–U+0DFF), morphologically rich, and — as this repo's own fertility
measurements show (Phase 8) — heavily under-tokenized by every tokenizer
tested. This project asks whether the English-pivot phenomenon replicates
under these conditions, weakens, disappears, or is replaced by a pivot
through a third, script- or lexically-adjacent language such as Hindi.

## Related work

Wendler et al. (2024), *Do Llamas Work in English? On the Latent Language of
Multilingual Transformers*, is the direct precedent for this repo's logit-lens
methodology (`src/logit_lens.py`) and its explicit distinction between a
correlational readout and a causal mechanism — a distinction this repo
enforces throughout (see every module docstring in `src/`) and resolves only
via activation patching (`src/patching.py`), following the broader
activation-patching / causal-tracing literature this repo does not attempt
to survey exhaustively.

## Methods

### Tokenizer fertility

Tokenizer fertility — tokens per word and tokens per character — is measured
first, as a control rather than a finding. If Sinhala costs substantially more
tokens per word than English under a given tokenizer, the model has more
sequence positions to work with per unit of meaning, and any apparent
difference in logit-lens pivot depth between languages could be a mechanical
artifact of tokenization rather than a representational difference. Fertility
is measured across four tokenizers (Gemma-2, Qwen2.5, Llama-3.2, and
gpt-oss's o200k_harmony) on parallel Sinhala/Tamil/English corpora, using
`tokens_per_word` and `tokens_per_char` (`src/fertility.py`), before any
logit-lens or probing result is interpreted.

### Models

Primary: `google/gemma-2-2b` (26 layers, d_model 2304), loaded via
TransformerLens (`src/model.py`). Replication: `Qwen/Qwen2.5-1.5B`, a
different pretraining mix and tokenizer, used to check that any finding is
not a Gemma-2 idiosyncrasy. `meta-llama/Llama-3.2-1B` is the fallback model,
used only if Gemma-2 fails to load — it did not, in this repo's own
verification (Phase 9), so all reported model-level results are Gemma-2.

### Logit lens

At each layer, `resid_post` is read from the residual stream, passed through
the model's final layernorm and unembedding matrix, and the argmax token and
its Unicode script are recorded (`src/logit_lens.py`, `src/scripts.py`). This
is a readout of what the output head would produce from that intermediate
state — correlational only, never treated as a mechanistic claim (see that
module's docstring for the full argument).

### Linear probes

A logistic regression probe is trained per layer on last-token `resid_post`
activations to predict prompt language (`src/probes.py`,
`src/probe_model.py`). This answers a narrower question than the logit lens
— is the language signal linearly present — not whether the model relies on
it.

### Activation patching

The only causal method in this repo (`src/patching.py`). A Sinhala prompt and
its literal English twin are each run through the model; one run's
`resid_post` at a chosen layer and position is transplanted into the other,
and the change in the correct-vs-incorrect answer's logit difference is
measured. The layer with the largest mean effect
(`causal_bottleneck_layer`, `src/metrics.py`) is compared directly against
the logit-lens pivot layer (`compare_to_logit_lens_pivot`) — the central
comparison this repo is built to make.

### Controls

English-to-English identity, shuffled-prompt, shuffled-label probe, placebo
patching (unrelated layers, random directions, cross-prompt activations),
and cross-model replication — see `docs/CONTROLS.md` for the rationale
behind each.

## Results (preliminary, n=4 of 100 probes)

This repo's development environment is CPU-only (no CUDA build of torch, 4GB
laptop GPU — see `docs/COMPUTE.md`), and a single 26-layer logit-lens trace
on `gemma-2-2b` takes on the order of minutes there. Only 4 of the 100 probe
prompts had completed a full batch trace (`results/traces.jsonl`) at the time
of this writeup; the figures and full-dataset statistics described elsewhere
in this repo (`README.md` "Headline figure", `results/PROBE_RESULTS_STATUS.md`,
`results/PATCHING_RESULTS_STATUS.md`) are pending completion of that batch
run and are not yet backed by data. What follows is reported honestly as a
preliminary read on 4 prompts, not a dataset-level finding.

Running `detect_pivot(target_script="sinhala")` (a *stable* switch to Sinhala
that persists through the final layer) on these 4 traces found **zero**
stable pivots to Sinhala. Two distinct patterns appeared instead:

- 2 of 4 prompts: the readout stays `latin`-script for nearly the entire
  layer stack, decaying into unclassified ("other") tokens only in the last
  few layers — consistent with Outcome 1 (English-pivot) *without* a
  detectable return to Sinhala at the output, at least under this repo's
  strict pivot definition.
- 2 of 4 prompts: the readout starts `sinhala`-script in the earliest
  layers, but degrades into unclassified ("other") tokens for the rest of
  the network rather than settling into either `latin` or `sinhala` — a
  diffuse, non-coherent readout consistent with Outcome 2 in this repo's
  framing (`README.md`), and worth flagging since Outcome 2 was predicted
  as the most likely result for a low-resource, heavily fragmented script
  going in.

**This is not a dataset-level claim.** n=4 is far too small to distinguish a
real pattern from noise, and no controls (Phase 16/24) or activation
patching (the only causal method in this repo) have been run against these 4
traces. It is reported here only because the honesty rule governing this
repo (`MASTER_PROMPT.md`) requires reporting what was actually run, not
projecting what the full run would likely show.
