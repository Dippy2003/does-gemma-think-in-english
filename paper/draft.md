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
