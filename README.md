# Does a multilingual LLM pivot through English when prompted in Sinhala?

An interpretability study of `google/gemma-2-2b` asking whether the model's
internal residual stream passes through an English-like representation when
processing Sinhala prompts, in the spirit of Wendler et al. (2024), *Do Llamas
Work in English? On the Latent Language of Multilingual Transformers*. Sinhala
is low-resource, Brahmic-script, and heavily under-tokenized — a condition no
prior latent-language study has covered.

## Research question

Wendler et al. (2024) showed that decoding the residual stream of Llama-family
models at middle layers, via the logit lens, produces English-like tokens even
when the prompt and target output are French, German or Chinese — evidence
that the model's internal concept space is biased toward English and that
target-language identity is applied late. Every language tested so far is
high-resource and mostly Latin-script. This project asks the same question of
Sinhala: does the same pivot appear, does it appear later and weaker, does it
fail to cohere at all, or does the model pivot through a third, script- or
lexically-adjacent language such as Hindi? Logit lens results are correlational
only; the causal claims in this repo rest on activation patching.
