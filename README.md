# Does a multilingual LLM pivot through English when prompted in Sinhala?

An interpretability study of `google/gemma-2-2b` asking whether the model's
internal residual stream passes through an English-like representation when
processing Sinhala prompts, in the spirit of Wendler et al. (2024), *Do Llamas
Work in English? On the Latent Language of Multilingual Transformers*. Sinhala
is low-resource, Brahmic-script, and heavily under-tokenized — a condition no
prior latent-language study has covered.
