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

## Tokenizer fertility (control, not a finding)

Mean tokens per word, per tokenizer and language, on the parallel corpora in
`data/`:

| tokenizer | en | si | ta |
|---|---|---|---|
| gemma-2 | 1.15 | 5.57 | 3.92 |
| gpt-oss (o200k_harmony) | 1.18 | 3.56 | 3.15 |
| qwen2.5 | 1.17 | 9.53 | 9.55 |

Sinhala costs 3.6x–9.5x as many tokens per word as English, depending on the
tokenizer. See `figures/fertility_bar_chart.png` and
`figures/fertility_distribution_violin.png`. `meta-llama/Llama-3.2-1B`'s
tokenizer could not be measured — see `results/fertility_llama-3.2_STATUS.md`.

## Headline figure

`figures/pivot_layer_language.png` shows, layer by layer, what fraction of
Sinhala-prompt readouts decode to each Unicode script — the logit-lens
analogue of Wendler et al.'s English-pivot curve, generated from
`results/traces.parquet` (Phase 17+). This is a correlational readout, not a
causal claim; see the note above and `src/logit_lens.py`.

**This is a confound, not a finding.** Higher fertility means the model has
more sequence positions to process the same content — more layers of residual
stream in which a pivot could, mechanically, have more room to happen or not
happen. Any difference between Sinhala's and English's logit-lens pivot depth
observed later in this project must be read against this number, not treated
as pure evidence about the model's internal representation of language.
