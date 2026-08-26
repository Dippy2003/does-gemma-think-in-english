# Does a multilingual LLM pivot through English when prompted in Sinhala?

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
