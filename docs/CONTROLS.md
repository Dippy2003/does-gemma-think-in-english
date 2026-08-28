# Control design rationale

Per `MASTER_PROMPT.md` section 2.5, a pivot-detection result that survives
none of these controls is noise — and reporting that honestly is a stronger
outcome than claiming a clean effect.

- **English -> English identity control** (`english_identity_condition`).
  Sanity check: if the pipeline can't detect the standard English pivot on
  English prompts, the bug is in the pipeline, not evidence about Sinhala.
- **Shuffled-prompt control** (`shuffled_prompt_condition`). Same tokens,
  same script, same fertility, destroyed syntax. Isolates whether a detected
  pivot depends on the prompt being a coherent sentence, or just on which
  characters are present.
- **Shuffled-label probe control** (Phase 20). If a linear probe trained on
  activations with permuted labels performs above chance, the probe is
  picking up something other than the language signal it claims to.
- **Placebo patching** (Phase 24). Patch at layers/positions/directions with
  no reason to matter; a nonzero effect there means the patching methodology
  itself is broken, not that those layers are causally relevant.
- **Cross-model replication** (Qwen2.5, Phase 16). A pattern that appears in
  Gemma-2 only and vanishes in a model with a different pretraining mix and
  tokenizer is a Gemma-2 idiosyncrasy, not a claim about multilingual LLMs in
  general.
