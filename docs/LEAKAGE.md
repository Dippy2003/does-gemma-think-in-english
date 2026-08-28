# Leakage risks in the linear probing pipeline

- **Prompt-level leakage.** Splitting by row instead of by prompt id would
  let, e.g., layer-3 and layer-14 activations of the *same* prompt land in
  different folds. A probe could then partly memorize prompt-specific
  activation quirks rather than learning a genuine language signal. Fixed by
  `train_test_split_by_prompt`.
- **Standardization leakage.** Fitting the mean/std used to normalize
  activations on the full dataset (train + test) leaks test-set distribution
  information into training. Fixed by `standardize`, which fits only on the
  train split.
- **Category leakage.** Because the probe set's categories (factual,
  spatial, temporal, kinship, numeric) use repeated sentence templates
  (e.g. every kinship row follows "My X's Y is my Z"), a probe could in
  principle learn to key off the template rather than the language. This is
  not fully addressed by the current split and is worth stratifying by
  category in a follow-up if probe accuracy looks implausibly high.
