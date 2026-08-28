# Negative results are results

Per `MASTER_PROMPT.md`: a result that survives none of the controls in
`docs/CONTROLS.md` is noise, and reporting that honestly is a stronger
outcome than claiming a clean effect. Concretely, in this repo's framing:

- If placebo patching (unrelated layers, random directions, cross-prompt
  activations) produces effects comparable to the real patch, the patching
  pipeline itself needs debugging before any causal claim can be trusted —
  publish that as a methods finding, not a null result to bury.
- If Sinhala shows **no coherent logit-lens pivot at all** (Outcome 2 in
  `README.md`'s three possible outcomes), that is not a failed experiment.
  It directly predicts the reliability problems Sinhala-facing systems
  actually exhibit in practice, and is arguably the most useful finding this
  repo could produce.
- If the causal bottleneck (patching) and the correlational pivot (logit
  lens) disagree, report the disagreement, not just whichever one looks
  cleaner.

The project's credibility rests on reporting whichever of these actually
happened, not on finding the tidiest story.
