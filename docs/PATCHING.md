# Where the answer is actually decided

`src/patching.py`'s `layer_sweep` transplants each layer's `resid_post` from
a Sinhala-prompt ("clean") run into its English-twin-prompt ("corrupt") run,
and measures how much of the clean run's logit-diff gets restored
(`PatchResult.effect`, normalized 0-1). The layer with the largest mean
effect (`causal_bottleneck_layer`) is the causal answer to "where does the
model's decision live" — a claim logit lens and linear probing cannot make.

**The key comparison in this repo** is `compare_to_logit_lens_pivot`: does
the causal bottleneck layer line up with the logit-lens pivot layer? Three
outcomes are all informative:

- **Aligned** — the readout and the causal mechanism agree; the logit lens
  was, in this case, a reasonable proxy for what the model relies on.
- **Causal bottleneck earlier than the readout pivot** — the decision is
  made before the readout looks like it's decided; the model "knows" before
  it "shows."
- **Causal bottleneck later, or not localized** — the readout pivot is
  cosmetic; whatever the logit lens shows mid-network is not what the model
  is actually using to answer.

None of these outcomes are a failure of the method. Reporting whichever one
actually occurs, honestly, is the finding.
