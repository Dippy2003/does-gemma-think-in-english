# notebooks/

`02_logit_lens.ipynb` runs a single-prompt trace through `src/logit_lens.py`.
Read the caveat in that module's docstring before drawing conclusions from
its output: an argmax token at layer N is a readout of what the unembedding
matrix would produce there, not evidence about what the model is internally
"doing" at layer N. Sinhala examples in this notebook are drawn from
`data/parallel_probes.csv`; multi-token prompts are truncated to the last
position for decoding.
