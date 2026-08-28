# Probe methodology

For each layer's last-token `resid_post` activation, a logistic regression
probe (`class_weight="balanced"`) is trained to predict prompt language from
a set of Sinhala/Tamil/English probe pairs. Splits are by prompt id
(`train_test_split_by_prompt`), never by row, and standardization statistics
are fit on train only (`standardize`) — see `docs/LEAKAGE.md`.

Reported per layer: test accuracy, 5-fold cross-validated accuracy with
bootstrap confidence intervals, and significance against two baselines: the
majority-class rate (`random_baseline_accuracy`) and a shuffled-label control
(`shuffled_label_probe_control`) that must land near chance by construction.

A high-accuracy layer means the language signal is linearly present in that
layer's residual stream — not that the model relies on it. See
`src/probes.py`'s module docstring for the full distinction from a causal
claim.
