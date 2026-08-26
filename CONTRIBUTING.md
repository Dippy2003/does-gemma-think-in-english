# Repo conventions

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`.
- Every commit leaves the repo in a working state.
- Cache activations to disk (parquet) rather than holding them all in RAM.
- Logit lens results are correlational. Only activation patching (`src/patching.py`)
  supports a causal claim. Do not phrase findings otherwise in docs, notebooks,
  or the app.
- Every experiment script must degrade to CPU rather than crash when no GPU is
  available.
