"""Results manifest: which artifact was produced by which model, when, and how."""

import json
from pathlib import Path


def write_manifest(entries: list[dict], path: str = "results/manifest.json") -> None:
    """Each entry should record at minimum: artifact, model, git_commit, n_rows."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(entries, indent=2), encoding="utf-8")


def read_manifest(path: str = "results/manifest.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))
