import hashlib
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_cache (
    prompt_hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_connection(path: str = "app/cache.sqlite3") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_or_compute(conn: sqlite3.Connection, prompt: str, compute_fn) -> dict:
    """Read-through cache: return the cached result_json if present, else
    call `compute_fn(prompt)`, store it, and return it. `compute_fn` should
    return a JSON-serializable dict."""
    import json

    key = hash_prompt(prompt)
    row = conn.execute(
        "SELECT result_json FROM trace_cache WHERE prompt_hash = ?", (key,)
    ).fetchone()
    if row is not None:
        return json.loads(row[0])

    result = compute_fn(prompt)
    conn.execute(
        "INSERT INTO trace_cache (prompt_hash, prompt, result_json) VALUES (?, ?, ?)",
        (key, prompt, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()
    return result
