from app.cache import get_connection, get_or_compute, hash_prompt


def test_hash_prompt_deterministic():
    assert hash_prompt("hello") == hash_prompt("hello")
    assert hash_prompt("hello") != hash_prompt("world")


def test_get_or_compute_cache_miss_then_hit(tmp_path):
    conn = get_connection(str(tmp_path / "test.sqlite3"))
    calls = []

    def compute(prompt):
        calls.append(prompt)
        return {"prompt": prompt, "value": 42}

    r1 = get_or_compute(conn, "hi", compute)
    r2 = get_or_compute(conn, "hi", compute)
    assert r1 == r2 == {"prompt": "hi", "value": 42}
    assert len(calls) == 1  # second call was a cache hit
