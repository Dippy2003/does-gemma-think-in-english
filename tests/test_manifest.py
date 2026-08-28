from src.manifest import read_manifest, write_manifest

REQUIRED_KEYS = {"artifact", "model", "git_commit", "n_rows"}


def test_write_and_read_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    entries = [
        {"artifact": "fertility_gemma-2", "model": "gemma-2-2b", "git_commit": "abc123", "n_rows": 30}
    ]
    write_manifest(entries, path=str(path))
    loaded = read_manifest(path=str(path))
    assert loaded == entries


def test_manifest_entries_have_required_keys(tmp_path):
    path = tmp_path / "manifest.json"
    entries = [
        {"artifact": "traces", "model": "gemma-2-2b", "git_commit": "abc123", "n_rows": 8},
    ]
    write_manifest(entries, path=str(path))
    loaded = read_manifest(path=str(path))
    for entry in loaded:
        assert REQUIRED_KEYS <= set(entry.keys())


def test_read_missing_manifest_returns_empty(tmp_path):
    assert read_manifest(path=str(tmp_path / "nope.json")) == []
