from src.io import patching_jsonl_to_parquet


def test_patching_jsonl_to_parquet(tmp_path):
    import json

    jsonl = tmp_path / "patching.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "results": [{"layer": 0, "effect": 0.1}, {"layer": 1, "effect": 0.8}]}),
                json.dumps({"id": "2", "results": [{"layer": 0, "effect": 0.0}]}),
            ]
        ),
        encoding="utf-8",
    )
    df = patching_jsonl_to_parquet(jsonl, tmp_path / "out.parquet")
    assert len(df) == 3
    assert set(df["id"]) == {"1", "2"}
