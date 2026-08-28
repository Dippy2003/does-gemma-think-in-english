import json

from src.io import read_done_ids, traces_jsonl_to_parquet


def test_read_done_ids_missing_file(tmp_path):
    assert read_done_ids(tmp_path / "nope.jsonl") == set()


def test_read_done_ids_and_flatten(tmp_path):
    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "1",
                        "prompt": "x",
                        "layers": [
                            {"layer": 0, "token": "a", "script": "latin", "prob": 0.1},
                            {"layer": 1, "token": "b", "script": "latin", "prob": 0.2},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "2",
                        "prompt": "y",
                        "layers": [{"layer": 0, "token": "c", "script": "latin", "prob": 0.3}],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert read_done_ids(jsonl) == {"1", "2"}

    df = traces_jsonl_to_parquet(jsonl, tmp_path / "out.parquet")
    assert len(df) == 3
    assert set(df["id"]) == {"1", "2"}
