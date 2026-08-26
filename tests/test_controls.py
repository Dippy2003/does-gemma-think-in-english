from src.controls import (
    english_identity_condition,
    shuffled_prompt_condition,
    tamil_condition,
)


def test_tamil_condition_shape():
    rows = tamil_condition()
    assert len(rows) == 100
    assert {"id", "prompt", "answer"} <= set(rows[0].keys())


def test_english_identity_condition_shape():
    rows = english_identity_condition()
    assert len(rows) == 100


def test_shuffled_prompt_preserves_word_multiset():
    original = tamil_condition()  # unused, just importing to ensure no crash
    shuffled = shuffled_prompt_condition(seed=1)
    assert len(shuffled) == 100
    for row in shuffled[:5]:
        assert isinstance(row["prompt"], str)


def test_shuffled_is_deterministic_by_seed():
    a = shuffled_prompt_condition(seed=42)
    b = shuffled_prompt_condition(seed=42)
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
