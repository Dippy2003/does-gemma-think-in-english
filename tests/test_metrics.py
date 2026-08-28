import pandas as pd

from src.metrics import fertility_comparison_table, per_language_distribution


def test_fertility_comparison_table():
    df = pd.DataFrame(
        {
            "language": ["en", "si"],
            "tokens_per_word": [1.1, 5.5],
            "tokens_per_char": [0.2, 1.0],
        }
    )
    table = fertility_comparison_table({"gemma-2": df})
    assert len(table) == 2
    assert set(table["language"]) == {"en", "si"}


def test_per_language_distribution():
    df = pd.DataFrame(
        {
            "language": ["en", "en", "si"],
            "tokens_per_word": [1.0, 2.0, 5.0],
        }
    )
    dist = per_language_distribution(df)
    assert dist.loc["en", "mean"] == 1.5
