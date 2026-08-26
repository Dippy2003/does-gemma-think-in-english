from transformers import AutoTokenizer

from src.fertility import tokens_per_char, tokens_per_word


def test_english_fertility_near_one():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tpw = tokens_per_word("the cat sat on the mat", tok)
    assert 0.8 <= tpw <= 2.0


def test_empty_string():
    tok = AutoTokenizer.from_pretrained("gpt2")
    assert tokens_per_word("", tok) == 0.0
    assert tokens_per_char("", tok) == 0.0
