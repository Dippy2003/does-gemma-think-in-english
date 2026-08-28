from src.scripts import script_of, script_ratios


def test_sinhala():
    assert script_of("සිංහල") == "sinhala"


def test_tamil():
    assert script_of("தமிழ்") == "tamil"


def test_latin():
    assert script_of("hello") == "latin"


def test_devanagari():
    assert script_of("हिन्दी") == "devanagari"


def test_arabic():
    assert script_of("العربية") == "arabic"


def test_han():
    assert script_of("中文") == "han"


def test_unknown():
    assert script_of("☃") == "other"


def test_empty_string():
    assert script_of("") == "other"
    assert script_ratios("") == {}


def test_punctuation_only():
    assert script_of("!?.,") == "other"


def test_digits_only():
    assert script_of("12345") == "other"


def test_mixed_script_ratio():
    ratios = script_ratios("aසb")
    assert ratios["latin"] == 2 / 3
    assert ratios["sinhala"] == 1 / 3
