from src.scripts import script_of


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
