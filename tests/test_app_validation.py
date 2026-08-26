from app.app import MAX_INPUT_CHARS, run_trace


def test_empty_input_shows_error():
    callout, strip, stack = run_trace("")
    assert "non-empty" in callout


def test_whitespace_only_input_shows_error():
    callout, strip, stack = run_trace("   ")
    assert "non-empty" in callout


def test_too_long_input_shows_error():
    callout, strip, stack = run_trace("a" * (MAX_INPUT_CHARS + 1))
    assert "too long" in callout.lower()
