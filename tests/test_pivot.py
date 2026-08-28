from src.logit_lens import LayerTrace, Trace
from src.pivot import detect_pivot, english_likeness_score, pivot_confidence


def make_trace(scripts, probs=None):
    probs = probs or [0.5] * len(scripts)
    return Trace(
        prompt="test",
        layers=[
            LayerTrace(layer=i, token="x", script=s, prob=p)
            for i, (s, p) in enumerate(zip(scripts, probs))
        ],
    )


def test_detect_pivot_finds_stable_switch():
    trace = make_trace(["latin", "latin", "sinhala", "sinhala", "sinhala"])
    assert detect_pivot(trace, target_script="sinhala") == 2


def test_detect_pivot_none_if_unstable():
    trace = make_trace(["latin", "sinhala", "latin", "sinhala", "latin"])
    assert detect_pivot(trace, target_script="sinhala") is None


def test_pivot_confidence():
    trace = make_trace(["latin", "sinhala", "sinhala"], probs=[0.1, 0.8, 0.6])
    assert pivot_confidence(trace, pivot_layer=1) == 0.7


def test_english_likeness_score():
    trace = make_trace(["latin", "sinhala", "latin"])
    assert english_likeness_score(trace) == [1.0, 0.0, 1.0]
