from src.controls import placebo_layers
from src.metrics import placebo_summary


def test_placebo_layers_excludes_real_layers():
    real = [10, 14, 18]
    picked = placebo_layers(n_layers=26, real_layers=real, n=5, seed=0)
    assert set(picked).isdisjoint(real)
    assert len(picked) == 5


def test_placebo_summary_flags_near_zero_effects():
    result = placebo_summary(placebo_effects=[0.01, -0.02, 0.03], real_effect=0.8)
    assert result["real_exceeds_placebo"] is True
    assert abs(result["placebo_mean"]) < 0.1
