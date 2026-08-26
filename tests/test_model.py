from src.model import get_device, select_dtype


def test_get_device_returns_known_value():
    assert get_device() in ("cuda", "cpu")


def test_select_dtype_cpu_is_float32():
    assert select_dtype("cpu") == "float32"


def test_select_dtype_low_vram_is_float16():
    assert select_dtype("cuda", vram_gb=8) == "float16"


def test_select_dtype_high_vram_is_bfloat16():
    assert select_dtype("cuda", vram_gb=24) == "bfloat16"
