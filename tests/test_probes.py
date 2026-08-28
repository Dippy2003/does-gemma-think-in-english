from src.io import CATEGORIES, PROBE_COLUMNS, load_probes


def test_columns_present():
    df = load_probes()
    assert list(df.columns) == PROBE_COLUMNS


def test_starts_unverified():
    df = load_probes()
    assert (~df["verified"]).all()


def test_categories_known():
    df = load_probes()
    assert set(df["category"]).issubset(set(CATEGORIES))
