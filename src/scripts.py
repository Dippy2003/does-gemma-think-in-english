SCRIPT_RANGES = (
    (0x0D80, 0x0DFF, "sinhala"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0900, 0x097F, "devanagari"),
    (0x0600, 0x06FF, "arabic"),
    (0x4E00, 0x9FFF, "han"),
    (0x0041, 0x007A, "latin"),
)


def script_of(s: str) -> str:
    for ch in s:
        cp = ord(ch)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= cp <= hi:
                return name
    return "other"


def script_ratios(s: str) -> dict:
    """Fraction of characters in `s` belonging to each detected script."""
    if not s:
        return {}
    counts: dict = {}
    for ch in s:
        counts[script_of(ch)] = counts.get(script_of(ch), 0) + 1
    n = len(s)
    return {k: v / n for k, v in counts.items()}
