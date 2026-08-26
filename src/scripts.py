def script_of(s: str) -> str:
    for ch in s:
        cp = ord(ch)
        if 0x0D80 <= cp <= 0x0DFF:
            return "sinhala"
        if 0x0B80 <= cp <= 0x0BFF:
            return "tamil"
        if 0x0900 <= cp <= 0x097F:
            return "devanagari"
        if 0x0600 <= cp <= 0x06FF:
            return "arabic"
        if 0x4E00 <= cp <= 0x9FFF:
            return "han"
        if 0x0041 <= cp <= 0x007A:
            return "latin"
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
