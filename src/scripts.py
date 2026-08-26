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
