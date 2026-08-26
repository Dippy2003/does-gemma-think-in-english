def script_of(s: str) -> str:
    for ch in s:
        cp = ord(ch)
        if 0x0D80 <= cp <= 0x0DFF:
            return "sinhala"
        if 0x0B80 <= cp <= 0x0BFF:
            return "tamil"
        if 0x0041 <= cp <= 0x007A:
            return "latin"
    return "other"
