def render_tokenization_strip(tokens: list) -> str:
    """One HTML span per token, so the user can see a word shatter into pieces."""
    return " ".join(f"<span class='token'>{t}</span>" for t in tokens)
