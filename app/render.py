def render_tokenization_strip(tokens: list) -> str:
    """One HTML span per token, so the user can see a word shatter into pieces."""
    return " ".join(f"<span class='token'>{t}</span>" for t in tokens)


def render_layer_stack(layers: list) -> str:
    """One row per layer: layer number, decoded token, script-coloured."""
    rows = []
    for lt in layers:
        rows.append(
            f"<tr><td>{lt['layer']}</td>"
            f"<td class='script-{lt['script']}'>{lt['token']}</td>"
            f"<td>{lt['prob']:.2f}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"
