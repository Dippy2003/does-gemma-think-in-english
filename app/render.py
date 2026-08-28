import html


def render_tokenization_strip(tokens: list) -> str:
    """One HTML span per token, so the user can see a word shatter into pieces.

    Escapes each token before embedding it: an unescaped Sinhala token that
    happens to contain `<`/`&` (rare, but tokenizer pieces are not guaranteed
    HTML-safe) would otherwise corrupt the surrounding markup or, worse, let
    a crafted prompt inject HTML into the page.
    """
    return " ".join(f"<span class='token'>{html.escape(t)}</span>" for t in tokens)


def render_layer_stack(layers: list) -> str:
    """One row per layer: layer number, decoded token, script-coloured."""
    rows = []
    for lt in layers:
        rows.append(
            f"<tr><td>{lt['layer']}</td>"
            f"<td class='script-{lt['script']}'>{html.escape(lt['token'])}</td>"
            f"<td>{lt['prob']:.2f}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"
