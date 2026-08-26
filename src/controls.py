"""Control conditions for the logit-lens / pivot experiment.

Per MASTER_PROMPT.md section 2.5, a result that survives none of these
controls is noise, and reporting that is a stronger outcome than claiming a
clean effect.
"""

from src.io import load_probes


def tamil_condition() -> "list[dict]":
    """Same probes, Tamil prompt column — tests whether a pivot pattern is
    Sinhala-specific or shared by another Brahmic-adjacent low-resource
    language."""
    df = load_probes()
    return df[["id", "tamil", "answer_ta"]].rename(
        columns={"tamil": "prompt", "answer_ta": "answer"}
    ).to_dict("records")
