"""Control conditions for the logit-lens / pivot experiment.

Per MASTER_PROMPT.md section 2.5, a result that survives none of these
controls is noise, and reporting that is a stronger outcome than claiming a
clean effect.
"""

import random

from src.io import load_probes


def tamil_condition() -> "list[dict]":
    """Same probes, Tamil prompt column — tests whether a pivot pattern is
    Sinhala-specific or shared by another Brahmic-adjacent low-resource
    language."""
    df = load_probes()
    return df[["id", "tamil", "answer_ta"]].rename(
        columns={"tamil": "prompt", "answer_ta": "answer"}
    ).to_dict("records")


def shuffled_prompt_condition(seed: int = 0) -> "list[dict]":
    """Sinhala prompts with their word order shuffled — destroys syntax while
    keeping token-level script/fertility identical, isolating "is this
    language" from "is this a coherent sentence in this language."""
    df = load_probes()
    rng = random.Random(seed)
    rows = []
    for _, row in df.iterrows():
        words = row["sinhala"].split()
        rng.shuffle(words)
        rows.append({"id": row["id"], "prompt": " ".join(words), "answer": row["answer_si"]})
    return rows


def shuffled_label_probe_control(y: list, seed: int = 0) -> list:
    """Permute labels independent of activations. A probe that beats chance
    on this can only be overfitting or leaking — there is no real signal for
    it to have found, by construction."""
    rng = random.Random(seed)
    shuffled = list(y)
    rng.shuffle(shuffled)
    return shuffled


def english_identity_condition() -> "list[dict]":
    """English prompt -> English answer. If the model doesn't pivot on its own
    native language, the whole pivot-detection pipeline is broken, not the
    model."""
    df = load_probes()
    return df[["id", "english", "answer_en"]].rename(
        columns={"english": "prompt", "answer_en": "answer"}
    ).to_dict("records")
