from transformers import AutoTokenizer

from src.io import load_probes, single_token_report


def test_single_token_report_shape():
    tok = AutoTokenizer.from_pretrained("gpt2")
    df = load_probes()
    report = single_token_report(df, tok)
    assert len(report) == len(df)
    assert {"answer_si_n_tokens", "answer_ta_n_tokens", "answer_en_n_tokens"} <= set(
        report.columns
    )
    assert (report["answer_en_n_tokens"] >= 1).all()
