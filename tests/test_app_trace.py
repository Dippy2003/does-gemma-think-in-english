from app.trace import trace_prompt


class FakeCache(dict):
    def __getitem__(self, key):
        return None


class FakeModel:
    class cfg:
        n_layers = 2

    def to_tokens(self, prompt):
        return [[1, 2, 3]]

    def run_with_cache(self, tokens):
        return None, {}


def test_trace_prompt_schema(monkeypatch):
    def fake_run_with_cache(model, prompt):
        import torch

        return torch.zeros(1, 3), torch.zeros(1, 3, 5), {}

    def fake_build_trace(model, prompt, cache):
        from src.logit_lens import LayerTrace, Trace

        return Trace(
            prompt=prompt,
            layers=[
                LayerTrace(layer=0, token="a", script="latin", prob=0.5),
                LayerTrace(layer=1, token="b", script="sinhala", prob=0.6),
            ],
        )

    monkeypatch.setattr("app.trace.run_with_cache", fake_run_with_cache)
    monkeypatch.setattr("app.trace.build_trace", fake_build_trace)

    result = trace_prompt(FakeModel(), "test prompt")
    assert result["prompt"] == "test prompt"
    assert len(result["layers"]) == 2
    assert "pivot_layer" in result
