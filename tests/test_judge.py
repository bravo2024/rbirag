"""ragas judge harness: compat shim, metric registry, chat wrapping."""
import pytest
from langchain_core.prompt_values import StringPromptValue

from regintel import judge


def test_compat_shim_is_idempotent():
    assert judge.install_ragas_compat() == {}      # already installed at import
    import ragas  # noqa: F401  (importable with the shim in place)
    assert ragas.__version__


def test_metric_registry():
    assert set(judge.METRIC_CLASSES) == {
        "faithfulness", "answer_relevancy", "llm_context_precision_without_reference"}
    assert judge.build_metrics(["faithfulness"])
    with pytest.raises(ValueError):
        judge.build_metrics(["nope"])


def test_judge_llm_routes_through_chat(monkeypatch):
    seen = {}

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        seen.update(messages=messages, pin=pin, temperature=temperature,
                    max_tokens=max_tokens)
        return {"text": "ok", "model": "m", "provider": "p", "cached": False, "attempts": []}

    monkeypatch.setattr(judge.llm_py, "chat", fake_chat)
    wrapper = judge.judge_llm(pin=("gemini", "gemini-3.7-flash"))
    res = wrapper.generate_text(StringPromptValue(text="hello?"), n=2)

    assert seen["pin"] == ("gemini", "gemini-3.7-flash")
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 6000
    assert seen["messages"][0]["content"] == "hello?"
    # one prompt, n generations — the LLMResult shape ragas reads
    assert len(res.generations) == 1
    assert [g.text for g in res.generations[0]] == ["ok", "ok"]
    assert wrapper.is_finished(res)


def test_num_marshalling():
    assert judge._num("x") is None
    assert judge._num(float("nan")) is None
    assert judge._num(0.5) == 0.5


def test_judge_llm_records_served_models_and_strict_flag(monkeypatch):
    seen = {}

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        seen["strict_pin"] = strict_pin
        return {"text": "ok", "model": "m", "provider": "p", "cached": True, "attempts": []}

    monkeypatch.setattr(judge.llm_py, "chat", fake_chat)
    wrapper = judge.judge_llm(pin=("gemini", "g"), strict_pin=True)
    wrapper.generate_text(StringPromptValue(text="q"), n=2)
    assert seen["strict_pin"] is True
    assert wrapper.served == [("p", "m", True), ("p", "m", True)]
