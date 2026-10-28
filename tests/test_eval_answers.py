"""scripts/eval_answers.py: pin parsing, served-model census, honest judge agreement."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_answers", ROOT / "scripts" / "eval_answers.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


def test_pin_parses_provider_and_model():
    assert ea._pin("gemini:gemini-3.7-flash") == ("gemini", "gemini-3.7-flash")
    assert ea._pin("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert ea._pin("") is None


def test_second_judge_receives_a_real_pin_tuple(monkeypatch):
    """Regression: the second judge's pin used to be passed as the raw string,
    so llm.chat saw provider 'g' / model 'e' and silently ran unpinned."""
    seen = {}

    def fake_run_eval(samples, judge_pin=None, metrics=None, show_progress=False,
                      strict_pin=False, **kw):
        seen.update(pin=judge_pin, strict=strict_pin)
        rows = [{"faithfulness": 0.5, "judge_served": {"gemini/gemini-3.7-flash": 2}}
                for _ in samples]
        return {"rows": rows, "scores": {}, "served": {}}

    monkeypatch.setattr(ea.judge, "run_eval", fake_run_eval)
    out = ea.stage_agreement([{}], [1.0], [{"groq/openai/gpt-oss-120b": 2}],
                             ea._pin("gemini:gemini-3.7-flash"), strict_pin=True)
    assert seen == {"pin": ("gemini", "gemini-3.7-flash"), "strict": True}
    assert out["n"] == 1 and out["mean_abs_delta"] == 0.5     # different models: computed
    assert out["judge2"] == "gemini:gemini-3.7-flash"


def test_agreement_refuses_to_compare_a_model_with_itself(monkeypatch):
    lite = {"gemini/gemini-3.5-flash-lite": 4}

    def fake_run_eval(samples, **kw):
        return {"rows": [{"faithfulness": 1.0, "judge_served": lite} for _ in samples],
                "scores": {}, "served": {}}

    monkeypatch.setattr(ea.judge, "run_eval", fake_run_eval)
    out = ea.stage_agreement([{}, {}], [1.0, 0.8], [lite, lite], ("gemini", "gemini-3.7-flash"))
    assert out["n"] == 0 and out["pearson"] is None
    assert out["note"].startswith("not computed: same model served both judges")


def test_census_sums_per_record_dicts():
    assert ea.census([{"a/x": 2}, {"a/x": 1, "b/y": 5}, {}]) == {"b/y": 5, "a/x": 3}
