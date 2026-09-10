import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import importlib.util

_spec = importlib.util.spec_from_file_location("app_module", Path(__file__).resolve().parents[1] / "app" / "app.py")
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)

from src.serve.guardrails import DISCLAIMER_YO, GuardrailOutcome


def test_log_interaction_never_writes_raw_question_text(tmp_path, monkeypatch):
    log_path = tmp_path / "interactions.log"
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", log_path)

    question = "Kí ni àwọn àmì àrùn ibà? Orúkọ mi ni Tayo Habib, foonu mi ni 0801234567."
    app_module.log_interaction(question, GuardrailOutcome.ANSWERED, "some answer text", 0.42)

    content = log_path.read_text(encoding="utf-8")
    assert "Tayo Habib" not in content
    assert "0801234567" not in content
    assert "ibà" not in content  # no fragment of the raw question text at all

    record = json.loads(content.strip())
    assert set(record.keys()) == {"timestamp_utc", "question_sha256", "guardrail_outcome",
                                   "response_char_len", "latency_seconds", "served_from_cache"}
    assert record["served_from_cache"] is False
    assert record["guardrail_outcome"] == "answered"
    assert record["response_char_len"] == len("some answer text")
    assert len(record["question_sha256"]) == 64  # sha256 hex digest


def test_log_interaction_appends_multiple_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "interactions.log"
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", log_path)

    app_module.log_interaction("q1", GuardrailOutcome.OUT_OF_SCOPE, "r1", 0.1)
    app_module.log_interaction("q2", GuardrailOutcome.CLINICAL_BOUNDARY, "r2", 0.2)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["guardrail_outcome"] == "out_of_scope"
    assert json.loads(lines[1])["guardrail_outcome"] == "clinical_boundary"


class FakeService:
    def __init__(self, response_text="Ìdáhùn.", outcome=GuardrailOutcome.ANSWERED, raise_no_backend=False):
        self._response_text = response_text
        self._outcome = outcome
        self._raise = raise_no_backend

    def answer(self, question):
        if self._raise:
            raise RuntimeError("no model backend loaded")
        from src.serve.guardrails import GuardedResponse

        return GuardedResponse(self._outcome, self._response_text)


def test_answer_fn_handles_empty_question():
    answer_fn = app_module.make_answer_fn(FakeService())
    result = answer_fn("   ")
    assert "ìbéèrè" in result.lower() or "Jọ̀wọ́" in result


def test_answer_fn_returns_service_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    answer_fn = app_module.make_answer_fn(FakeService(response_text="Àmì àrùn ibà ni ibà gbígbóná."))
    result = answer_fn("Kí ni àwọn àmì àrùn ibà?")
    assert result == "Àmì àrùn ibà ni ibà gbígbóná."


def test_answer_fn_serves_cached_answer_without_calling_service(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    service = FakeService(response_text="should not be used")
    cached = {"Kí ni àwọn àmì àrùn ibà?": "Èyí ni ìdáhùn gidi tí a ti kó sílẹ̀ tẹ́lẹ̀."}
    answer_fn = app_module.make_answer_fn(service, cached_answers=cached)

    result = answer_fn("Kí ni àwọn àmì àrùn ibà?")

    assert "Èyí ni ìdáhùn gidi tí a ti kó sílẹ̀ tẹ́lẹ̀." in result
    assert DISCLAIMER_YO in result  # cached path still applies G3 via apply_disclaimer()


def test_answer_fn_cache_miss_falls_back_to_live_service(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    service = FakeService(response_text="live answer")
    cached = {"a different question entirely": "cached answer"}
    answer_fn = app_module.make_answer_fn(service, cached_answers=cached)

    result = answer_fn("Kí ni àwọn àmì àrùn ibà?")

    assert result == "live answer"


def test_log_interaction_records_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    service = FakeService()
    cached = {"Kí ni àwọn àmì àrùn ibà?": "cached text"}
    answer_fn = app_module.make_answer_fn(service, cached_answers=cached)

    answer_fn("Kí ni àwọn àmì àrùn ibà?")

    log_content = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    record = json.loads(log_content.strip())
    assert record["served_from_cache"] is True


def test_load_cached_answers_returns_empty_dict_when_file_missing(monkeypatch):
    monkeypatch.setattr(app_module, "CACHED_ANSWERS_PATH", Path("/does/not/exist.json"))
    assert app_module.load_cached_answers() == {}


def test_load_cached_answers_loads_real_file(tmp_path, monkeypatch):
    cache_file = tmp_path / "cached_answers.json"
    cache_file.write_text(json.dumps({"Q1?": "A1."}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(app_module, "CACHED_ANSWERS_PATH", cache_file)
    assert app_module.load_cached_answers() == {"Q1?": "A1."}


def test_answer_fn_handles_missing_backend_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    answer_fn = app_module.make_answer_fn(FakeService(raise_no_backend=True))
    result = answer_fn("Kí ni àwọn àmì àrùn ibà?")
    assert isinstance(result, str) and len(result) > 0  # does not crash the app


def test_answer_fn_missing_backend_still_carries_disclaimer(tmp_path, monkeypatch):
    # Regression test: the "no model configured" placeholder message must
    # still carry the G3 disclaimer (FR6: on every answer), not bypass it.
    monkeypatch.setattr(app_module, "INTERACTION_LOG_PATH", tmp_path / "log.jsonl")
    answer_fn = app_module.make_answer_fn(FakeService(raise_no_backend=True))
    result = answer_fn("Kí ni àwọn àmì àrùn ibà?")
    assert DISCLAIMER_YO in result
