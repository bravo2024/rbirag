"""Conversational path: greetings answer directly instead of being refused."""
import pytest

from regintel import answer, llm


@pytest.mark.parametrize("q", [
    "hi", "Hi!", "hello", "HELLO.", "hey there", "namaste", "good morning",
    "thanks", "Thank you so much", "bye", "see you", "ok", "got it", "cool",
    "who are you?", "what can you do?", "what is this app", "how does this work?",
    "help", "can you help me", "are you there?", "how are you?",
])
def test_smalltalk_detected(q):
    assert answer.is_smalltalk(q)


@pytest.mark.parametrize("q", [
    "minimum capital adequacy ratio requirement for commercial banks",
    "what is the KYC update frequency for NBFCs?",
    "how are you calculating CRAR?",
    "hi, what is the CRAR requirement for NBFCs?",
    "help me find the KYC norms",
    "which circular covers provisioning for NPAs?",
    "what can you do about a bank that misses the LCR threshold?",
    "",
])
def test_real_questions_are_not_smalltalk(q):
    assert not answer.is_smalltalk(q)


def test_chat_reply_uses_llm(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    seen = {}

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        seen["messages"] = messages
        seen["max_tokens"] = max_tokens
        return {"text": "Hello! Ask me about RBI rules.", "model": "m", "provider": "p",
                "cached": False, "attempts": []}

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = answer.chat_reply("hi", history=[{"role": "user", "content": "hello"},
                                           {"role": "assistant", "content": "Hi!"}])
    assert out["mode"] == "chat" and out["abstain"] is False
    assert out["answer_md"] == "Hello! Ask me about RBI rules."
    assert out["provider"] == "p" and out["model"] == "m"
    assert out["citations"] == [] and out["claims"] == []
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][-1] == {"role": "user", "content": "hi"}
    assert seen["max_tokens"] == 400


def test_chat_reply_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("all providers failed")

    monkeypatch.setattr(llm, "chat", boom)
    out = answer.chat_reply("hello")
    assert out["mode"] == "chat" and out["abstain"] is False
    assert out["answer_md"] == answer.CHAT_FALLBACK
    assert out["provider"] is None
    assert out["errors"] and "all providers failed" in out["errors"][0]


def test_chat_reply_skips_llm_when_disabled(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("LLM must not be called"))
    out = answer.chat_reply("hi", use_llm=False)
    assert out["mode"] == "chat" and out["answer_md"] == answer.CHAT_FALLBACK


def test_chat_reply_treats_empty_completion_as_failure(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: {
        "text": "   ", "model": "m", "provider": "p", "cached": False, "attempts": []})
    out = answer.chat_reply("hi")
    assert out["answer_md"] == answer.CHAT_FALLBACK
    assert out["errors"]


# ---------------------------------------------------------------------------
# Multi-turn: a follow-up keeps its topic for retrieval, and earlier turns
# reach the prompt as context only.
# ---------------------------------------------------------------------------
CRAR_TURNS = [
    {"role": "user", "content": "What is the minimum CRAR for a commercial bank?"},
    {"role": "assistant", "content": "A bank must keep Pillar 1 CRAR of 9 per cent [C1]."},
]


def test_contextual_query_leaves_standalone_questions_alone():
    assert answer.contextual_query("What is CRAR?") == "What is CRAR?"
    assert answer.contextual_query("What is CRAR?", []) == "What is CRAR?"
    assert answer.contextual_query("What is CRAR?", CRAR_TURNS) == "What is CRAR?"
    assert (answer.contextual_query("How often must KYC records be updated for high-risk customers?",
                                    CRAR_TURNS)
            == "How often must KYC records be updated for high-risk customers?")


@pytest.mark.parametrize("follow_up", [
    "What about NBFCs?",
    "and for co-operative banks?",
    "How is it computed?",
    "Does that apply to small finance banks?",
])
def test_contextual_query_prepends_the_previous_question(follow_up):
    assert (answer.contextual_query(follow_up, CRAR_TURNS)
            == f"What is the minimum CRAR for a commercial bank? {follow_up}")


def test_contextual_query_uses_the_most_recent_user_turn():
    history = CRAR_TURNS + [
        {"role": "user", "content": "What is the LCR?"},
        {"role": "assistant", "content": "The liquidity coverage ratio [C1]."},
    ]
    assert answer.contextual_query("How is it computed?", history) == "What is the LCR? How is it computed?"


def test_contextual_query_ignores_blank_and_assistant_only_history():
    blank = [{"role": "user", "content": "   "}, {"role": "assistant", "content": "hello"}]
    assert answer.contextual_query("What about NBFCs?", blank) == "What about NBFCs?"
    assert answer.contextual_query("What about NBFCs?", [{"role": "assistant", "content": "hi"}]) \
        == "What about NBFCs?"


def _hit(text: str = "KYC is updated every two years."):
    from regintel.index.fusion import Hit
    return Hit(chunk_id="c1", doc_id="d1", text=text, title="T", reference="R",
               issue_date="2025-01-01", doc_type="master_direction", status="current",
               entities=["nbfc"], entity_scope="specific", themes=[], section="", para="",
               page_start=1, page_end=1, page_url=None, pdf_url=None,
               scores={"rerank": 5.0})


def _capture_prompt(monkeypatch):
    from regintel import llm as llm_mod
    seen = {}

    def fake_chat(messages, temperature=0.2, max_tokens=1400, providers=None, pin=None,
                  strict_pin=False):
        seen["user"] = messages[1]["content"]
        return {"text": '{"abstain": false, "answer": "A grounded answer.", "claims": []}',
                "model": "m", "provider": "p", "cached": False, "attempts": []}

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    return seen


def test_answer_prompt_carries_history_as_context_only(monkeypatch):
    seen = _capture_prompt(monkeypatch)
    out = answer.answer("What about NBFCs?", [_hit()], llm_pin=("p", "m"), history=CRAR_TURNS)
    prompt = seen["user"]
    assert out["mode"] == "llm"
    assert "CONVERSATION SO FAR" in prompt
    assert "User: What is the minimum CRAR for a commercial bank?" in prompt
    assert "Assistant: A bank must keep Pillar 1 CRAR of 9 per cent [C1]." in prompt
    assert "never cite it" in prompt
    assert "Resolve references" in prompt
    assert prompt.index("CONVERSATION SO FAR") < prompt.index("QUESTION: What about NBFCs?")
    assert prompt.index("QUESTION: What about NBFCs?") < prompt.index("CONTEXT:")


def test_answer_prompt_without_history_has_no_conversation_block(monkeypatch):
    seen = _capture_prompt(monkeypatch)
    answer.answer("What is CRAR?", [_hit()], llm_pin=("p", "m"))
    assert "CONVERSATION SO FAR" not in seen["user"]
    assert "Resolve references" not in seen["user"]


def test_answer_prompt_truncates_long_history_turns(monkeypatch):
    seen = _capture_prompt(monkeypatch)
    history = [{"role": "user", "content": "x" * 900}]
    answer.answer("What about NBFCs?", [_hit()], llm_pin=("p", "m"), history=history)
    assert "x" * 400 in seen["user"]
    assert "x" * 401 not in seen["user"]
