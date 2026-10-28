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
