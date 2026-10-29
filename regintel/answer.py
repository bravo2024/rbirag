"""Answer synthesis with claim-level citations and a hard validator.

    query -> retrieved passages -> LLM (strict JSON) -> citation validator -> result

The model must answer ONLY from the numbered passages and attach the passage
number(s) to every claim. Claims whose citations do not resolve are dropped and
counted; if nothing survives, the result is flagged unverified and the UI falls
back to showing the passages. Without any LLM provider the same structure is
filled extractively, so the product never depends on a free API being up.
"""
from __future__ import annotations

import json
import re
import time

from regintel import llm
from regintel.classify import ENTITY_LABELS
from regintel.index.fusion import Hit

SYSTEM_PROMPT = """You answer questions about Reserve Bank of India (RBI) regulations for the compliance \
and audit staff of a commercial bank and an NBFC in India. Use ONLY the numbered passages in the context.

Rules:
1. Cite the passage number(s) for every factual claim, like [C2] or [C1][C4].
2. Never state a number, date, threshold, form name or obligation that is not in the passages.
3. If the passages do not answer the question, set "abstain": true and say briefly what is missing.
4. If a passage is withdrawn, superseded, or not yet in force (marked forthcoming with an effective date), say so when you rely on it.
5. Be operational: name the instrument, the paragraph and the page when the context shows them; when a passage applies only to specific entity types, name them too.
6. Answer in 2-6 sentences (or a short list when the source is a list).
7. Every passage header says which institution types it applies to ("applies to: ..."). State which institution type each cited rule applies to, and never merge rules from passages with different "applies to" scopes into one statement as if they were a single rule: when the versions differ, give them separately.

Reply with strict JSON only:
{"abstain": false, "answer": "...", "claims": [{"text": "...", "citations": ["C1", "C3"]}]}"""

# Small talk never reaches retrieval: a greeting scores below the abstain
# cutoff, so the grounded path would refuse it. This prompt answers it
# directly, with no citations, and steers real questions back to the corpus.
CHAT_PROMPT = """You are RBIRAG, a friendly assistant for Reserve Bank of India (RBI) regulation. \
You answer from a corpus of RBI Master Directions, circulars and notifications for compliance \
and audit teams at commercial banks and NBFCs, and every grounded answer cites the instrument \
and paragraph.

For greetings, thanks, goodbyes and questions about yourself, reply warmly and briefly \
(under 4 sentences). If asked what you can do, mention that you can:
- answer questions on capital adequacy, KYC/AML, provisioning and NPAs, exposure and \
liquidity norms, returns and forms, and recent circulars;
- point to the exact instrument and paragraph behind each answer, with a link to the RBI source;
- filter by institution type (commercial bank, NBFC, or all) and by date.

Rules:
1. Never invent a regulatory requirement, threshold, date or form name. You have no corpus in \
front of you for this turn, so do not state regulatory facts.
2. If the user asks an actual regulatory question, invite them to ask it directly so it can be \
answered from the corpus with citations.
3. No citations are needed here. Plain prose, no JSON."""

CHAT_FALLBACK = (
    "Hi \u2014 I'm RBIRAG, a regulatory assistant for RBI Master Directions and "
    "notifications. Ask me about capital adequacy, KYC, provisioning, exposure or "
    "liquidity norms, or a specific circular, and I'll answer from the corpus with "
    "citations to the instrument and paragraph.")

# Exact normalised messages that are conversation, not questions. Matching the
# whole message (not a substring) keeps real questions such as "how are you
# calculating CRAR?" on the grounded path.
_SMALLTALK = {
    "hi", "hii", "hiya", "hey", "heya", "hello", "helo", "hola", "namaste", "greetings",
    "yo", "sup", "howdy", "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thankyou", "thx", "ty", "cheers", "appreciate it",
    "much appreciated", "thanks a lot", "thank you so much", "thanks very much",
    "bye", "byebye", "goodbye", "see you", "see ya", "later", "take care",
    "ok", "okay", "k", "got it", "cool", "nice", "great", "awesome", "perfect",
    "sounds good", "alright", "sure", "yes", "no", "hmm", "hm",
    "how are you", "how are you doing", "hows it going", "whats up", "wassup",
    "who are you", "what are you", "what is your name", "whats your name", "your name",
    "what can you do", "what do you do", "what can i ask", "what can i ask you",
    "how do i use this", "how does this work", "how does it work",
    "what is this", "what is this app", "what is rbirag", "whats rbirag",
    "help", "help me", "i need help", "can you help", "can you help me",
    "are you there", "you there", "test", "testing",
}

_SMALLTALK_RX = [
    re.compile(p) for p in (
        r"^(hi|hey|hello|namaste|greetings)( there| bot| rbirag)?$",
        r"^(good )?(morning|afternoon|evening|night)$",
        r"^(who|what) (are|r) (you|u)( exactly)?$",
        r"^whats (your name|this|this app|rbirag)$",
        r"^(what|which) (can|could) (you|u) (do|help( me)?( with)?)$",
        r"^how (are|r) (you|u)( doing| today)?$",
        r"^(how )?(do|does) (i|you) (use|work)( this| it)?$",
        r"^(can|could) you help( me)?$",
        r"^(i )?(need|want) help$",
        r"^(are|r) (you|u) (there|online|awake)$",
        r"^(thanks|thank you|thx|ty)( a lot| so much| very much)?$",
        r"^(bye|goodbye|see (you|ya)|take care|later)( for now)?$",
        r"^(ok|okay|k|got it|cool|nice|great|awesome|perfect|sounds good|alright|sure|"
        r"yes|no|hmm+|hm+)$",
    )
]


def is_smalltalk(query: str) -> bool:
    """True when the whole message is greeting/small talk rather than a
    regulatory question. Deliberately a full-message match: a question that
    merely starts with "hi" or contains "help" stays on the retrieval path."""
    text = (query or "").lower().replace("\u2019", "'").strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    return text in _SMALLTALK or any(rx.match(text) for rx in _SMALLTALK_RX)


# A follow-up turn ("what about NBFCs?", "and for co-operative banks?") reads
# as a topic-less question on its own, so retrieval would search the literal
# words. Detecting it lets the previous question ride along as context.
_FOLLOWUP_RX = re.compile(
    r"^(and|also|so|then|what about|how about|what if|same|and for|what of|"
    r"how does that|does that|is that|are there)\b", re.I)
_ANAPHORA_RX = re.compile(
    r"\b(it|its|that|this|these|those|they|them|there|such)\b", re.I)


def contextual_query(query: str, history: list[dict] | None = None) -> str:
    """Retrieval query for a follow-up turn.

    Returns the query unchanged unless it reads as a follow-up: an explicit
    connector ("what about..."), or an anaphor inside a short message. In that
    case the previous user question is prepended so the retriever sees the
    topic. The answer prompt still receives the user's original words.
    """
    q = (query or "").strip()
    if not q or not history:
        return q
    prev = ""
    for turn in reversed(history):
        if turn.get("role") == "user" and (turn.get("content") or "").strip():
            prev = turn["content"].strip()
            break
    if not prev:
        return q
    follow = bool(_FOLLOWUP_RX.match(q)) or (
        len(q.split()) <= 12 and bool(_ANAPHORA_RX.search(q)))
    return f"{prev} {q}" if follow else q


def scope_label(entity_scope: str | None, entities: list[str] | None) -> str:
    """Human label of who a passage applies to ("Commercial Banks, Authorised Dealers")."""
    if entity_scope == "broad" or not entities:
        return "all regulated entities"
    return ", ".join(ENTITY_LABELS.get(e, e) for e in entities)


def entity_groups(items) -> list[tuple[str, ...]]:
    """Distinct entity groups among the entity-specific passages (Hits or citation
    cards); cross-cutting (broad) passages apply to everyone and are not counted."""
    groups: dict[tuple[str, ...], None] = {}
    for it in items:
        get = it.get if isinstance(it, dict) else (lambda k, d=None: getattr(it, k, d))
        if get("entity_scope") == "specific" and get("entities"):
            groups[tuple(sorted(get("entities")))] = None
    return list(groups)


def _resolve_cards(hits: list[Hit]) -> list[dict]:
    cards = []
    for i, h in enumerate(hits, start=1):
        cards.append({
            "ref": f"C{i}",
            "chunk_id": h.chunk_id,
            "title": h.title,
            "reference": h.reference,
            "issue_date": h.issue_date,
            "doc_type": h.doc_type,
            "status": h.status,
            "effective_from": h.effective_from,
            "amends": h.amends,
            "amended_by": h.amended_by,
            "entities": h.entities,
            "entity_scope": h.entity_scope,
            "section": h.section,
            "para": h.para,
            "para_end": h.para_end,
            "page_start": h.page_start,
            "page_end": h.page_end,
            "page_url": h.page_url,
            "pdf_url": h.pdf_url,
            "scores": h.scores,
            "snippet": h.text[:420],
        })
    return cards


_NUM_PARA = re.compile(r"^\d+(?:\.\d+)*$")


def format_para(para: str, para_end: str | None = None) -> str | None:
    """Human citation label: "Para 4.2" or "Paras 4.2-4.4" when the passage
    spans a contiguous numeric run. Mixed labels ((a), (ii), Explanation) keep
    the opening label only: a range across label styles is not a legal span."""
    if not para:
        return None
    end = (para_end or "").strip()
    if not end or end == para:
        return f"Para {para}"
    if _NUM_PARA.match(para) and _NUM_PARA.match(end):
        if end.startswith(para + "."):
            return f"Para {para}"          # the parent covers its sub-paragraphs
        return f"Paras {para}\u2013{end}"
    return f"Para {para}"


def build_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        tag = None
        if h.status != "current":
            tag = (f"[{h.status} \u2014 effective {h.effective_from}]"
                   if h.status == "forthcoming" and h.effective_from else f"[{h.status}]")
        meta = " | ".join(x for x in [
            h.title, h.reference, f"applies to: {scope_label(h.entity_scope, h.entities)}", f"dated {h.issue_date}" if h.issue_date else None,
            h.section or None, format_para(h.para, h.para_end),
            f"pp. {h.page_start}-{h.page_end}" if h.page_start else None,
            tag] if x)
        parts.append(f"[C{i}] {meta}\n{h.text}")
    return "\n\n".join(parts)


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in a reply (handles code fences and prose).

    Uses a full JSON decoder positioned at the first '{', so braces and
    escapes inside strings cannot confuse the boundary search."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_claims(claims: list[dict], n_passages: int) -> tuple[list[dict], int]:
    """Keep claims with >=1 resolvable [Cn] citation; return (claims, dropped)."""
    valid: list[dict] = []
    dropped = 0
    for c in claims or []:
        if not isinstance(c, dict) or not (c.get("text") or "").strip():
            dropped += 1
            continue
        cites = []
        for cit in c.get("citations") or []:
            m = re.fullmatch(r"C(\d+)", str(cit).strip())
            if m and 1 <= int(m.group(1)) <= n_passages:
                cites.append(f"C{int(m.group(1))}")
        if cites:
            valid.append({"text": str(c["text"]).strip(), "citations": sorted(set(cites))})
        else:
            dropped += 1
    return valid, dropped


# ---------------------------------------------------------------------------
# Numeric grounding (the agenticresearch eval rule — "every number in the
# answer must appear in tool output" — applied to the cited passages)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm_numbers(text: str) -> set[str]:
    t = (text.replace("per cent", "%").replace("percent", "%")
         .replace("per-cent", "%").replace(",", ""))
    return {m.group(0).rstrip(".") for m in _NUM_RE.finditer(t)}


def check_grounding(claim_text: str, passage_texts: list[str]) -> list[str]:
    """Numbers in the claim that appear in none of the cited passages."""
    have: set[str] = set()
    for p in passage_texts:
        have |= _norm_numbers(p)
    return sorted(n for n in _norm_numbers(claim_text) if n not in have)


def annotate_grounding(claims: list[dict], texts_by_ref: dict[str, str]) -> int:
    """Flag claims whose numbers are not verbatim in their cited passages.

    Returns the number of claims flagged and annotates each claim in place
    with `grounded: true` or `number_issues: [...]`. Flagged claims are kept —
    paraphrase like "ninety days" for "90 days" is legitimate; the UI shows
    the mismatch so a reviewer can check it.
    """
    ungrounded = 0
    for c in claims:
        joined = "\n".join(texts_by_ref.get(x, "") for x in c["citations"])
        issues = check_grounding(c["text"], [joined])
        if issues:
            c["number_issues"] = issues
            ungrounded += 1
        else:
            c["grounded"] = True
    return ungrounded


def _extractive(query: str, cards: list[dict]) -> str:
    lines = [f"**Passages retrieved for:** {query}", ""]
    for c in cards:
        meta = " · ".join(x for x in [c["title"], c["reference"],
                                      c["issue_date"], (f"para {c['para']}" if c["para"] else None)] if x)
        lines.append(f"**[{c['ref']}]** {meta}")
        lines.append(c["snippet"] + ("…" if len(c["snippet"]) == 420 else ""))
        lines.append("")
    return "\n".join(lines)


def answer(query: str, hits: list[Hit], low_confidence: bool = False,
           use_llm: bool = True, max_tokens: int = 2600,
           llm_pin: tuple[str, str] | None = None, strict_pin: bool = False,
           history: list[dict] | None = None) -> dict:
    """Returns the UI-ready answer object (mode: llm | extractive | abstain).

    `llm_pin` = (provider, model) chosen in the UI: tried first, with the
    normal failover chain as the fallback (``strict_pin``: no fallback).

    `history` = earlier turns as ``{"role": "user"|"assistant", "content": str}``.
    They are shown to the model so follow-up questions resolve ("what about
    NBFCs?"), but marked as context only: every fact must still come from the
    passages, and nothing in the history may be cited."""
    t0 = time.time()
    cards = _resolve_cards(hits)
    result = {
        "query": query,
        "mode": "abstain",
        "abstain": True,
        "low_confidence": low_confidence,
        "answer_md": "",
        "claims": [],
        "claims_dropped": 0,
        "claims_ungrounded": 0,
        "parse_retries": 0,
        "citations": cards,
        "model": None,
        "provider": None,
        "pin": list(llm_pin) if llm_pin else None,
        "cached": False,
        "latency_s": 0.0,
        "errors": [],
    }

    if not hits:
        result["answer_md"] = ("No passage in the corpus (under the current filters) covers this "
                               "question. Try widening the institution filter or syncing new "
                               "notifications first.")
        result["latency_s"] = round(time.time() - t0, 2)
        return result

    if not (use_llm and llm.available()):
        result.update(mode="extractive", abstain=False,
                      answer_md=_extractive(query, cards))
        result["latency_s"] = round(time.time() - t0, 2)
        return result

    convo = ""
    if history:
        lines = []
        for turn in history[-6:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                who = "User" if role == "user" else "Assistant"
                lines.append(f"{who}: {content[:400]}")
        if lines:
            convo = ("CONVERSATION SO FAR (context only \u2014 not evidence, never cite it):\n"
                     + "\n".join(lines) + "\n\n")
    follow = ("Resolve references such as 'that', 'it' or 'what about X' from the "
              "conversation above, but answer only from the passages.\n\n") if convo else ""
    user = (f"{convo}QUESTION: {query}\n\nCONTEXT:\n{build_context(hits)}\n\n"
            f"{follow}Answer with strict JSON per the system instructions.")
    if low_confidence:
        user += ("\n\nNOTE: the retrieval layer flagged LOW confidence for this question. "
                 "Only answer if the passages clearly and directly answer it; otherwise "
                 'set "abstain": true.')
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    try:
        call = llm.chat(messages, max_tokens=max_tokens, pin=llm_pin, strict_pin=strict_pin)
        parsed = extract_json(call["text"])
        if parsed is None:
            # Reasoning models can run the token budget out mid-JSON (a valid
            # prefix with a truncated tail). One retry with headroom; the
            # retry is cached separately, so reruns stay cheap.
            result["parse_retries"] = 1
            call = llm.chat(messages, max_tokens=max(max_tokens * 2, 4000), pin=llm_pin,
                            strict_pin=strict_pin)
            parsed = extract_json(call["text"])
        if parsed is None:
            raise ValueError("reply is not valid JSON")
        claims, dropped = validate_claims(parsed.get("claims") or [], len(hits))
        ungrounded = annotate_grounding(
            claims, {f"C{i}": h.text for i, h in enumerate(hits, start=1)})
        answer_md = (parsed.get("answer") or "").strip()
        abstain = bool(parsed.get("abstain")) or (not claims and not answer_md)
        if not answer_md and abstain:
            answer_md = ("The retrieved passages do not contain a direct answer. Closest sources "
                         "are listed below in the audit trail.")
        result.update(mode="abstain" if abstain else "llm", abstain=abstain,
                      answer_md=answer_md, claims=claims, claims_dropped=dropped,
                      claims_ungrounded=ungrounded,
                      model=call["model"], provider=call["provider"], cached=call["cached"])
    except Exception as exc:  # noqa: BLE001
        result.update(mode="extractive", abstain=False, answer_md=_extractive(query, cards),
                      errors=[f"{type(exc).__name__}: {str(exc)[:200]}"])
    result["latency_s"] = round(time.time() - t0, 2)
    return result


def refuse(query: str, hits: list[Hit], threshold: float) -> dict:
    """Retrieval-level refusal: no LLM call, passages kept for the audit trail.

    Used by the app's "refuse low-confidence queries" mode when the best
    passage scores below the confidence cutoff. Same shape as ``answer()``
    so the UI renders it identically.
    """
    return {
        "query": query,
        "mode": "abstain",
        "abstain": True,
        "low_confidence": True,
        "answer_md": (
            f"No passage scored above the retrieval confidence threshold "
            f"({threshold:+.1f} on the cross-encoder logit) for this question, so the "
            "answer layer was not asked. The closest passages are listed in the audit "
            "trail \u2014 lower the threshold or widen the filters if you expected an "
            "answer here."),
        "claims": [],
        "claims_dropped": 0,
        "claims_ungrounded": 0,
        "parse_retries": 0,
        "citations": _resolve_cards(hits),
        "model": None,
        "provider": None,
        "pin": None,
        "cached": False,
        "latency_s": 0.0,
        "errors": [],
    }


def chat_reply(query: str, history: list[dict] | None = None, use_llm: bool = True,
               llm_pin: tuple[str, str] | None = None,
               strict_pin: bool = False) -> dict:
    """Conversational reply for greetings and questions about the assistant.

    Same result shape as ``answer()`` so the UI renders it identically, with
    ``mode="chat"``, no citations and nothing for the audit trail to resolve.
    Falls back to a fixed greeting when the LLM is off or unreachable, so a
    "hi" is never answered with a retrieval refusal.
    """
    t0 = time.time()
    result = {
        "query": query,
        "mode": "chat",
        "abstain": False,
        "low_confidence": False,
        "answer_md": CHAT_FALLBACK,
        "claims": [],
        "claims_dropped": 0,
        "claims_ungrounded": 0,
        "parse_retries": 0,
        "citations": [],
        "model": None,
        "provider": None,
        "pin": list(llm_pin) if llm_pin else None,
        "cached": False,
        "latency_s": 0.0,
        "errors": [],
    }
    if not (use_llm and llm.available()):
        result["latency_s"] = round(time.time() - t0, 2)
        return result

    messages = [{"role": "system", "content": CHAT_PROMPT}]
    for turn in (history or [])[-4:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:400]})
    messages.append({"role": "user", "content": query})
    try:
        call = llm.chat(messages, temperature=0.4, max_tokens=400, pin=llm_pin,
                        strict_pin=strict_pin)
        text = (call.get("text") or "").strip()
        if not text:
            raise ValueError("empty completion")
        result.update(answer_md=text, model=call["model"], provider=call["provider"],
                      cached=call["cached"])
    except Exception as exc:  # noqa: BLE001
        result["errors"] = [f"{type(exc).__name__}: {str(exc)[:200]}"]
    result["latency_s"] = round(time.time() - t0, 2)
    return result
