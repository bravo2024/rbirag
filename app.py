"""RBIRAG — RBI Regulatory Intelligence RAG (Streamlit UI).

Tabs
  Ask        grounded answers with claim-level citations + full retrieval audit trail
  Circulars  browse every document with its parsed text and provenance
  Compare    side-by-side circular inspection with paragraph-overlap highlighting
  Audit      every query, filter set and retrieved passage, exportable

Sidebar
  institution filter (commercial bank / NBFC-HFC group / all), doc-type,
  theme and date filters, retrieval settings, and the RBI updates panel
  (check the feed, sync new notifications + re-crawl the Master Directions
  listings, optional 2-minute auto-sync).
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import date
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from regintel import audit, classify, config, ingest, llm, store  # noqa: E402
from regintel import answer as answer_mod  # noqa: E402
from regintel.classify import ENTITY_LABELS, THEME_LABELS  # noqa: E402
from regintel.index import Engine, QueryFilters  # noqa: E402

DASH = "\u2014"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun if n == 1 else noun + 's'}"


def _link_titles(ids: list[str], docs: dict | None) -> str:
    out = []
    for did in ids or []:
        title = ((docs or {}).get(did) or {}).get("title") or did
        out.append(f"{title[:70]} (`{did}`)")
    return "; ".join(out)


st.set_page_config(page_title="RBIRAG \u2014 RBI Regulatory Intelligence",
                   page_icon="\u00a7", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2.1rem; padding-bottom: 4rem; max-width: 1120px;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}

.rbi-hero h1 {font-size: 1.7rem; line-height: 1.2; letter-spacing: -0.01em;
              margin: 0 0 .35rem 0;}
.rbi-hero p {color: #6b7280; font-size: .95rem; margin: 0; max-width: 68ch;}
.rbi-stats {color: #9ca3af; font-size: .78rem; margin-top: .7rem;}
.rbi-stats a {color: #9ca3af; text-decoration: underline;}

.rbi-meta {color: #9ca3af; font-size: .78rem; margin: 0 0 .5rem 0;}
.rbi-meta b {color: #6b7280; font-weight: 600;}

.rbi-claims {margin: .9rem 0 0 0; border-top: 1px solid rgba(128,128,128,.18);
             padding-top: .65rem;}
.rbi-claims .h {font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
                color: #9ca3af; margin-bottom: .45rem;}
.rbi-claim {margin: 0 0 .45rem 0; font-size: .9rem; line-height: 1.5; color: #4b5563;}
.rbi-ref {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: .7rem; color: #6b7280; background: rgba(128,128,128,.12);
          border-radius: 3px; padding: .04rem .28rem; margin-left: .2rem;}
.rbi-note {display: block; font-size: .74rem; color: #b45309; margin-top: .12rem;}
</style>
"""

EXAMPLES = [
    ("Capital adequacy", "What is the minimum capital adequacy ratio for small finance banks?"),
    ("KYC", "How often must KYC records be updated for high-risk customers?"),
    ("Out of scope", "What is SEBI's framework for insider trading?"),
]


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources (invalidated by the corpus version in the manifest)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading retrieval index\u2026")
def _engine(version: str) -> Engine:
    return Engine()


@st.cache_resource(show_spinner=False)
def _documents(version: str) -> dict:
    return store.load_documents()


def corpus_version() -> str:
    m = store.read_manifest()
    return (m or {}).get("version", "none")


def engine_and_docs() -> tuple[Engine | None, dict, str]:
    v = corpus_version()
    docs = _documents(v)
    eng = _engine(v) if v != "none" else None
    return eng, docs, v


@st.cache_data(ttl=600, show_spinner=False)
def _llm_status() -> list[dict]:
    """Provider key state + live model counts for the sidebar (10-min TTL;
    the per-provider /models discovery inside is cached just as long)."""
    return llm.provider_status()


# ---------------------------------------------------------------------------
# Sidebar — filters + RBI updates
# ---------------------------------------------------------------------------
def sidebar_filters() -> dict:
    st.sidebar.title("RBIRAG")
    st.sidebar.caption("RBI Master Directions and notifications, with an audit trail "
                       "for every answer.")

    labels = {k: v["label"] for k, v in config.INSTITUTIONS.items()}
    inst_key = st.sidebar.radio(
        "Institution view", options=list(config.INSTITUTIONS), format_func=lambda k: labels[k],
        help="Filters the corpus to mandates that apply to the selected institution "
             "(cross-cutting RBI directions are always included).")
    st.sidebar.divider()

    with st.sidebar.expander("Document filters", expanded=True):
        doc_types = st.multiselect("Instrument type", config.DOC_TYPES, default=[])
        themes = st.multiselect("Theme", THEME_LABELS, default=[])
        since = st.date_input("Issued since", value=None,
                              min_value=date(2018, 1, 1), max_value=date.today())
        include_withdrawn = st.checkbox("Include withdrawn / superseded documents", value=False)
        as_of = st.date_input(
            "In force on (point-in-time)", value=None,
            min_value=date(2018, 1, 1), max_value=date.today(),
            help="Keeps only document versions in force on this date. A superseded "
                 "version is kept when the date predates its replacement; documents "
                 "known to be effective later are excluded. Undated documents are "
                 "kept unless the strict option below is on.")
        strict_as_of = st.checkbox(
            "Strict point-in-time (exclude undated / later-updated text)", value=False,
            help="Also drops undated versions and versions whose text was last "
                 "updated after the date. The corpus keeps current text, not "
                 "historical snapshots, so strict mode can return little for older "
                 "dates \u2014 that is the honest answer when the historical text is "
                 "not held.")
        regulatory_only = st.checkbox("Regulatory content only", value=True,
                                      help="Drops market-operations notices "
                                           "(auctions, T-bills, rates) from retrieval.")

    with st.sidebar.expander("Retrieval settings", expanded=False):
        rerank = st.checkbox("Cross-encoder reranking", value=True)
        top_k = st.slider("Passages in context", 4, 10, config.TOP_K)
        abstain_score = st.slider(
            "Abstain threshold (rerank score)", -12.0, 6.0, config.ABSTAIN_SCORE, 0.5,
            help="Retrieval-level confidence cutoff on the cross-encoder logit (raw "
                 "score, not a probability). A query whose best passage scores below "
                 "it is low-confidence. Calibrated 2026-10-17: in-scope eval queries "
                 "scored \u2265 3.7, out-of-scope mostly \u2264 0.5. Takes effect while "
                 "cross-encoder reranking is on.")
        gate = st.checkbox(
            "Refuse low-confidence queries without the LLM", value=True,
            help="On: a query below the cutoff is refused immediately \u2014 no LLM "
                 "call, the closest passages are still listed in the audit trail. "
                 "Off: the LLM is asked with a caution to abstain unless the passages "
                 "directly answer the question.")
        retrieve_only = st.checkbox("Retrieve-only mode (no LLM)", value=False)

    llm_pin: tuple[str, str] | None = None
    with st.sidebar.expander("LLM provider", expanded=False):
        prov_choice = st.selectbox(
            "Provider", ["Auto (failover)"] + llm.ORDER,
            help="Auto tries providers in order \u2014 keyed first (Groq \u2192 Gemini \u2192 "
                 "NVIDIA Build), then the free gateway \u2014 skipping providers without a "
                 "key and models that recently failed. Pinning tries the chosen model "
                 "first, with the chain as fallback.")
        if prov_choice == "Auto (failover)":
            st.caption("Model is selected per call by the failover chain.")
        else:
            opts = llm.available_models(prov_choice) or llm.PROVIDERS[prov_choice]["models"]
            model_choice = st.selectbox(
                "Model", opts,
                help="Live list from the provider's /models endpoint (cached 10 min), "
                     "curated free models first. Falls back to the built-in list when "
                     "the endpoint is unreachable.")
            llm_pin = (prov_choice, model_choice)
        st.caption("Provider status")
        for row in _llm_status():
            live = f"{row['live_models']} models" if row["live_models"] else "no models"
            first = f", first {row['first']}" if row["first"] else ""
            st.caption(f"{row['provider']}: {row['key']}, {live}{first}")
        st.caption("Keys come from the environment or Streamlit secrets \u2014 flat, or one "
                   "section deep. Groq, Gemini and NVIDIA Build have free tiers.")

    return {
        "entities": frozenset(config.INSTITUTIONS[inst_key]["entities"]),
        "institution_key": inst_key,
        "institution_label": labels[inst_key],
        "doc_types": frozenset(doc_types),
        "themes": frozenset(themes),
        "since": since.isoformat() if since else None,
        "as_of": as_of.isoformat() if as_of else None,
        "strict_as_of": strict_as_of,
        "include_withdrawn": include_withdrawn,
        "regulatory_only": regulatory_only,
        "rerank": rerank,
        "top_k": top_k,
        "abstain_score": abstain_score,
        "gate": gate,
        "retrieve_only": retrieve_only,
        "llm_pin": llm_pin,
    }


def filters_from(ui: dict) -> QueryFilters:
    return QueryFilters(entities=ui["entities"], doc_types=ui["doc_types"],
                        themes=ui["themes"], since=ui["since"],
                        as_of=ui.get("as_of"), strict_as_of=ui.get("strict_as_of", False),
                        include_withdrawn=ui["include_withdrawn"],
                        regulatory_only=ui["regulatory_only"])


def rbi_updates_panel() -> None:
    with st.sidebar.expander("RBI updates (feed + listings)", expanded=False):
        seen = json.loads(config.SEEN.read_text(encoding="utf-8")) if config.SEEN.exists() else {}
        m = store.read_manifest() or {}
        c1, c2 = st.columns(2)
        c1.metric("Last feed poll", (seen.get("last_poll") or "\u2014")[11:16] or "\u2014")
        c2.metric("Corpus built", (m.get("built_at") or "\u2014")[11:16] or "\u2014")
        st.caption(f"{m.get('n_docs', 0)} instruments across RBI's Master Directions "
                   "listings, plus the latest notifications. Every sync re-crawls the "
                   "listings, so new directions are picked up and pages RBI updated in "
                   "place are re-ingested as new versions.")

        if st.button("Check RSS for new notifications", width="stretch"):
            with st.spinner("Polling the RBI feed\u2026"):
                upd = ingest.check_updates()
            st.session_state["update_check"] = upd
        upd = st.session_state.get("update_check")
        if upd:
            if not upd.get("ok"):
                st.error(f"Feed unreachable: {upd.get('error')}")
            elif upd["new"]:
                st.warning(f"{_plural(upd['new'], 'new notification')} in the feed:")
                for it in upd["items"][:5]:
                    st.caption(f"\u2022 {it['published']} \u2014 {it['title'][:70]}")
            else:
                st.success("Feed checked \u2014 nothing new.")

        sync_clicked = st.button("Sync now (feed + listings)", type="primary",
                                 width="stretch")
        auto = st.checkbox("Auto-sync every 2 minutes", value=False,
                           help="Checks the feed and every Master Directions listing "
                                "every 2 minutes and ingests anything new, then reloads "
                                "the corpus.")
        if sync_clicked:
            _run_sync()
        if auto:
            _auto_check()


def _run_sync(label: str = "Syncing from RBI\u2026") -> None:
    status = st.status(label, expanded=True)
    try:
        result = ingest.sync(source="both", refresh=True, progress=status.write)
        if result.get("skipped"):
            status.update(label=f"Sync skipped — {result['reason']}", state="error",
                          expanded=False)
            return
        new = result["rss"]["new_docs"] + result["backfill"]["new_docs"]
        refreshed = result.get("refresh", {}).get("refreshed", 0)
        links = result.get("links", {}).get("linked", 0)
        extra = (f", {_plural(refreshed, 'updated direction')} re-ingested" if refreshed else "")
        extra += (f", {_plural(links, 'amendment link')}" if links else "")
        status.update(
            label=f"Sync complete \u2014 {_plural(new, 'new document')} "
                  f"({result['rss']['new_docs']} feed, {result['backfill']['new_docs']} listings)"
                  f"{extra}",
            state="complete", expanded=False)
        st.session_state.pop("update_check", None)  # the pre-sync warning is stale now
        st.cache_resource.clear()
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        status.update(label=f"Sync failed: {exc}", state="error")


@st.fragment(run_every=120)
def _auto_check() -> None:
    """Every 2 minutes: poll the feed; if something new appears, ingest it."""
    upd = ingest.check_updates()
    if upd.get("ok") and upd["new"]:
        _run_sync(label=f"Auto-sync: {_plural(upd['new'], 'new notification')} found\u2026")


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------
def render_citation_card(i: int, card: dict, docs: dict | None = None) -> None:
    head = f"[{card['ref']}] {card.get('reference') or 'RBI'} \u2014 {card['title'][:64]}"
    with st.expander(head, expanded=False):
        cols = st.columns(4)
        cols[0].markdown(f"**Date**  \n{card.get('issue_date') or DASH}")
        cols[1].markdown(f"**Para**  \n{answer_mod.format_para(card.get('para'), card.get('para_end')) or DASH}")
        cols[2].markdown(f"**Pages**  \n{card.get('page_start')}\u2013{card.get('page_end')}")
        cols[3].markdown(f"**Passage**  \n`{card['chunk_id']}`")
        tags = []
        if card.get("entity_scope") == "broad":
            tags.append("All regulated entities")
        tags += [ENTITY_LABELS.get(e, e) for e in card.get("entities") or []]
        if card.get("status") and card["status"] != "current":
            tags.append(card["status"].upper() +
                        (f" \u00b7 effective {card['effective_from']}"
                         if card.get("effective_from") else ""))
        if tags:
            st.caption(" \u00b7 ".join(tags))
        if card.get("section"):
            st.caption(f"Section: {card['section']}")
        if card.get("amends"):
            st.caption(f"**Amends:** {_link_titles(card['amends'], docs)}")
        if card.get("amended_by"):
            st.caption(f"**Amended by:** {_link_titles(card['amended_by'], docs)} "
                       "\u2014 verify the amendment's effect on this passage.")
        st.markdown(f"> {escape(card['snippet'])}\u2026")
        s = card.get("scores") or {}
        bits = []
        if s.get("rerank") is not None:
            bits.append(f"rerank {s['rerank']:.3f}")
        if s.get("dense_rank") is not None:
            bits.append(f"dense #{s['dense_rank']}")
        if s.get("bm25_rank") is not None:
            bits.append(f"BM25 #{s['bm25_rank']}")
        if bits:
            st.caption(" \u00b7 ".join(bits))
        links = []
        if card.get("pdf_url"):
            links.append(f"[Official PDF]({card['pdf_url']})")
        if card.get("page_url"):
            links.append(f"[RBI page]({card['page_url']})")
        if links:
            st.markdown(" \u00b7 ".join(links))


def render_assistant(msg: dict, docs: dict | None = None) -> None:
    ans, filt, trace = msg["answer"], msg["filters"], msg["trace"]
    mode = ans["mode"]
    label = {"llm": "Grounded", "extractive": "Extractive", "chat": "Chat",
             "abstain": "Refused"}.get(mode, mode)
    bits = [f"<b>{label}</b>"]
    if ans.get("provider") or ans.get("model"):
        bits.append(escape(f"{ans.get('provider') or '?'}/{ans.get('model') or '?'}"))
    if ans.get("cached"):
        bits.append("cached")
    if ans.get("latency_s") is not None:
        bits.append(f"{ans['latency_s']:.1f} s")
    if mode != "chat":
        bits.append(escape(filt["institution_label"]))
    st.markdown(f"<div class='rbi-meta'>{' &middot; '.join(bits)}</div>",
                unsafe_allow_html=True)
    if ans.get("low_confidence") and ans["mode"] == "llm":
        st.caption("Low retrieval confidence \u2014 verify against the cited text.")
    if filt.get("institution_key") == "all" and ans["citations"]:
        groups = answer_mod.entity_groups(ans["citations"])
        if len(groups) > 1:
            names = "; ".join(" + ".join(ENTITY_LABELS.get(e, e) for e in g) for g in groups)
            st.caption(f"Passages span different institution types ({names}); their rules can "
                       "differ. Pick the matching institution view in the sidebar for the "
                       "version that applies to you.")
        elif groups and ans["citations"][0].get("entity_scope") == "specific":
            ents = ", ".join(ENTITY_LABELS.get(e, e) for e in ans["citations"][0]["entities"])
            st.caption(f"Top passages are the {ents} version of these rules.")
    st.markdown(ans["answer_md"])
    flagged = (ans.get("claims_dropped") or ans.get("claims_ungrounded")
               or any(c.get("number_issues") for c in ans["claims"]))
    if ans["claims"] and flagged:
        rows = ["<div class='rbi-claims'><div class='h'>Claim check</div>"]
        for c in ans["claims"]:
            refs = "".join(f"<span class='rbi-ref'>[{escape(str(x))}]</span>"
                           for x in c["citations"])
            note = ""
            if c.get("number_issues"):
                note = (f"<span class='rbi-note'>numbers not found in cited text: "
                        f"{escape(', '.join(c['number_issues']))}</span>")
            rows.append(f"<div class='rbi-claim'>{escape(c['text'])}{refs}{note}</div>")
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
    if ans.get("claims_dropped"):
        st.caption(f"{_plural(ans['claims_dropped'], 'claim')} dropped by the citation validator.")
    if ans.get("claims_ungrounded"):
        st.caption(f"{_plural(ans['claims_ungrounded'], 'claim')} with numbers not found "
                   "in the cited passages \u2014 verify against the source.")
    if ans.get("errors"):
        st.caption(f"LLM note: {ans['errors'][0]}")
    if ans["citations"]:
        with st.popover(f"Sources \u2014 {_plural(len(ans['citations']), 'passage')}"):
            if isinstance(trace, dict) and trace.get("bm25_hits"):
                st.caption(f"BM25 {trace['bm25_hits']} \u00b7 dense {trace.get('dense_hits', 0)} "
                           f"\u00b7 fused {trace.get('fused', 0)} \u00b7 "
                           f"reranked {trace.get('reranked', 0)}")
            st.dataframe(pd.DataFrame([{
                "ref": c["ref"], "passage": c["chunk_id"], "reference": c.get("reference"),
                "title": c["title"][:60], "para": c.get("para"), "page": c.get("page_start"),
                "rerank": c["scores"].get("rerank"), "dense_rank": c["scores"].get("dense_rank"),
                "bm25_rank": c["scores"].get("bm25_rank"),
            } for c in ans["citations"]]), hide_index=True, width="stretch")
            for i, card in enumerate(ans["citations"], start=1):
                render_citation_card(i, card, docs)


def ask_tab(eng: Engine | None, docs: dict, ui: dict) -> None:
    if eng is None or not eng.ready:
        st.info("Corpus or index missing. Build it from the sidebar (RBI updates \u2192 "
                "Sync now) or run:\n\n```\npython scripts/sync.py --source both\n"
                "python scripts/build_index.py --full\n```")
        return
    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.session_id = uuid.uuid4().hex[:12]

    for msg in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(msg["q"])
        with st.chat_message("assistant"):
            render_assistant(msg, docs)

    if not st.session_state.chat and not st.session_state.get("pending_q"):
        cols = st.columns(len(EXAMPLES))
        for col, (label, example) in zip(cols, EXAMPLES):
            if col.button(label, key=f"ex-{label}", width="stretch", help=example):
                st.session_state.pending_q = example
                st.rerun()

    q = st.chat_input("Ask about RBI directions, circulars, thresholds, forms\u2026")
    if not q:
        q = st.session_state.get("pending_q")
        st.session_state.pending_q = None
    if not q:
        return
    history = []
    for prev in st.session_state.chat[-3:]:
        history.append({"role": "user", "content": prev["q"]})
        history.append({"role": "assistant", "content": prev["answer"]["answer_md"]})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if answer_mod.is_smalltalk(q):
            with st.spinner("Thinking\u2026"):
                ans = answer_mod.chat_reply(q, history=history,
                                            use_llm=not ui["retrieve_only"],
                                            llm_pin=ui["llm_pin"])
            result = {"hits": [], "abstain": False, "trace": {"mode": "chat"}}
        else:
            retrieval_q = answer_mod.contextual_query(q, history)
            with st.spinner("Retrieving relevant passages\u2026"):
                filt = filters_from(ui)
                result = eng.retrieve(retrieval_q, filters=filt, k=ui["top_k"],
                                      rerank=ui["rerank"],
                                      abstain_score=ui["abstain_score"])
            if retrieval_q != q:
                result["trace"]["retrieval_query"] = retrieval_q
            if result["abstain"] and result["hits"] and ui["gate"]:
                ans = answer_mod.refuse(q, result["hits"],
                                        threshold=float(result["trace"].get(
                                            "abstain_threshold", ui["abstain_score"])))
            else:
                with st.spinner("Drafting the grounded answer\u2026"):
                    ans = answer_mod.answer(q, result["hits"],
                                            low_confidence=bool(result["abstain"] and result["hits"]),
                                            use_llm=not ui["retrieve_only"],
                                            llm_pin=ui["llm_pin"], history=history)
        msg = {"q": q, "answer": ans, "trace": result["trace"], "filters": ui}
        audit.append_query(session=st.session_state.session_id, query=q,
                           filters={k: list(v) if isinstance(v, frozenset) else v
                                    for k, v in ui.items()},
                           k=ui["top_k"], rerank=ui["rerank"], trace=result["trace"],
                           hits=result["hits"], answer=ans, mode=ans["mode"])
        st.session_state.chat.append(msg)
        st.rerun()


# ---------------------------------------------------------------------------
# Circulars tab
# ---------------------------------------------------------------------------
def circulars_tab(docs: dict) -> None:
    if not docs:
        st.info("No documents yet \u2014 run a sync first.")
        return
    rows = sorted(docs.values(), key=lambda d: (d.get("issue_date") or ""), reverse=True)
    q = st.text_input("Search titles and references", "")
    if q:
        ql = q.lower()
        rows = [d for d in rows if ql in d["title"].lower()
                or ql in (d.get("reference") or "").lower()]
    label = lambda d: (f"{d.get('issue_date') or DASH}  |  {d['title'][:88]}  |  "
                       f"{d.get('reference') or ''}")
    pick = st.selectbox(_plural(len(rows), "document"), rows, format_func=label)
    if not pick:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instrument", pick["doc_type"].replace("_", " ").capitalize())
    c2.metric("Reference", pick.get("reference") or "\u2014")
    c3.metric("Dated", pick.get("issue_date") or pick.get("last_updated") or DASH)
    c4.metric("Passages", pick.get("n_chunks", 0))
    tags = ("All regulated entities" if pick.get("entity_scope") == "broad"
            else ", ".join(ENTITY_LABELS.get(e, e) for e in pick.get("entities") or []))
    source = {"rss": "RSS feed", "backfill": "RBI listings"}.get(
        pick.get("source"), pick.get("source") or DASH)
    status = classify.effective_status(pick.get("status"), pick.get("title") or "",
                                       pick.get("effective_from"))
    eff = (f" \u00b7 effective {pick['effective_from']}"
           if status == "forthcoming" and pick.get("effective_from") else "")
    st.caption(f"{status}{eff} \u00b7 {tags} \u00b7 {', '.join(pick.get('themes') or []) or 'no themes'} \u00b7 "
               f"source: {source} \u00b7 sha256 {pick.get('sha256', '')[:12]}\u2026")
    links = []
    if pick.get("pdf_url"):
        links.append(f"[Official PDF]({pick['pdf_url']})")
    if pick.get("page_url"):
        links.append(f"[RBI page]({pick['page_url']})")
    if links:
        st.markdown(" \u00b7 ".join(links))
    with st.container(height=460, border=True):
        for p in pick.get("pages", []):
            st.markdown(f"<div style='color:#888;font-size:0.75rem'>page {p['page']}</div>",
                        unsafe_allow_html=True)
            st.markdown(p["text"])
            if p.get("footnotes"):
                st.caption("Notes: " + " ".join(p["footnotes"]))


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------
def _paragraphs(doc: dict) -> list[str]:
    text = "\n\n".join(p["text"] for p in doc.get("pages", []))
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 60]


def _overlap_pairs(pa: list[str], pb: list[str], ratio: float = 0.62) -> set[tuple[int, int]]:
    """Paragraph pairs across two documents that are near-duplicates.

    Candidate generation goes through a token inverted index (rare shared
    tokens) so 1000+ paragraph documents compare in well under a second;
    SequenceMatcher only runs on strong candidates.
    """
    import re as _re

    stop = {"the", "and", "shall", "that", "with", "this", "within", "from", "for", "under",
            "any", "all", "such", "which", "have", "been", "their", "entity", "entities"}

    def tokens(s: str) -> set[str]:
        toks = [t for t in _re.findall(r"[a-z]{3,}", s.lower()) if t not in stop]
        return set(toks)

    def norm(s: str) -> str:
        return _re.sub(r"\W+", " ", s.lower())[:400]

    tb = [tokens(p) for p in pb]
    index: dict[str, list[int]] = {}
    for j, toks in enumerate(tb):
        for t in toks:
            index.setdefault(t, []).append(j)

    hits: set[tuple[int, int]] = set()
    for i, a in enumerate(pa):
        ta = tokens(a)
        if not ta:
            continue
        cands: dict[int, int] = {}
        for t in ta:
            js = index.get(t) or []
            if len(js) > 60:  # ubiquitous token — not discriminative
                continue
            for j in js:
                cands[j] = cands.get(j, 0) + 1
        if not cands:
            continue
        na = norm(a)
        for j, shared in sorted(cands.items(), key=lambda kv: -kv[1])[:12]:
            tbj = tb[j]
            jac = shared / max(len(ta) + len(tbj) - shared, 1)
            if jac < 0.45:
                continue
            nb = norm(pb[j])
            if SequenceMatcher(None, na, nb).quick_ratio() >= ratio and \
                    SequenceMatcher(None, na, nb).ratio() >= ratio:
                hits.add((i, j))
    return hits


def _render_paras(paras: list[str], overlap_idx: set[int], highlight: bool,
                  only_overlap: bool) -> None:
    chunks = []
    for i, p in enumerate(paras):
        if only_overlap and i not in overlap_idx:
            continue
        body = escape(p)
        if highlight and i in overlap_idx:
            body = f"<mark style='background:#fff3b0'>{body}</mark>"
        chunks.append(f"<p style='margin:0.4rem 0'>{body}</p>")
    st.markdown("".join(chunks) or "<i>No paragraphs to display.</i>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def _overlap_cached(doc_a_id: str, doc_b_id: str) -> list[list[int]]:
    docs = store.load_documents()
    if doc_a_id not in docs or doc_b_id not in docs:
        return []
    pairs = _overlap_pairs(_paragraphs(docs[doc_a_id]), _paragraphs(docs[doc_b_id]))
    return sorted([i, j] for i, j in pairs)


def compare_tab(docs: dict) -> None:
    if len(docs) < 2:
        st.info("Need at least two documents to compare.")
        return
    rows = sorted(docs.values(), key=lambda d: (d.get("issue_date") or ""), reverse=True)
    label = lambda d: f"{d.get('issue_date') or DASH} | {d['title'][:80]}"
    ca, cb = st.columns(2)
    da = ca.selectbox("Left document", rows, index=0, format_func=label, key="cmp_a")
    db = cb.selectbox("Right document", rows, index=min(1, len(rows) - 1),
                      format_func=label, key="cmp_b")
    hl1, hl2 = st.columns([1, 1])
    highlight = hl1.toggle("Highlight overlapping paragraphs (\u226562% similar)", value=True)
    only = hl2.toggle("Show only overlapping paragraphs", value=False)

    pa, pb = _paragraphs(da), _paragraphs(db)
    pairs = ({(i, j) for i, j in _overlap_cached(da["doc_id"], db["doc_id"])}
             if (highlight or only) else set())
    ia = {i for i, _ in pairs}
    ib = {j for _, j in pairs}
    if highlight or only:
        st.caption(f"{len(ia)} of {len(pa)} paragraphs on the left have a near-match in the right "
                   f"document ({len(pairs)} paragraph pairs).")
    la, lb = st.columns(2)
    for col, doc, paras, idx in ((la, da, pa, ia), (lb, db, pb, ib)):
        with col:
            st.markdown(f"**{doc['title'][:90]}**")
            st.caption(f"{doc.get('reference') or ''} \u00b7 {doc.get('issue_date') or ''} \u00b7 "
                       f"{doc.get('doc_type', '').replace('_', ' ')}")
            if doc.get("pdf_url"):
                st.markdown(f"[Official PDF]({doc['pdf_url']})")
            with st.container(height=420, border=True):
                _render_paras(paras, idx, highlight, only)


# ---------------------------------------------------------------------------
# Audit tab
# ---------------------------------------------------------------------------
def audit_tab() -> None:
    entries = audit.load_recent(300)
    if not entries:
        st.caption("No queries recorded yet \u2014 ask something and the trail appears here.")
        return
    st.markdown(f"**{_plural(len(entries), 'query record')}** \u2014 question, filters, "
                "retrieved passages and answer settings for each.")
    table = pd.DataFrame([{
        "ts": e["ts"], "question": e["query"][:80],
        "institution": e["filters"].get("institution_label"),
        "mode": e["mode"], "model": (e["answer"].get("model") or "\u2014")[:42],
        "abstain": e["answer"].get("abstain"), "claims": e["answer"].get("claims"),
        "latency_s": e["answer"].get("latency_s"), "hits": len(e["hits"]),
    } for e in entries])
    st.dataframe(table, hide_index=True, width="stretch")

    pick = st.selectbox("Inspect a record", entries,
                        format_func=lambda e: f"{e['ts']} \u2014 {e['query'][:70]}")
    st.json(pick)
    if config.AUDIT_LOG.exists():
        st.download_button("Download full audit trail (JSONL)",
                           data=config.AUDIT_LOG.read_text(encoding="utf-8"),
                           file_name="regintel_audit.jsonl", mime="application/jsonl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render_header(docs: dict, m: dict) -> None:
    chunks = m.get("n_chunks")
    bits = [f"{len(docs)} instruments"]
    if isinstance(chunks, int):
        bits.append(f"{chunks:,} passages")
    built = str(m.get("built_at") or "")[:10]
    if built:
        bits.append(f"built {built}")
    st.markdown(
        "<div class='rbi-hero'><h1>RBIRAG</h1>"
        "<p>Grounded answers over RBI Master Directions and notifications. Every answer "
        "cites the instrument, paragraph and page, or refuses.</p></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='rbi-stats'>{' &middot; '.join(bits)} &middot; "
        "<a href='https://rbi.org.in'>rbi.org.in</a> &middot; not legal advice</div>",
        unsafe_allow_html=True)


def main() -> None:
    llm.warm_model_registry()  # parallel /models fetch, cached 10 min
    _inject_css()
    ui = sidebar_filters()
    rbi_updates_panel()
    eng, docs, _version = engine_and_docs()

    render_header(docs, store.read_manifest() or {})

    tab_ask, tab_circ, tab_cmp, tab_audit = st.tabs(
        ["Ask", "Circulars", "Compare", "Audit"])
    with tab_ask:
        ask_tab(eng, docs, ui)
    with tab_circ:
        circulars_tab(docs)
    with tab_cmp:
        compare_tab(docs)
    with tab_audit:
        audit_tab()


if __name__ == "__main__":
    main()
