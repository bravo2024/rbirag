"""Document classification: entities, doc type, themes, content class.

Entity tagging is title-first: RBI titles carry the addressee in parentheses
(e.g. "Reserve Bank of India (Non-Banking Financial Companies - KYC)
Directions, 2025"), which is authoritative. Only when the title has no entity
do we scan the head of the text ("To, All Commercial Banks...").

Canonical entities:
  commercial_banks, authorised_dealers, nbfc, hfc, mgc, cic, spd, nofhc,
  payments_banks, small_finance_banks, cooperative_banks, regional_rural_banks,
  local_area_banks, aifi, arc, credit_information_companies

`entity_scope == "broad"` marks cross-cutting documents (no specific
addressee) — they apply to every institution filter.
"""
from __future__ import annotations

import datetime as dt
import re

from regintel import util

# (pattern, canonical, label) — order matters: first match wins per pattern.
ENTITY_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"Non-?Banking Financial Compan|NBFCS?\b", re.I), "nbfc", "NBFCs"),
    (re.compile(r"Housing Finance Compan|HFCs?\b", re.I), "hfc", "Housing Finance Companies"),
    (re.compile(r"Mortgage Guarantee Compan", re.I), "mgc", "Mortgage Guarantee Companies"),
    (re.compile(r"Core Investment Compan", re.I), "cic", "Core Investment Companies"),
    (re.compile(r"Standalone Primary Dealers?", re.I), "spd", "Standalone Primary Dealers"),
    (re.compile(r"Non-?Operative Financial Holding Compan|NOFHC", re.I), "nofhc", "NOFHCs"),
    (re.compile(r"Small Finance Banks?", re.I), "small_finance_banks", "Small Finance Banks"),
    (re.compile(r"Payments? Banks?", re.I), "payments_banks", "Payments Banks"),
    (re.compile(r"Prepaid Payment Instruments?|PPIs?\b", re.I), "ppi_issuers", "PPI Issuers"),
    (re.compile(r"Regional Rural Banks?|RRBs?\b", re.I), "regional_rural_banks", "Regional Rural Banks"),
    (re.compile(r"Local Area Banks?", re.I), "local_area_banks", "Local Area Banks"),
    (re.compile(r"(?:Urban |Rural )?Co-?operative Banks?", re.I), "cooperative_banks", "Co-operative Banks"),
    (re.compile(r"All India Financial Institutions?|AIFIs?\b", re.I), "aifi", "All India Financial Institutions"),
    (re.compile(r"Assets? Reconstruction Compan|ARCs?\b", re.I), "arc", "Asset Reconstruction Companies"),
    (re.compile(r"Credit Information Compan", re.I), "credit_information_companies", "Credit Information Companies"),
    (re.compile(r"Authorised Dealers?|Authorized Dealers?", re.I), "authorised_dealers", "Authorised Dealers"),
    (re.compile(r"(?:Scheduled |Foreign |All )?Commercial Banks?", re.I), "commercial_banks", "Commercial Banks"),
]
ENTITY_LABELS = {canon: label for _, canon, label in ENTITY_RULES}

# Title-side aliases for the same canon.
BROAD_TITLE_RE = re.compile(
    r"All Regulated Entities|All Entities|Regulated Entities|All REs\b|"
    r"Banking and Non-?Banking|Banks and NBFCs|Financial Sector", re.I)
HEAD_ADD_RESS = re.compile(r"\b(To|Addressed to)[^.]{0,80}$", re.I)

NOISE_TITLE_RE = re.compile(
    r"Auction|Treasury Bills|Sovereign Gold|Ways and Means|"
    r"Market Operations|Underwriting|Liquidity Adjustment Facility|"
    r"Standalone Primary Dealers for|Conversion of|Exchange Rate.*(?:Today|Reference)", re.I)

DOC_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Amendment", re.I), "amendment_direction"),
    (re.compile(r"Master Directions?\b|\(.*?\)\s*Directions?\b", re.I), "master_direction"),
    (re.compile(r"Scheme\s+\d{4}|Scheme\b", re.I), "scheme"),
    (re.compile(r"\bCircular\b", re.I), "circular"),
]

# (pattern -> theme label)
THEME_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Know Your Customer|\bKYC\b|\bAML\b|Anti-?Money Laundering|Combating.*Terrorist|CFT\b", re.I), "KYC / AML"),
    (re.compile(r"Fraud", re.I), "Fraud Risk"),
    (re.compile(r"Digital Lending|Fintech|Digital Banking|Digital Payments Security", re.I), "Digital Lending / Fintech"),
    (re.compile(r"Credit Card|Debit Card|Cards?\b", re.I), "Cards"),
    (re.compile(r"Deposits?\b|Savings|Current Account|Interest Rate on (?:Deposits?|Advances?)", re.I), "Deposits"),
    (re.compile(r"Capital Adequacy|Basel|Market Risk|Capital Charge|Credit Risk|Provisioning|Asset Classification|Income Recognition|Prudential N", re.I), "Prudential Norms"),
    (re.compile(r"Liquidity|\bLCR\b|\bNSFR\b|Asset[- ]Liability", re.I), "Liquidity / ALM"),
    (re.compile(r"Grievance|Ombudsman|Nodal Officer|Customer (?:Protection|Service|Care)|Fair Practices|Conduct", re.I), "Customer Protection / Grievance"),
    (re.compile(r"Cyber|Information (?:Security|Technology)|IT (?:Governance|Framework)|Outsourcing of IT", re.I), "Cyber / IT Risk"),
    (re.compile(r"Outsourcing", re.I), "Outsourcing"),
    (re.compile(r"Foreign Exchange|FEMA|Authorised Dealer|External Commercial|Cross-?Border", re.I), "Forex / FEMA"),
    (re.compile(r"Priority Sector|Financial Inclusion|Microfinance|Self Help Group", re.I), "Financial Inclusion / Microfinance"),
    (re.compile(r"Climate", re.I), "Climate Risk"),
    (re.compile(r"Governance|Fit and Proper|Compensation|Directors?", re.I), "Governance"),
    (re.compile(r"Recovery|Repossession|Non-?Performing|NPA|Resolution|Insolvency", re.I), "Recovery / NPA"),
    (re.compile(r"Payments?\b|PPI|Prepaid|UPI|Settlement", re.I), "Payments"),
    (re.compile(r"Reporting|Returns?|Supervisory", re.I), "Reporting / Supervision"),
]
THEME_LABELS = [label for _, label in THEME_RULES]


def classify_entities(title: str, text_head: str) -> tuple[list[str], list[list[str]], str]:
    """Returns (entities, evidence, scope). Title matches are authoritative."""
    entities: list[str] = []
    evidence: list[list[str]] = []

    def add(canon: str, snippet: str, source: str) -> None:
        if canon not in entities:
            entities.append(canon)
        evidence.append([canon, snippet.strip()[:80], source])

    for rx, canon, _label in ENTITY_RULES:
        m = rx.search(title or "")
        if m:
            add(canon, m.group(0), "title")
    if entities:
        return entities, evidence, "specific"

    head = (text_head or "")[:1500]
    for rx, canon, _label in ENTITY_RULES:
        m = rx.search(head)
        if m:
            add(canon, m.group(0), "text")
    if entities:
        return entities, evidence, "specific"

    if BROAD_TITLE_RE.search(title or ""):
        evidence.append(["broad", BROAD_TITLE_RE.search(title).group(0), "title"])
        return [], evidence, "broad"
    return [], evidence, "broad"


def classify_doc_type(title: str, reference: str | None) -> str:
    for rx, dtype in DOC_TYPE_RULES:
        if rx.search(title or ""):
            return dtype
    if reference and re.match(r"RBI/\d{4}-\d{2,4}/\d+", reference):
        return "notification"
    return "other"


def classify_themes(title: str, text_head: str) -> list[str]:
    hay = f"{title} {text_head[:2500]}"
    themes = [label for rx, label in THEME_RULES if rx.search(hay)]
    return themes[:4]


def content_class(title: str) -> str:
    return "market_ops" if NOISE_TITLE_RE.search(title or "") else "regulation"


# Whole-document commencement: "These Directions shall come into (force|effect)
# ... from <date>" or "... shall be applicable/effective ... from <date>".
# Sentence-anchored: the sentence must open with the Directions/instructions
# subject, so paragraph- or section-level effective dates ("Section B of the
# Directions ... shall come into effect from ...") are ignored.
MONTH_DATE_RE = (r"(?:January|February|March|April|May|June|July|August|September|"
                 r"October|November|December)\s+\d{1,2},\s+\d{4}")
_COMMENCEMENT_RE = re.compile(
    r"^(?:These|This|The|All)\s+(?:Master\s+|Amendment\s+)?"
    r"(?:Directions?|instructions?|provisions?)"
    r"[^.]{0,80}?(?:come\s+into\s+(?:force|effect)|(?:shall\s+)?be\s+(?:applicable|effective))"
    r"[^.]{0,60}?(?:from|on)\s+(?:the\s+)?(?:\d{1,2}\s+day\s+of\s+)?(" + MONTH_DATE_RE + r")",
    re.I)
_ENUM_RE = re.compile(r"^\s*(?:\(\d+[a-z]?\)|\([ivxlc]+\)|\[\d+\]|\d+[.)])\s+", re.I)


def effective_date(text: str) -> str | None:
    """Latest whole-document commencement date in the text (ISO or None)."""
    best = None
    for sentence in util.sentence_split(text or ""):
        m = _COMMENCEMENT_RE.match(_ENUM_RE.sub("", sentence.strip()))
        if m:
            d = util.parse_date(m.group(1))
            if d and (best is None or d > best):
                best = d
    return best


def status_for(title: str, effective_from: str | None, today: str | None = None) -> str:
    """Document status: withdrawn > forthcoming > current.

    "Forthcoming" when the title says so (unless it is an amendment notice,
    which acts immediately) or when the document's own commencement date is in
    the future.
    """
    t = (title or "").lower()
    today = today or dt.date.today().isoformat()
    if t.startswith("withdrawn"):
        return "withdrawn"
    if "forthcoming" in t and "amendment" not in t:
        return "forthcoming"
    if effective_from and effective_from > today:
        return "forthcoming"
    return "current"


def effective_status(status: str | None, title: str, effective_from: str | None,
                     today: str | None = None) -> str:
    """The status as of ``today``, derived from what is stored on the document.

    The status written at ingest / recompute time goes stale: a forthcoming
    document becomes current on its effective date without anyone re-running a
    recompute. Withdrawn / superseded are lifecycle facts, not date-derived, and
    are kept as stored; everything else is re-derived with ``status_for``.
    """
    status = status or "current"
    if status in ("withdrawn", "superseded"):
        return status
    return status_for(title, effective_from, today)


def classify_document(title: str, text_head: str, reference: str | None) -> dict:
    entities, evidence, scope = classify_entities(title, text_head)
    return {
        "entities": entities,
        "entity_scope": scope,
        "entity_evidence": evidence,
        "doc_type": classify_doc_type(title, reference),
        "themes": classify_themes(title, text_head),
        "content_class": content_class(title),
    }
