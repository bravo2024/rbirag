"""Central paths, models, retrieval knobs and institution mapping.

Everything else imports from here. Values can be overridden with environment
variables where noted; a project-root `.env` file is loaded on import.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, does not override real env vars."""
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

# Base data dir; override for sandboxed runs (the Kaggle GPU kernel writes to
# a writable copy of the corpus: REGINTEL_DATA_DIR=/kaggle/working/data).
DATA = Path(os.environ.get("REGINTEL_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
PARSED = DATA / "parsed"
INDEX_DIR = DATA / "index"
STATE = DATA / "state"
CACHE_DIR = RAW / "_cache"

DOCUMENTS = DATA / "documents.jsonl"
CHUNKS = DATA / "chunks.jsonl"
MANIFEST = DATA / "manifest.json"
SEEN = STATE / "seen.json"
SYNC_LOG = STATE / "sync_log.jsonl"
AUDIT_LOG = STATE / "audit.jsonl"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
RSS_URL = "https://www.rbi.org.in/notifications_rss.xml"

# RBI's own Master Directions listings — the complete current catalog.
# 19 categories served from 30 listing pages: 8 cross-cutting, 11 Department of
# Regulation (entity-wise consolidated directions) and 11 Department of
# Supervision (the 2026 supervisory directions: audit, fraud, cyber, returns).
_LISTING_BASE = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx?did="
LISTING_URLS = {
    # cross-cutting
    "banker_debt_manager": _LISTING_BASE + "334",
    "banker_to_govt": _LISTING_BASE + "386",
    "consumer_protection": _LISTING_BASE + "338",
    "financial_inclusion": _LISTING_BASE + "343",
    "financial_market": _LISTING_BASE + "336",
    "fema": _LISTING_BASE + "335",
    "issuer_of_currency": _LISTING_BASE + "339",
    "payment_systems": _LISTING_BASE + "344",
    # Department of Regulation — entity-wise directions
    "commercial_banks": _LISTING_BASE + "403",
    "small_finance_banks": _LISTING_BASE + "404",
    "payments_banks": _LISTING_BASE + "405",
    "local_area_banks": _LISTING_BASE + "406",
    "regional_rural_banks": _LISTING_BASE + "407",
    "urban_cooperative_banks": _LISTING_BASE + "408",
    "rural_cooperative_banks": _LISTING_BASE + "409",
    "aifi": _LISTING_BASE + "410",
    "nbfc": _LISTING_BASE + "411",
    "arc": _LISTING_BASE + "412",
    "cic": _LISTING_BASE + "413",
    # Department of Supervision — supervisory directions (2026 wave)
    "commercial_banks_sup": _LISTING_BASE + "419",
    "small_finance_banks_sup": _LISTING_BASE + "420",
    "payments_banks_sup": _LISTING_BASE + "421",
    "local_area_banks_sup": _LISTING_BASE + "422",
    "regional_rural_banks_sup": _LISTING_BASE + "423",
    "urban_cooperative_banks_sup": _LISTING_BASE + "424",
    "rural_cooperative_banks_sup": _LISTING_BASE + "425",
    "aifi_sup": _LISTING_BASE + "426",
    "nbfc_sup": _LISTING_BASE + "427",
    "arc_sup": _LISTING_BASE + "428",
    "cic_sup": _LISTING_BASE + "429",
}
VIEW_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id={pid}"

FETCH_THROTTLE_S = float(os.environ.get("REGINTEL_THROTTLE", "1.2"))
FETCH_RETRIES = 4

# ---------------------------------------------------------------------------
# Retrieval models (CPU friendly)
# ---------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("REGINTEL_EMBED", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Per-model inference overrides (same idea as TAXRAGENCHANCE's SEQ_LEN_OVERRIDE):
# keep each model inside its trained envelope. hinglish-st-33m was trained at 192
# tokens and its own evals encode queries plain — no bge instruction prefix.
EMBED_MAX_SEQ_OVERRIDES = {"vivekkopthsd/hinglish-st-33m": 192}
EMBED_QUERY_PREFIX_OVERRIDES = {"vivekkopthsd/hinglish-st-33m": ""}
RERANK_MODEL = os.environ.get("REGINTEL_RERANK", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# None = auto (CUDA when available); set REGINTEL_DEVICE=cpu to force CPU.
EMBED_DEVICE = os.environ.get("REGINTEL_DEVICE") or None

# ---------------------------------------------------------------------------
# Chunking / retrieval parameters
# ---------------------------------------------------------------------------
CHUNK_MIN_WORDS = 120
CHUNK_MAX_WORDS = 380
CHUNK_HARD_MAX_WORDS = 520

CANDIDATES = 200          # per arm, before fusion
RERANK_DEPTH = int(os.environ.get("REGINTEL_RERANK_DEPTH", "30"))            # pairs sent to the cross-encoder (30 = eval-verified: recall unchanged, ~1.5x faster)
RERANK_MAX_LENGTH = int(os.environ.get("REGINTEL_RERANK_MAX_LENGTH", "512"))  # token cap per rerank pair (384 measurably costs recall — keep 512)
TOP_K = 6                 # passages shown to the answer layer
RRF_K = 60
# Retrieval-level abstain cutoff on the cross-encoder logit (raw score, not a
# probability). Calibrated 2026-10-17: in-scope eval queries score >= 3.67,
# out-of-scope queries mostly <= 0.44 (one outlier at 4.0). Recalibrate with
# scripts/eval_abstention.py when the corpus or the reranker changes.
ABSTAIN_SCORE = float(os.environ.get("REGINTEL_ABSTAIN", "2.0"))

# ---------------------------------------------------------------------------
# Institutions served by this deployment.
# "entities" is the set of canonical entity tags a document must intersect
# (documents classified as broad / all-entities always pass).
# ---------------------------------------------------------------------------
INSTITUTIONS = {
    "all":  {"label": "All institutions", "entities": set()},
    "bank": {"label": "Commercial bank",
             "entities": {"commercial_banks", "authorised_dealers"}},
    "nbfc": {"label": "NBFC / HFC group",
             "entities": {"nbfc", "hfc"}},
}

DOC_TYPES = ["master_direction", "amendment_direction", "circular",
             "notification", "scheme", "other"]
