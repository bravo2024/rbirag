"""RBIRAG — RBI Regulatory Intelligence RAG.

Live RBI notifications (RSS) + Master Directions backfill -> layout-aware
extraction -> structure-aware chunking -> hybrid retrieval (BM25 + dense +
cross-encoder) with an institution filter (Bank vs NBFC), audit-trail
citations and a Streamlit UI.
"""

__version__ = "1.0.0"
