"""Cross-encoder reranker (CPU), loaded lazily and shared per process."""
from __future__ import annotations

from regintel import config


class Reranker:
    def __init__(self, model: str | None = None, max_length: int | None = None):
        self.model_name = model or config.RERANK_MODEL
        self.max_length = max_length or config.RERANK_MAX_LENGTH
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import CrossEncoder

            try:
                # cached models load without hub round-trips (saves 5-45 s per load)
                self._m = CrossEncoder(self.model_name, max_length=self.max_length,
                                       local_files_only=True)
            except Exception:  # not cached yet, or older ST without the flag
                self._m = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._m

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        out = self.model.predict([(query, t) for t in texts],
                                 batch_size=batch_size, show_progress_bar=False)
        return [float(x) for x in out]
