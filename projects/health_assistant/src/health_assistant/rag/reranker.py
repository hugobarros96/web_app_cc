"""Cross-encoder reranker for hybrid retrieval.

Uses BAAI/bge-reranker-base (Xiao et al. 2023, arXiv:2309.07597).
Loaded once per process via the get_reranker() lru_cache helper.
"""
from __future__ import annotations

from functools import lru_cache

import mlflow


class Reranker:
    """Wraps a sentence-transformers CrossEncoder to rerank hybrid retrieval hits."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    @mlflow.trace(name="reranker.rerank")
    def rerank(self, query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
        """Score each (query, hit['text']) pair, attach rerank_score, return top_k by score."""
        if not hits:
            return []
        pairs = [(query, h["text"]) for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        return sorted(hits, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Process-cached reranker. First call downloads the model (~280 MB) into
    the local HuggingFace cache; subsequent calls are instant."""
    return Reranker()
