"""Agent tool: hybrid retrieval over the 1,050 clinical-brief markdowns.

Use for patient-specific questions ("what meds was the heart attack patient
on?", "summarize the treatment plan for diabetic patients"). For general
medical knowledge, the agent should use `search_medical_knowledge` instead.
"""
from __future__ import annotations

from functools import lru_cache

from health_assistant.config import paths
from health_assistant.rag.retriever import HybridRetriever


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever.load(persist_dir=paths.clinical_faiss_dir)


def search_clinical_documents(query: str, k: int = 5) -> list[dict]:
    """Return top-k hybrid-retrieval hits from the clinical-brief corpus.

    Each hit: {text, metadata: {source_file, chunk_id, headings}, retriever: "dense"|"sparse"|"both"}.
    The agent should cite source_file + headings when quoting from a hit.
    """
    return _retriever().search(query, k=k)
