"""Agent tool: hybrid retrieval over the MedRAG/textbooks corpus (Xiong et al.,
2024, arXiv:2402.13178). Use for general medical knowledge questions
("symptoms of seasonal allergies", "what does an elevated ALT mean").

For patient-specific questions, the agent should use search_clinical_documents
instead.
"""
from __future__ import annotations

from functools import lru_cache

from health_assistant.config import paths
from health_assistant.rag.retriever import HybridRetriever


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever.load(persist_dir=paths.medical_faiss_dir)


def search_medical_knowledge(query: str, k: int = 5) -> list[dict]:
    """Return top-k hybrid-retrieval hits from MedRAG/textbooks.

    Each hit: {text, metadata: {medrag_id, title}, retriever: "dense"|"sparse"|"both"}.
    The agent should cite the textbook title when quoting from a hit
    (e.g. "Harrison's Internal Medicine - Chapter on Diabetes").
    """
    return _retriever().search(query, k=k)
