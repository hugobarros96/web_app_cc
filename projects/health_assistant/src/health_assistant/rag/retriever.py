"""Hybrid retriever - dense FAISS (MiniLM embeddings) + sparse BM25, fused via
weighted Reciprocal Rank Fusion (RRF) using `EnsembleRetriever`.

Why hybrid? Dense embeddings catch semantic matches ("heart attack" ≈
"myocardial infarction"); sparse BM25 catches exact-term matches (drug names,
ICD codes, acronyms like COPD/ALT). On medical text, both signals matter.

Usage:
    hr = HybridRetriever.load(persist_dir=paths.clinical_faiss_dir)
    hits = hr.search("metformin diabetes elderly", k=5)
    # → list of {"text", "metadata", "score", "retriever": "dense"|"sparse"|"both"}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from health_assistant.config import settings

# Per-retriever shortlist size before fusion. Both retrievers return their top-K;
# the EnsembleRetriever then fuses and returns the top-K (passed to search()).
SHORTLIST_K = 20


@dataclass
class HybridRetriever:
    faiss: FAISS
    bm25: BM25Retriever
    ensemble: EnsembleRetriever

    @classmethod
    def build_from_documents(
        cls,
        docs: list[Document],
        persist_dir: Path,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> "HybridRetriever":
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        faiss = FAISS.from_documents(docs, embeddings)
        faiss.save_local(str(persist_dir))
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = SHORTLIST_K
        dense = faiss.as_retriever(search_kwargs={"k": SHORTLIST_K})
        ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=list(weights))
        return cls(faiss=faiss, bm25=bm25, ensemble=ensemble)

    @classmethod
    def load(
        cls,
        persist_dir: Path,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> "HybridRetriever":
        persist_dir = Path(persist_dir)
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        faiss = FAISS.load_local(
            str(persist_dir), embeddings, allow_dangerous_deserialization=True
        )
        docs = list(faiss.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = SHORTLIST_K
        dense = faiss.as_retriever(search_kwargs={"k": SHORTLIST_K})
        ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=list(weights))
        return cls(faiss=faiss, bm25=bm25, ensemble=ensemble)

    @staticmethod
    def _key(d: Document) -> str:
        """Stable identity for a Document across retrievers (so we can dedupe and
        tag which retriever surfaced each hit)."""
        meta = d.metadata
        if "source_file" in meta and "chunk_id" in meta:
            return f"{meta['source_file']}#{meta['chunk_id']}"
        if "medrag_id" in meta and meta["medrag_id"]:
            return f"medrag:{meta['medrag_id']}"
        return f"text:{d.page_content[:64]}"

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Run both retrievers, tag origin, RRF-fuse to a shortlist of SHORTLIST_K,
        then rerank with the cross-encoder and return the top-k."""
        # Import inside method so the ~280 MB reranker only loads when a retriever
        # is actually used (not at module import time).
        from health_assistant.rag.reranker import get_reranker

        dense_keys = {self._key(d) for d in self.faiss.similarity_search(query, k=SHORTLIST_K)}
        sparse_keys = {self._key(d) for d in self.bm25.invoke(query)}
        fused = self.ensemble.invoke(query)  # full SHORTLIST_K, no slice

        out: list[dict] = []
        for d in fused:
            k_ = self._key(d)
            if k_ in dense_keys and k_ in sparse_keys:
                origin = "both"
            elif k_ in dense_keys:
                origin = "dense"
            else:
                origin = "sparse"
            out.append(
                {
                    "text": d.page_content,
                    "metadata": d.metadata,
                    "retriever": origin,
                }
            )
        return get_reranker().rerank(query, out, top_k=k)
