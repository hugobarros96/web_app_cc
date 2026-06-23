"""Download MedRAG/textbooks (Xiong et al., 2024 arXiv:2402.13178) → wrap as Documents
→ embed → save FAISS index.

Output:
  artifacts/medical_faiss_index/
    ├── index.faiss
    └── index.pkl

The MedRAG team published the corpus already chunked at paragraph level
(~125k chunks), tuned for medical RAG and evaluated on the MIRAGE benchmark.
We do NOT re-chunk - we use the publisher's chunks as-is and just embed.

Run:
  python -m health_assistant.rag.ingest_medrag                 # full ~125k chunks (~30-45 min CPU)
  python -m health_assistant.rag.ingest_medrag --limit 1000    # quick smoke (~1-2 min)
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from health_assistant.config import paths, settings
from health_assistant.rag.chunking import medrag_row_to_document

CORPUS = "MedRAG/textbooks"


def build_medical_index(limit: int | None = None) -> int:
    paths.medical_faiss_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {CORPUS} from HuggingFace ...")
    ds = load_dataset(CORPUS, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Loaded {len(ds)} rows.")

    docs = [medrag_row_to_document(row) for row in ds]
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    print(f"Embedding {len(docs)} chunks with {settings.embedding_model} (this is the slow step) ...")
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(paths.medical_faiss_dir))
    print(f"Saved FAISS index to {paths.medical_faiss_dir}")
    return len(docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap chunks (for quick smoke)")
    args = ap.parse_args()
    n = build_medical_index(limit=args.limit)
    print(f"Indexed {n} medical chunks.")
