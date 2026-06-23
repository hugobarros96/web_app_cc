"""Build the clinical-briefs FAISS index from `data/documents_data/markdowns/`.

Output:
  artifacts/clinical_faiss_index/
    ├── index.faiss   ← binary vector index
    └── index.pkl     ← docstore (chunks + metadata)

Run:
  python -m health_assistant.rag.ingest_clinical_docs
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from health_assistant.config import paths, settings
from health_assistant.rag.chunking import chunk_clinical_brief


def build_clinical_index() -> int:
    """Read all clinical-brief markdowns, chunk them, embed, save FAISS. Returns chunk count."""
    paths.clinical_faiss_dir.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        p for p in paths.clinical_docs_dir.glob("medical_document_*.md")
        if "Zone.Identifier" not in p.name
    )
    print(f"Found {len(md_files)} clinical briefs.")

    docs = []
    for p in md_files:
        text = p.read_text(encoding="utf-8")
        docs.extend(chunk_clinical_brief(text, source_file=p.name))
    print(f"Chunked into {len(docs)} sections.")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    print(f"Embedding with {settings.embedding_model} ...")
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(paths.clinical_faiss_dir))
    print(f"Saved FAISS index to {paths.clinical_faiss_dir}")
    return len(docs)


if __name__ == "__main__":
    n = build_clinical_index()
    print(f"Indexed {n} clinical chunks.")
