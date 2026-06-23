"""One-shot orchestrator: build the two FAISS indices.

The trained models (COPD + ALT) ship committed under `artifacts/models/`, so
bootstrap does NOT train them - it only builds the RAG indices, which are too
large to commit. Idempotent: skips a step if its output already exists; use
`--force` to rebuild, `--skip-medrag` to skip the slow MedRAG ingestion.

Typical first-run flow inside Docker:
    python -m health_assistant.scripts.bootstrap

Quick local iteration (no MedRAG textbook index):
    python -m health_assistant.scripts.bootstrap --skip-medrag

To (re)train the models - not part of bootstrap - run the training modules
directly:
    python -m health_assistant.models.train_copd --n-trials 30
    python -m health_assistant.models.train_alt  --n-trials 30
"""
from __future__ import annotations

import argparse
import time

from health_assistant.config import paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap the RAG FAISS indices.")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild the indices even if outputs already exist.")
    ap.add_argument("--skip-medrag", action="store_true",
                    help="Skip the slow MedRAG/textbooks ingestion.")
    args = ap.parse_args()

    overall_start = time.time()

    # 1. Clinical FAISS index ------------------------------------------------
    clinical_index = paths.clinical_faiss_dir / "index.faiss"
    if args.force or not clinical_index.exists():
        from health_assistant.rag.ingest_clinical_docs import build_clinical_index
        print(">> [1/2] Building clinical FAISS index ...")
        t0 = time.time()
        n = build_clinical_index()
        print(f">> [1/2] Done: {n} chunks in {time.time() - t0:.1f}s")
    else:
        print(">> [1/2] Clinical index already exists, skipping.")

    # 2. MedRAG FAISS index --------------------------------------------------
    medical_index = paths.medical_faiss_dir / "index.faiss"
    if args.skip_medrag:
        print(">> [2/2] Skipping MedRAG/textbooks ingestion per --skip-medrag.")
    elif args.force or not medical_index.exists():
        from health_assistant.rag.ingest_medrag import build_medical_index
        print(">> [2/2] Building MedRAG FAISS index (slow: ~5-10 min) ...")
        t0 = time.time()
        n = build_medical_index()
        print(f">> [2/2] Done: {n} chunks in {time.time() - t0:.1f}s")
    else:
        print(">> [2/2] Medical index already exists, skipping.")

    print(f">> Bootstrap complete in {time.time() - overall_start:.1f}s total.")
    print(">> Models ship pre-trained under artifacts/models/. To retrain, run "
          "`python -m health_assistant.models.train_copd` / `train_alt`.")


if __name__ == "__main__":
    main()
