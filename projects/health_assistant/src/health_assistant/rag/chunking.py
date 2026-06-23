from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2")]
MAX_TOKENS_PER_CHUNK = 800
OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4  # rough approximation for the safety-net splitter

_md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
_safety_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN,
    chunk_overlap=OVERLAP_TOKENS * CHARS_PER_TOKEN,
    separators=["\n\n", "\n", " ", ""],
)


def chunk_clinical_brief(md_text: str, source_file: str) -> list[Document]:
    """Two-step chunking: split markdown on H2 sections, then safety-net split anything oversized.

    Each output Document carries metadata: {source_file, chunk_id, headings: [H1, H2]}.
    """
    sections = _md_splitter.split_text(md_text)
    out: list[Document] = []
    for sec_idx, sec in enumerate(sections):
        headings = [v for k, v in sec.metadata.items() if k in {"h1", "h2"}]
        if len(sec.page_content) > MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN:
            pieces = _safety_splitter.split_text(sec.page_content)
        else:
            pieces = [sec.page_content]
        for piece_idx, text in enumerate(pieces):
            out.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": source_file,
                        "chunk_id": f"{sec_idx}-{piece_idx}",
                        "headings": headings,
                    },
                )
            )
    return out


def medrag_row_to_document(row: dict) -> Document:
    """Wrap a pre-chunked MedRAG row as a LangChain Document (no re-chunking)."""
    return Document(
        page_content=row["content"],
        metadata={
            "medrag_id": row.get("id", ""),
            "title": row.get("title", ""),
        },
    )
