"""Digital-PDF text extraction via pypdf.

Scanned PDFs are flagged but not OCR'd (future work; would need Tesseract or
AWS Textract).
"""
from __future__ import annotations

import io

import pypdf


def extract_text(file_bytes: bytes, max_chars: int = 32_000) -> dict:
    """Extract text from a digital PDF.

    Returns:
        {"text": str, "page_count": int, "truncated": bool, "scanned_warning": bool}
    """
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    page_count = len(reader.pages)
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n".join(parts).strip()

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    # Heuristic: pages > 0 but barely any text -> probably a scanned PDF where
    # pypdf gave us nothing because the content is rasterized images.
    scanned_warning = page_count > 0 and len(text) < 100

    return {
        "text": text,
        "page_count": page_count,
        "truncated": truncated,
        "scanned_warning": scanned_warning,
    }
