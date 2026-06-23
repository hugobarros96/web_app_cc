"""Attachment dataclass used by run() to thread PDFs / images through the agent.

The payload shape differs by kind:
- pdf:   {"text": str, "page_count": int, "truncated": bool, "scanned_warning": bool}
- image: {"format": str, "bytes": bytes, "width": int, "height": int}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Attachment:
    kind: Literal["pdf", "image"]
    name: str
    payload: dict
