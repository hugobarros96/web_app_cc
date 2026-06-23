"""Image preprocessing for the vision content block.

Resizes large images to a max dimension (preserving aspect) and normalizes
to PNG bytes ready to embed in a Strands content block.
"""
from __future__ import annotations

import io

from PIL import Image


def prepare_image(file_bytes: bytes, max_dim: int = 2048) -> dict:
    """Decode, optionally downscale, re-encode as PNG.

    Returns:
        {"format": "png", "bytes": bytes, "width": int, "height": int}
    """
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")  # normalize alpha + webp

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        w, h = img.size

    out_buf = io.BytesIO()
    img.save(out_buf, format="PNG")
    return {
        "format": "png",
        "bytes": out_buf.getvalue(),
        "width": w,
        "height": h,
    }
