from __future__ import annotations
import os
from typing import Dict, Any
from PIL import Image
import numpy as np
import pytesseract

# For macOS brew, tesseract lands in /opt/homebrew/bin/tesseract (Apple Silicon)
# If needed: pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def ocr_image(path: str, lang: str = "eng") -> Dict[str, Any]:
    """
    OCR for screenshots, statements, charts with numbers/labels.
    Returns plain text + rough per-line split.
    """
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img, lang=lang)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {"text": text.strip(), "lines": lines, "num_lines": len(lines)}
