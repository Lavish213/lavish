"""
Lavish Core — VisionReader (Wall Street Edition)
================================================
Dual/tri OCR (EasyOCR + PaddleOCR + Tesseract fallback) with confidence ensemble.
Parses BUY/SELL + ticker + optional price hints, dedupes, and triggers trades.

Folders (auto-created):
- lavish_core/vision/raw           ← inbound images
- lavish_core/vision/images_out    ← parsed JSON + debug
- lavish_core/vision/logs          ← logs

Env (optional):
VISION_MIN_CONF=0.55
VISION_MAX_SYMBOLS=5
VISION_SKIP_DUP_MIN=600    # seconds to ignore same file hash
VISION_ENABLE_TESSERACT=true|false
VISION_ENABLE_EASYOCR=true|false
VISION_ENABLE_PADDLE=true|false
"""

from __future__ import annotations
import os, io, re, json, time, hashlib
from pathlib import Path

WATCH = Path(__file__).resolve().parents[1] / "vision" / "inbox"
WATCH.mkdir(parents=True, exist_ok=True)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Optional OCR engines
EASYOCR_OK = False
PADDLE_OK  = False
TESSER_OK  = False

try:
    import easyocr
    EASYOCR_OK = True
except Exception:
    EASYOCR_OK = False

try:
    from paddleocr import PaddleOCR
    PADDLE_OK = True
except Exception:
    PADDLE_OK = False

try:
    import pytesseract
    TESSER_OK = True
except Exception:
    TESSER_OK = False

# Lavish modules
from lavish_core.logger_setup import get_logger
from lavish_core.trading.trade_handler import execute_trade_from_post

ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "vision"
RAW_DIR    = VISION_DIR / "raw"
OUT_DIR    = VISION_DIR / "images_out"
LOG_DIR    = VISION_DIR / "logs"
for p in [RAW_DIR, OUT_DIR, LOG_DIR]: p.mkdir(parents=True, exist_ok=True)

log = get_logger("vision.reader", log_dir=str(LOG_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# Config via env
# ──────────────────────────────────────────────────────────────────────────────
def _env_bool(k: str, default: bool) -> bool:
    v = os.getenv(k, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")

MIN_CONF     = float(os.getenv("VISION_MIN_CONF", "0.55"))
MAX_SYMBOLS  = int(os.getenv("VISION_MAX_SYMBOLS", "5"))
SKIP_DUP_S   = int(os.getenv("VISION_SKIP_DUP_MIN", "600"))
USE_EASY     = _env_bool("VISION_ENABLE_EASYOCR", True)
USE_PADDLE   = _env_bool("VISION_ENABLE_PADDLE", True)
USE_TESSER   = _env_bool("VISION_ENABLE_TESSERACT", True)

# Regexes
SYM_RE   = re.compile(r"\b([A-Z]{1,5})\b")
ACT_RE   = re.compile(r"\b(BUY|SELL|SHORT|CALLS?|PUTS?)\b", re.I)
PRICE_RE = re.compile(r"\$?\s*(\d{1,4}(?:\.\d{1,2})?)")

STOPWORDS = {"THE","AND","FOR","WITH","THIS","THAT","CALL","CALLS","PUT","PUTS","BUY","SELL"}

@dataclass
class OCRPiece:
    text: str
    conf: float   # normalized 0..1

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def _read_image_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except Exception as e:
        log.csv_line("read_error", f"{path.name}:{e}", level="ERROR")
        return None

def _pil_load(b: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# OCR engines
# ──────────────────────────────────────────────────────────────────────────────
class _EasyRunner:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)  # CPU default (safe)
    def run(self, img: Image.Image) -> List[OCRPiece]:
        res = self.reader.readtext(np.asarray(img), detail=1)  # (bbox, text, conf)
        out = []
        for _bbox, txt, cf in res:
            if not txt: continue
            out.append(OCRPiece(txt, float(max(0.0, min(1.0, cf)))))
        return out

class _PaddleRunner:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    def run(self, img: Image.Image) -> List[OCRPiece]:
        import numpy as np
        arr = np.asarray(img)
        res = self.ocr.ocr(arr, cls=True)
        out: List[OCRPiece] = []
        if not res:
            return out
        for line in res[0]:
            txt = line[1][0]; cf = line[1][1]
            if txt:
                out.append(OCRPiece(txt, float(max(0.0, min(1.0, cf)))))
        return out

class _TesserRunner:
    def run(self, img: Image.Image) -> List[OCRPiece]:
        try:
            txt = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            if not txt.strip():
                return []
            # No confidence by line; assign a soft fixed conf
            return [OCRPiece(txt, 0.60)]
        except Exception:
            return []

# Safe import for numpy (lazy)
def _ensure_np():
    global np
    try:
        import numpy as np  # noqa
    except Exception as e:
        raise RuntimeError(f"Numpy is required for EasyOCR/PaddleOCR: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble & parse
# ──────────────────────────────────────────────────────────────────────────────
def _ensemble_text(pieces: List[OCRPiece]) -> Tuple[str, float]:
    """
    Combine OCR fragments into one string and overall confidence (mean).
    """
    if not pieces:
        return "", 0.0
    text = " ".join(p.text.strip() for p in pieces if p.text.strip())
    conf = sum(p.conf for p in pieces) / max(1, len(pieces))
    return text, float(max(0.0, min(1.0, conf)))

def _parse_intent(text: str) -> Dict[str, Any]:
    # direction
    actm = ACT_RE.search(text or "")
    action = "BUY"
    if actm:
        w = actm.group(1).upper()
        action = "SELL" if w in ("SELL","SHORT") or "PUT" in w else "BUY"

    # symbols
    cands = {m.group(1).upper() for m in SYM_RE.finditer(text or "")}
    symbols = [c for c in cands if c not in STOPWORDS][:MAX_SYMBOLS]

    # price hint
    price: Optional[float] = None
    for m in PRICE_RE.finditer(text or ""):
        try:
            v = float(m.group(1).replace(",", ""))
            if 0.05 <= v <= 10_000:
                price = v
        except Exception:
            pass

    base_conf = 0.55 + (0.25 if actm else 0.0) + (0.10 if symbols else 0.0)
    return {
        "symbols": symbols,
        "action": action,
        "confidence": round(min(0.99, base_conf), 2),
        "ref_price": price
    }

# ──────────────────────────────────────────────────────────────────────────────
# Vision Reader (public)
# ──────────────────────────────────────────────────────────────────────────────
class VisionReader:
    def __init__(self):
        # Initialize available engines lazily
        self.easy = _EasyRunner() if (USE_EASY and EASYOCR_OK) else None
        self.padl = _PaddleRunner() if (USE_PADDLE and PADDLE_OK) else None
        self.tess = _TesserRunner() if (USE_TESSER and TESSER_OK) else None

        # dedupe state (file-hash -> ts)
        self._seen: Dict[str, float] = {}
        log.csv_line("init", f"engines: easy={bool(self.easy)} paddle={bool(self.padl)} tesser={bool(self.tess)}")

    def _should_skip(self, fid: str) -> bool:
        ts = self._seen.get(fid, 0.0)
        if time.time() - ts < SKIP_DUP_S:
            return True
        self._seen[fid] = time.time()
        return False

    def process_image_file(self, path: Path) -> Optional[Dict[str, Any]]:
        b = _read_image_bytes(path)
        if not b: return None
        fid = _sha1_bytes(b)
        if self._should_skip(fid):
            log.csv_line("dup_skip", path.name)
            return None

        img = _pil_load(b)
        if img is None:
            log.csv_line("bad_image", path.name, level="ERROR")
            return None

        # gather pieces
        pieces: List[OCRPiece] = []
        if self.easy:
            _ensure_np()
            try:
                pieces.extend(self.easy.run(img))
            except Exception as e:
                log.csv_line("easyocr_error", f"{path.name}:{e}", level="ERROR")
        if self.padl:
            _ensure_np()
            try:
                pieces.extend(self.padl.run(img))
            except Exception as e:
                log.csv_line("paddle_error", f"{path.name}:{e}", level="ERROR")
        if self.tess:
            try:
                pieces.extend(self.tess.run(img))
            except Exception as e:
                log.csv_line("tesser_error", f"{path.name}:{e}", level="ERROR")

        text, ocr_conf = _ensemble_text(pieces)
        intent = _parse_intent(text)
        intent_conf = max(ocr_conf, intent["confidence"])

        payload = {
            "file": path.name,
            "hash": fid,
            "ocr_conf": round(ocr_conf, 3),
            "text_excerpt": (text or "")[:1200],
            "intent": intent,
            "intent_conf": round(intent_conf, 3),
            "ts": int(time.time())
        }

        # persist JSON
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        # trigger trades if confident enough
        if intent_conf >= MIN_CONF and intent["symbols"]:
            for sym in intent["symbols"]:
                execute_trade_from_post({
                    "source": "vision",
                    "action": intent["action"],
                    "symbol": sym,
                    "confidence": float(intent_conf),
                    "amount_usd": None,
                    "note": f"vision:{path.name}"
                })

        return payload