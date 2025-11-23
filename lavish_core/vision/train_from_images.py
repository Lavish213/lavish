# lavish_core/vision/train_from_images.py
"""
Lavish Vision Trainer — ingest a folder of historical trade images, run OCR with
smart preprocessing + caching, parse into BUY/SELL/CALL/PUT intents, and export
a clean training dataset you can hand-correct and re-run for incremental learning.

Design goals:
- Local OCR only (no external APIs).
- Robust OCR pipeline with OpenCV preproc; local ocr.ocr_image; fallback to Paddle/Tesseract.
- Idempotent: caches per-image OCR outputs.
- Parser integration: lavish_core.vision.parser_triggers.parse_trade_text
- Safe: training/export only.
Outputs:
  • data/training_labels.csv
  • data/training_labels.jsonl
  • data/ocr_cache/<sha1>.json
  • data/vision_model.pkl          (optional light heuristic model)
"""

from __future__ import annotations

import os
import re
import cv2
import csv
import sys
import json
import time
import glob
import hashlib
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Stable repo paths (works from any CWD)
_THIS = Path(__file__).resolve()
_LAVISH_CORE = _THIS.parent.parent                 # .../lavish_core
_DATA_ROOT = Path(__file__).resolve().parents[2] / "lavish_core" / "data"
IMAGES_IN_DIR = _DATA_ROOT / "images_in"

IMAGES_IN_DIR        = (_DATA_ROOT / "images_in").resolve()
IMAGES_DONE_DIR      = (_DATA_ROOT / "images_done").resolve()
IMAGES_PROCESSED_DIR = (_DATA_ROOT / "images_processed").resolve()
OCR_CACHE_DIR        = (_DATA_ROOT / "ocr_cache").resolve()
DATASET_CSV          = (_DATA_ROOT / "training_labels.csv").resolve()
DATASET_JSONL        = (_DATA_ROOT / "training_labels.jsonl").resolve()
HEUR_MODEL_PKL       = (_DATA_ROOT / "vision_model.pkl").resolve()

for _p in [IMAGES_IN_DIR, IMAGES_DONE_DIR, IMAGES_PROCESSED_DIR, OCR_CACHE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Flexible imports (prefer in-repo layout; also works if package installed as Lavish_Bot)
try:
    from lavish_core.vision import ocr
    from lavish_core.vision.parser_triggers import parse_trade_text
except ModuleNotFoundError:
    try:
        from Lavish_Bot.lavish_core.vision import ocr
        from Lavish_Bot.lavish_core.vision.parser_triggers import parse_trade_text
    except ModuleNotFoundError:
        # Last resort: add repo root to sys.path dynamically
        sys.path.append(str(_LAVISH_CORE.parent))
        from lavish_core.vision import ocr
        from lavish_core.vision.parser_triggers import parse_trade_text

# ── Defaults
RAW_DIR_DEFAULT    = str(IMAGES_IN_DIR)
CACHE_DIR_DEFAULT  = str(OCR_CACHE_DIR)
CSV_OUT_DEFAULT    = str(DATASET_CSV)
JSONL_OUT_DEFAULT  = str(DATASET_JSONL)
MODEL_OUT_DEFAULT  = str(HEUR_MODEL_PKL)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".TIF", ".TIFF"}

logger = logging.getLogger("LavishVisionTrainer")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                       "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted({os.path.normpath(p) for p in files})

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ──────────────────────────────────────────────────────────────────────────────
# OpenCV preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(path: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {"preproc": []}
    img = cv2.imread(path)
    if img is None:
        return None, meta
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); meta["preproc"].append("gray")
    gray = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
    meta["preproc"].append("denoise")
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 8); meta["preproc"].append("athr")
    # deskew heuristic
    try:
        coords = cv2.findNonZero(255 - thr)
        angle = cv2.minAreaRect(coords)[-1] if coords is not None else 0
        angle = -(90 + angle) if angle < -45 else -angle
        h, w = thr.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thr = cv2.warpAffine(thr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        meta["preproc"].append(f"deskew:{round(angle, 2)}deg")
    except Exception:
        pass
    return thr, meta

# ──────────────────────────────────────────────────────────────────────────────
# OCR pipeline (prefer local ocr.ocr_image; fallbacks to Paddle/Tesseract)
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_paddle(image_bgr_or_gray) -> Optional[str]:
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = engine.ocr(image_bgr_or_gray, cls=True)
        lines: List[str] = []
        for page in result or []:
            for item in page or []:
                txt = item[1][0] if isinstance(item, list) and len(item) >= 2 else ""
                if txt:
                    lines.append(txt)
        return "\n".join(lines).strip()
    except Exception:
        return None

def _ocr_tesseract(image_gray) -> Optional[str]:
    try:
        import pytesseract
        return pytesseract.image_to_string(image_gray or "", lang="eng")
    except Exception:
        return None

def ocr_with_fallback(img_path: str, pre_img) -> str:
    # 1) Your local OCR (ocr.py) — returns dict {'text': ..., 'lines': ...}
    try:
        res = ocr.ocr_image(img_path, lang="eng")
        if isinstance(res, dict) and res.get("text"):
            return str(res["text"])
        if isinstance(res, str) and res.strip():
            return res
    except Exception:
        pass
    # 2) PaddleOCR
    text = _ocr_paddle(pre_img)
    if text and text.strip():
        return text
    # 3) Tesseract
    text = _ocr_tesseract(pre_img)
    return text or ""

# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def cache_load(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def cache_save(cache_dir: str, key: str, data: Dict[str, Any]) -> None:
    ensure_dir(cache_dir)
    tmp = os.path.join(cache_dir, f"{key}.json.tmp")
    out = os.path.join(cache_dir, f"{key}.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)

# ──────────────────────────────────────────────────────────────────────────────
# Quick heuristic “model” (optional)
# ──────────────────────────────────────────────────────────────────────────────

def build_quick_model(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights: Dict[str, float] = {}

    def bump(word: str, y: float) -> None:
        if not word:
            return
        w = word.lower()
        weights[w] = weights.get(w, 0.0) + y

    for ex in examples:
        text = (ex.get("ocr_text") or "")
        label = (ex.get("label") or ex.get("parsed", {}).get("label") or "").upper()
        y = 1.0 if label in ("BUY", "CALL") else (-1.0 if label in ("SELL", "PUT") else 0.0)
        for w in re.findall(r"[A-Za-z]{3,}", text):
            bump(w, y)

    for k in list(weights.keys()):
        weights[k] = round(weights[k] / 5.0, 4)

    return {"word_weights": weights, "built_at": datetime.now(timezone.utc).isoformat()}

# ──────────────────────────────────────────────────────────────────────────────
# Core worker
# ──────────────────────────────────────────────────────────────────────────────

def process_one(img_path: str, cache_dir: str, fast: bool = False) -> Dict[str, Any]:
    key = sha1_file(img_path)
    cached = cache_load(cache_dir, key)
    if cached and fast:
        cached["from_cache"] = True
        return cached

    t0 = time.time()
    pre_img, pre_meta = preprocess_image(img_path)
    if pre_img is None:
        rec = {"image_path": img_path, "error": "read-failed"}
        cache_save(cache_dir, key, rec)
        return rec

    text = ocr_with_fallback(img_path, pre_img)
    text_norm = normalize_text(text)

    try:
        parsed = parse_trade_text(text_norm) or {}
    except Exception as e:
        parsed = {"error": f"parse-failed: {e}"}

    rec: Dict[str, Any] = {
        "image_path": img_path,
        "cache_key": key,
        "preproc": pre_meta.get("preproc", []),
        "ocr_text": text_norm,
        "parsed": parsed,
        "duration_sec": round(time.time() - t0, 3),
    }
    cache_save(cache_dir, key, rec)
    rec["from_cache"] = False
    return rec

# ──────────────────────────────────────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "image_path", "label", "ticker", "action", "option_type", "expiry",
    "strike", "confidence", "ocr_text", "cache_key", "notes"
]

def to_label_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    p = rec.get("parsed") or {}
    return {
        "image_path": rec.get("image_path", ""),
        "label": (p.get("label") or p.get("action") or "").upper(),
        "ticker": p.get("ticker", ""),
        "action": (p.get("action") or "").upper(),
        "option_type": (p.get("option_type") or "").upper(),
        "expiry": p.get("expiry", ""),
        "strike": p.get("strike", ""),
        "confidence": p.get("confidence", ""),
        "ocr_text": rec.get("ocr_text", ""),
        "cache_key": rec.get("cache_key", ""),
        "notes": "",
    }

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Lavish Vision Trainer")
    ap.add_argument("--raw", default=RAW_DIR_DEFAULT, help="Folder with historical images")
    ap.add_argument("--cache", default=CACHE_DIR_DEFAULT, help="OCR cache folder")
    ap.add_argument("--csv", default=CSV_OUT_DEFAULT, help="Output CSV labels")
    ap.add_argument("--jsonl", default=JSONL_OUT_DEFAULT, help="Output JSONL")
    ap.add_argument("--model-out", default=MODEL_OUT_DEFAULT, help="Output heuristic model pkl")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Thread workers")
    ap.add_argument("--fast", action="store_true", help="Trust cache when present (skip re-OCR)")
    ap.add_argument("--retrain", action="store_true", help="Also build/update heuristic model from outputs")
    args = ap.parse_args()

    ensure_dir(args.cache)
    imgs = list_images(args.raw)
    if not imgs:
        logger.warning(f"No images found in {args.raw}. Nothing to do.")
        return

    logger.info(f"Found {len(imgs)} images in {args.raw}")
    logger.info(f"Workers={args.workers}  Cache={args.cache}  Fast={args.fast}")

    results: List[Dict[str, Any]] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, p, args.cache, args.fast): p for p in imgs}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                rec = fut.result()
                results.append(rec)
                img = os.path.basename(rec.get("image_path", ""))
                lbl = (rec.get("parsed") or {}).get("label", "") or "NA"
                src_flag = "cache" if rec.get("from_cache") else "ocr"
                logger.info(f"OK  {img}  → {lbl}  ({src_flag})")
            except Exception as e:
                failures += 1
                logger.error(f"FAIL processing {src}: {e}")

    # Persist datasets
    rows = [to_label_row(r) for r in results]
    write_csv(rows, args.csv)
    write_jsonl(results, args.jsonl)
    logger.info(f"Saved CSV   → {args.csv}  ({len(rows)} rows)")
    logger.info(f"Saved JSONL → {args.jsonl}")

    # Optional lightweight model
    if args.retrain:
        model = build_quick_model(results)
        with open(args.model_out, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved heuristic model → {args.model_out}")

    logger.info(f"Done. Success={len(results) - failures}  Failures={failures}")

if __name__ == "__main__":
    main()