#!/usr/bin/env python3
"""
run_vision_batch.py — master controller for screenshot ingestion

Features
- Auto-discovers raw screenshots folder (local & iCloud fallbacks)
- Processes images in batches (default 100) every 15 minutes (or run once)
- Parallel workers (default 4)
- OCR via pytesseract; optional LLM-vision enrichment (OpenAI if key present)
- Robust logging, rotating log file, manifest to skip duplicates
- Moves handled files to processed/ or failed/
- Writes extracted text/JSON to vision/outputs/

Usage
  # one batch of up to 100 images
  python run_vision_batch.py --once

  # continuous (100 every 15 min)
  python run_vision_batch.py --loop

  # customize
  python run_vision_batch.py --batch 200 --interval 10 --workers 6
"""

from __future__ import annotations
import os, sys, time, json, csv, shutil, traceback, argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Config (tune here) ----------
BATCH_SIZE_DEFAULT   = 100
INTERVAL_MIN_DEFAULT = 15
MAX_WORKERS_DEFAULT  = 4
VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".heic"}

# Try these raw paths in order (first one that exists is used)
RAW_CANDIDATES = [
    Path("lavish_core/vision/raw"),
    Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Lavish_bot/vision/raw",  # iCloud (macOS)
    Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Lavish_bot/raw",
    Path("vision/raw"),
]

VISION_ROOT = Path("lavish_core/vision")
OUTPUT_DIR  = VISION_ROOT / "outputs"
PROCESSED_DIR = VISION_ROOT / "processed"
FAILED_DIR    = VISION_ROOT / "failed"
MANIFEST_JSON = VISION_ROOT / "processed_manifest.json"
LOG_DIR     = Path("logs")
LOG_PATH    = LOG_DIR / "batch_master.log"

# Optional OpenAI vision (set OPENAI_API_KEY to enable)
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")  # change if you want
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------- Lightweight logger ----------
def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{_ts()}] {msg}\n")
    print(f"[{_ts()}] {msg}")

# ---------- Paths & manifest ----------
def ensure_dirs():
    for p in [VISION_ROOT, OUTPUT_DIR, PROCESSED_DIR, FAILED_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def discover_raw_dir() -> Path:
    for p in RAW_CANDIDATES:
        if p.exists() and p.is_dir():
            log(f"✅ Using RAW dir: {p}")
            return p
    # last resort: create local raw
    fallback = VISION_ROOT / "raw"
    fallback.mkdir(parents=True, exist_ok=True)
    log(f"⚠️ No RAW candidates found; using {fallback}")
    return fallback

def load_manifest() -> set[str]:
    if MANIFEST_JSON.exists():
        try:
            data = json.loads(MANIFEST_JSON.read_text())
            if isinstance(data, list):
                return set(data)
        except Exception:
            pass
    return set()

def save_manifest(done: set[str]):
    try:
        MANIFEST_JSON.write_text(json.dumps(sorted(list(done)), indent=2))
    except Exception as e:
        log(f"⚠️ Failed to save manifest: {e}")

# ---------- OCR & optional LLM ----------
def safe_import_tesseract():
    try:
        import pytesseract
        from PIL import Image
        return pytesseract, Image
    except Exception as e:
        log(f"⚠️ pytesseract/Pillow not available: {e}")
        return None, None

def run_ocr(image_path: Path) -> str:
    pytesseract, Image = safe_import_tesseract()
    if not pytesseract or not Image:
        return ""

    try:
        img = Image.open(image_path)
        # Tesseract hint: set PSM for block of text (6) & stocky UI text (single uniform block)
        cfg = "--psm 6"
        text = pytesseract.image_to_string(img, config=cfg)
        return text.strip()
    except Exception as e:
        log(f"❌ OCR error {image_path.name}: {e}")
        return ""

def openai_vision_summarize(text: str) -> dict | None:
    """Optional: call LLM to structure signals; requires OPENAI_API_KEY."""
    if not OPENAI_API_KEY or not text.strip():
        return None
    try:
        import requests
        prompt = (
            "You are a trading assistant. Extract any trade instructions, symbols, directions, "
            "price targets, stop levels, timeframes, and extra notes from the text. "
            "Return JSON with fields: {symbols:[], actions:[], targets:[], stops:[], timeframe:'', notes:''}. "
            "If nothing actionable, return empty arrays."
        )
        body = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role":"system","content":"You convert noisy OCR text into clean trade signals."},
                {"role":"user","content": f"{prompt}\n\n---\n{text}\n---"}
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        r = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        out = r.json()
        content = out["choices"][0]["message"]["content"]
        # try to parse JSON from content
        try:
            parsed = json.loads(content)
            return parsed
        except Exception:
            # fallback: wrap raw string
            return {"raw_summary": content}
    except Exception as e:
        log(f"⚠️ OpenAI vision summary failed: {e}")
        return None

# ---------- Single-file worker ----------
def process_one(image_path: Path) -> dict:
    t0 = time.time()
    info = {
        "file": str(image_path),
        "start": _ts(),
        "status": "unknown",
        "ocr_chars": 0,
        "elapsed_s": 0.0,
        "reason": "",
        "llm_used": bool(OPENAI_API_KEY),
    }
    try:
        if not image_path.exists():
            info["status"] = "skip"
            info["reason"] = "missing"
            return info

        text = run_ocr(image_path)
        info["ocr_chars"] = len(text)

        # write OCR text dump
        out_txt = (OUTPUT_DIR / (image_path.stem + ".txt"))
        out_txt.write_text(text or "", encoding="utf-8")

        # optional LLM summarize
        summary = openai_vision_summarize(text) if OPENAI_API_KEY else None
        if summary:
            out_json = (OUTPUT_DIR / (image_path.stem + ".json"))
            out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # if we reached here, call it success even if OCR empty (we still “processed”)
        (PROCESSED_DIR / image_path.name).write_bytes(image_path.read_bytes())
        image_path.unlink(missing_ok=True)

        info["status"] = "ok"
        return info
    except Exception as e:
        info["status"] = "fail"
        info["reason"] = f"{e}"
        try:
            # move to failed
            (FAILED_DIR / image_path.name).write_bytes(image_path.read_bytes())
            image_path.unlink(missing_ok=True)
        except Exception as e2:
            log(f"⚠️ could not move failed file {image_path.name}: {e2}")
        return info
    finally:
        info["elapsed_s"] = round(time.time() - t0, 3)

# ---------- Batch orchestration ----------
def list_pending(raw_dir: Path, manifest_done: set[str], limit: int) -> list[Path]:
    files = []
    for p in sorted(raw_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0):
        if p.is_file() and p.suffix.lower() in VALID_EXTS and (str(p) not in manifest_done):
            files.append(p)
            if len(files) >= limit:
                break
    return files

def run_batch(raw_dir: Path, batch_size: int, workers: int, manifest_done: set[str]) -> dict:
    todo = list_pending(raw_dir, manifest_done, batch_size)
    if not todo:
        log("🔎 No pending images to process.")
        return {"count": 0, "ok": 0, "fail": 0, "elapsed_s": 0.0}

    t0 = time.time()
    ok = fail = 0
    results = []

    log(f"🚀 Starting batch: {len(todo)} images | workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, p): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                info = fut.result()
                results.append(info)
                if info["status"] == "ok":
                    ok += 1
                elif info["status"] == "fail":
                    fail += 1
                manifest_done.add(str(p))
            except Exception as e:
                fail += 1
                manifest_done.add(str(p))
                log(f"❌ Unexpected worker error for {p.name}: {e}")
                log(traceback.format_exc())

    # write batch CSV log (append)
    batch_log = OUTPUT_DIR / "batches.csv"
    is_new = not batch_log.exists()
    with batch_log.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","start","status","ocr_chars","elapsed_s","reason","llm_used"])
        if is_new:
            w.writeheader()
        for r in results:
            w.writerow(r)

    elapsed = round(time.time() - t0, 2)
    log(f"✅ Batch complete: total={len(todo)} ok={ok} fail={fail} elapsed={elapsed}s")
    return {"count": len(todo), "ok": ok, "fail": fail, "elapsed_s": elapsed}

# ---------- CLI loop ----------
def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Lavish vision batch processor")
    parser.add_argument("--once", action="store_true", help="run one batch and exit")
    parser.add_argument("--loop", action="store_true", help="run forever (sleep between batches)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE_DEFAULT, help="images per batch")
    parser.add_argument("--interval", type=int, default=INTERVAL_MIN_DEFAULT, help="minutes between batches (loop mode)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT, help="parallel workers")
    args = parser.parse_args()

    raw_dir = discover_raw_dir()
    done = load_manifest()

    if args.once and args.loop:
        log("⚠️ Choose either --once OR --loop, not both.")
        sys.exit(2)

    if args.once:
        stats = run_batch(raw_dir, args.batch, args.workers, done)
        save_manifest(done)
        # also write a simple heartbeat
        (OUTPUT_DIR / "last_run.json").write_text(json.dumps({
            "ended_at": _ts(),
            "stats": stats
        }, indent=2))
        return

    # default: loop
    interval_s = max(1, args.interval) * 60
    while True:
        stats = run_batch(raw_dir, args.batch, args.workers, done)
        save_manifest(done)
        (OUTPUT_DIR / "last_run.json").write_text(json.dumps({
            "ended_at": _ts(),
            "stats": stats
        }, indent=2))
        log(f"💤 Sleeping {args.interval} min ...")
        time.sleep(interval_s)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("👋 Stopping on user interrupt.")