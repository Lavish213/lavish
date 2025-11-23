# lavish_core/trainer_loader.py
# Ingests everything in /raw (png/jpg/webp/pdf/txt/md/csv/json), runs OCR if available,
# cleans + chunks, builds embeddings, and saves to hybrid store.
# Also writes a JSONL cache so the super_core can warm-load in FAISS/TF-IDF modes.

import os, sys, re, io, json, uuid, glob, logging, traceback
from datetime import datetime, timezone
from typing import List, Dict, Tuple

RAW_DIR = "raw"
CACHE_DIR = "memory/vector_store_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("lavish_trainer")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)

# ---- Import hybrid pieces from super_core (so they stay consistent) ----
from lavish_core.super_core import HybridEmbedder, HybridStore

# ---- Optional OCR & PDF ----
def _read_pdf(path: str) -> str:
    text = ""
    try:
        import pdfminer.high_level
        text = pdfminer.high_level.extract_text(path) or ""
    except Exception:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e2:
            logger.warning(f"PDF read failed for {path}: {e2}")
    return text

def _read_image(path: str) -> str:
    # OCR if available
    try:
        import pytesseract
        from PIL import Image, ImageOps
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)
        txt = pytesseract.image_to_string(img)
        return txt
    except Exception as e:
        logger.warning(f"OCR not available or failed for {path}: {e}")
        return ""

def _read_text_like(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def _chunk(s: str, max_tokens: int = 700, overlap: int = 120) -> List[str]:
    """
    Token-agnostic chunker (approx). Adjust sizes for your embedder.
    """
    words = s.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + max_tokens)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        if j == len(words): break
        i = max(0, j - overlap)
    return chunks

def _detect_source(path: str) -> str:
    if "/news/" in path.lower() or "news_" in os.path.basename(path).lower():
        return "news"
    if "/trade/" in path.lower() or "trade_" in os.path.basename(path).lower():
        return "trade"
    if "/ocr/" in path.lower():
        return "ocr"
    return "memory"

def _write_cache(items: List[Dict]):
    if not items: 
        return
    out_path = os.path.join(CACHE_DIR, f"cache_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info(f"Wrote cache: {out_path} ({len(items)} chunks)")

def scan_files() -> List[Tuple[str, str]]:
    """
    Returns list of (path, text)
    """
    files = []
    exts = (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".pdf",".txt",".md",".csv",".json")
    for root, _, names in os.walk(RAW_DIR):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    logger.info(f"Found {len(files)} files in /{RAW_DIR}")
    pairs = []
    for p in files:
        txt = ""
        try:
            if p.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")):
                txt = _read_image(p)
            elif p.lower().endswith(".pdf"):
                txt = _read_pdf(p)
            elif p.lower().endswith((".txt",".md",".csv",".json")):
                txt = _read_text_like(p)
            else:
                txt = ""
        except Exception as e:
            logger.warning(f"Read failed for {p}: {e}")
        if txt and txt.strip():
            pairs.append((p, _clean_text(txt)))
    return pairs

def train_and_persist():
    # 1) collect text from files
    pairs = scan_files()
    if not pairs:
        logger.warning("No readable content in /raw")
        return

    texts_all, metas_all, ids_all = [], [], []
    items_for_cache = []

    # 2) chunk & label
    for path, txt in pairs:
        source = _detect_source(path)
        chunks = _chunk(txt, max_tokens=700, overlap=120)
        for ch in chunks:
            did = str(uuid.uuid4())
            meta = {"source": source, "file": path}
            texts_all.append(ch); metas_all.append(meta); ids_all.append(did)
            items_for_cache.append({"id": did, "text": ch, "meta": meta})

    # 3) fit + embed + add to store
    emb = HybridEmbedder()
    try:
        emb.fit(texts_all)
    except Exception:
        pass

    store = HybridStore(embedder=emb, persist_dir="memory/vector_store")
    store.add(texts_all, metas_all, ids_all)
    store.persist()

    # 4) write cache for fast warm-load (FAISS/TFIDF)
    _write_cache(items_for_cache)

    logger.info(f"Trainer finished: {len(texts_all)} chunks ingested.")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Lavish Trainer/Loader")
    p.add_argument("--once", action="store_true", help="Run a single pass and exit (default).")
    p.add_argument("--watch", action="store_true", help="Watch /raw and re-train on new files every N seconds.")
    p.add_argument("--interval", type=int, default=60, help="Watch interval seconds.")
    args = p.parse_args()

    if args.watch:
        seen = set()
        logger.info("Watching /raw for new content...")
        while True:
            try:
                current = set(glob.glob(os.path.join(RAW_DIR, "**/*"), recursive=True))
                new_files = [p for p in current if p not in seen and os.path.isfile(p)]
                if new_files:
                    logger.info(f"New files detected: {len(new_files)}")
                    train_and_persist()
                    seen |= set(new_files)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.error("Watcher error:\n" + traceback.format_exc())
                time.sleep(args.interval)
    else:
        train_and_persist()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal trainer error:\n" + traceback.format_exc())
        raise