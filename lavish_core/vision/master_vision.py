# master_vision.py
import os, sys, json, hashlib, time
from pathlib import Path
from datetime import datetime
import pandas as pd

# optional deps
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# optional: local captioning (BLIP) if installed
BLIP_OK = False
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_OK = True
except Exception:
    BLIP_OK = False

# optional: OpenAI Vision (server call)
OPENAI_OK = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if OPENAI_API_KEY:
    OPENAI_OK = True
    import requests

DATA = Path("data")
IN_DIR = DATA / "images_in"
OUT_DIR = DATA / "images_out"
VEC_DIR = DATA / "vector_index"
for p in [IN_DIR, OUT_DIR, VEC_DIR]: p.mkdir(parents=True, exist_ok=True)

CAP_PATH = OUT_DIR / "captions.jsonl"
OCR_PATH = OUT_DIR / "ocr.jsonl"
EMB_PATH = OUT_DIR / "embeddings.parquet"

# -------- helpers
def iter_images(root: Path):
    exts = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

def sha1(path: Path)->str:
    h=hashlib.sha1()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def openai_embed(text: str) -> list | None:
    if not OPENAI_OK: return None
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model":"text-embedding-3-small","input": text[:8000]}
    for i in range(3):
        r = requests.post(url, json=data, headers=headers, timeout=30)
        if r.status_code==200:
            return r.json()["data"][0]["embedding"]
        time.sleep(1.5*(i+1))
    return None

def openai_caption(p: Path) -> str | None:
    if not OPENAI_OK: return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    # send image via URL or base64. We’ll send as base64 for local files:
    import base64
    b64 = base64.b64encode(p.read_bytes()).decode()
    payload = {
      "model":"gpt-4o-mini",
      "messages":[
        {"role":"user","content":[
            {"type":"text","text":"Describe this image in one detailed paragraph, concise and neutral."},
            {"type":"image_url","image_url":{"url":f"data:image/{p.suffix[1:]};base64,{b64}"}}
        ]}
      ],
      "temperature":0.2
    }
    for i in range(3):
        r=requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code==200:
            return r.json()["choices"][0]["message"]["content"].strip()
        time.sleep(1.5*(i+1))
    return None

# optional BLIP init
blip_processor = blip_model = None
if BLIP_OK:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def blip_caption(p: Path) -> str | None:
    if not BLIP_OK or Image is None: return None
    raw_image = Image.open(p).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def local_ocr(p: Path) -> str | None:
    if pytesseract is None or Image is None: return None
    try:
        img = Image.open(p)
        return pytesseract.image_to_string(img)
    except Exception:
        return None

def upsert_embedding(df: pd.DataFrame, row: dict):
    # append row dict (id, path, text, embedding(list[float]))
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(EMB_PATH, index=False)
    return df

def main():
    # load existing embeddings parquet if present
    if EMB_PATH.exists():
        df = pd.read_parquet(EMB_PATH)
    else:
        df = pd.DataFrame(columns=["id","path","text","embedding"])
    seen_ids = set(df["id"].tolist())

    for img_path in iter_images(IN_DIR):
        img_id = sha1(img_path)
        if img_id in seen_ids: 
            continue

        record = {
            "id": img_id,
            "path": str(img_path.relative_to(DATA))
        }

        # OCR
        ocr_txt = local_ocr(img_path)
        if ocr_txt and ocr_txt.strip():
            append_jsonl(OCR_PATH, {"id":img_id,"path":record["path"],"text":ocr_txt})
        # captioning: prefer local BLIP (fast/offline), else OpenAI
        caption = blip_caption(img_path) if BLIP_OK else openai_caption(img_path)
        if caption:
            append_jsonl(CAP_PATH, {"id":img_id,"path":record["path"],"caption":caption})

        # pick best text for embedding (caption > OCR > filename)
        text_for_embed = caption or (ocr_txt if ocr_txt and ocr_txt.strip() else img_path.stem)
        emb = openai_embed(text_for_embed)  # simple, robust embeddings
        if emb:
            record.update({"text": text_for_embed, "embedding": emb})
            df = upsert_embedding(df, record)

    print("✅ vision ingestion complete")
    print(f"  ↳ {CAP_PATH} / {OCR_PATH} / {EMB_PATH}")

if __name__ == "__main__":
    main()
def parse_signal(text: str):
    """
    Extract a trade signal (buy/sell) from Patreon text.
    """
    from lavish_core.trading.executor import TradeSignal

    lower = text.lower()
    if "buy" in lower:
        return TradeSignal(symbol="AAPL", action="BUY", reason="parsed from Patreon")
    elif "sell" in lower:
        return TradeSignal(symbol="AAPL", action="SELL", reason="parsed from Patreon")
    return None