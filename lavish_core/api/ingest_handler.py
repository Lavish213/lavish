# lavish_bot/api/ingest_handler.py
import os, io, requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from lavish_core.vision.parser import analyze_image
from lavish_core.utils.quotes import latest_price
from lavish_core.utils.alerts import post_discord

app = FastAPI(title="Lavish Bot — Image Ingest")

class ImageURL(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "Lavish Bot image ingest online."}

def _act_on_parse(parsed):
    """Light action layer: fetch prices, craft alert."""
    if not parsed.get("tickers"):
        raise HTTPException(status_code=422, detail="No ticker detected in image.")
    prices = {}
    for t in parsed["tickers"]:
        q = latest_price(t)
        prices[t] = q["price"] if q else None
    # Build Discord message
    msg = f"📸 New signal (OCR): {parsed.get('direction','unclear').upper()} | Tickers: {', '.join(parsed['tickers'])}"
    if any(prices.values()):
        msg += "\n" + "\n".join([f"• {t}: {prices[t]}" for t in parsed["tickers"]])
    post_discord(msg, embed={
        "title": "Parsed Post",
        "description": (parsed.get("raw_text","")[:900] + "…") if len(parsed.get("raw_text",""))>900 else parsed.get("raw_text",""),
    })
    return {"tickers": parsed["tickers"], "direction": parsed["direction"], "prices": prices}

@app.post("/ingest_image")
async def ingest_image(file: UploadFile = File(...)):
    data = await file.read()
    parsed = analyze_image(data)
    result = _act_on_parse(parsed)
    return JSONResponse(result)

@app.post("/ingest_url")
def ingest_url(payload: ImageURL):
    try:
        r = requests.get(payload.url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    parsed = analyze_image(r.content)
    result = _act_on_parse(parsed)
    return JSONResponse(result)
