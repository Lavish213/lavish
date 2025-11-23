# lavish_bot/utils/quotes.py
import os, requests
from dotenv import load_dotenv
load_dotenv()

POLY = os.getenv("POLYGON_API_KEY")
FINN1 = os.getenv("FINNHUB_API_KEY")
FINN2 = os.getenv("FINNHUB_API_KEY_2")

def polygon_last(ticker: str):
    if not POLY: return None
    url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}"
    r = requests.get(url, params={"apiKey": POLY}, timeout=15)
    if r.status_code != 200: return None
    j = r.json()
    if "results" in j and j["results"]:
        p = j["results"]["p"]
        t = j["results"]["t"]
        return {"source": "polygon", "price": p, "ts": t}
    return None

def finnhub_last(ticker: str, key: str):
    url = "https://finnhub.io/api/v1/quote"
    r = requests.get(url, params={"symbol": ticker.upper(), "token": key}, timeout=15)
    if r.status_code != 200: return None
    j = r.json()
    p = j.get("c")
    if p: return {"source":"finnhub", "price": float(p), "ts": j.get("t")}
    return None

def latest_price(ticker: str):
    # Try Polygon, then Finnhub keys
    for fn in (polygon_last, lambda t: finnhub_last(t, FINN1), lambda t: finnhub_last(t, FINN2)):
        try:
            out = fn(ticker)
            if out: return out
        except: pass
    return None
