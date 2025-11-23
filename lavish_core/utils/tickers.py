# lavish_bot/utils/tickers.py
import os, json, time, requests
from pathlib import Path
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

load_dotenv()

CACHE = Path(__file__).resolve().parents[1] / "data" / "ticker_cache.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)

FINNHUB_KEYS = [k for k in [os.getenv("FINNHUB_API_KEY"), os.getenv("FINNHUB_API_KEY_2")] if k]

def _fetch_from_finnhub(key: str):
    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {"exchange": "US", "token": key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = r.json()
    # Keep only normal equities/ETFs
    tickers = sorted({it["symbol"] for it in items if isinstance(it.get("symbol"), str)})
    return tickers

def refresh_cache(force=False):
    if CACHE.exists() and not force:
        return
    last_err = None
    for k in FINNHUB_KEYS:
        try:
            tickers = _fetch_from_finnhub(k)
            CACHE.write_text(json.dumps({"ts": time.time(), "tickers": tickers}))
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to refresh tickers: {last_err}")

def load_tickers(max_age_hours=24):
    if not CACHE.exists():
        refresh_cache(force=True)
    else:
        data = json.loads(CACHE.read_text())
        if time.time() - data.get("ts", 0) > max_age_hours * 3600:
            try: refresh_cache(force=True)
            except: pass  # keep stale cache
    data = json.loads(CACHE.read_text())
    return set(data["tickers"])

def fuzzy_find_tickers(words: list[str], all_tickers: set[str], limit=6, score_cutoff=86):
    # Try exact first
    found = {w for w in words if w in all_tickers}
    remain = [w for w in words if w not in found and 1 < len(w) <= 5 and w.isupper()]
    if not remain: return sorted(found)
    # Fuzzy on remainder
    choices = list(all_tickers)
    for w in remain:
        match = process.extractOne(w, choices, scorer=fuzz.QRatio, score_cutoff=score_cutoff)
        if match: found.add(match[0])
    return sorted(found)