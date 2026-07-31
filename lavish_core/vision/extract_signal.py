# lavish_core/vision/extract_signal.py
# Fast parser for Discord/IG screenshots of trade cards (calls/puts),
# extracts: ticker, side, strike, expiry, price hints, targets, stops, confidence.
# Dependencies: pillow, pytesseract, opencv-python, rapidfuzz, pandas, pyarrow (optional)

import re, os, json, math, string
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Tuple, Optional

from rapidfuzz import process, fuzz

# cv2/PIL/pytesseract are only needed for the image-OCR path (ocr_image/
# parse_image). Text-only alert parsing (parse_text/parse_alert with no
# image) shouldn't require installing OpenCV + Tesseract just to read a
# Discord message, so these are imported lazily inside ocr_image() instead.

# Optional but nice
try:
    import pandas as pd
except Exception:
    pd = None

# ---------- Config ----------

ROOT = Path(__file__).resolve().parents[2]  # repo root guess
VISION_DIR = ROOT / "vision"
RAW_DIR = VISION_DIR / "raw"
OUT_DIR = VISION_DIR / "images_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_SIDES = ["CALL", "PUT"]
# weekday expiries common for weeklies
WEEKDAYS = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

# common month names/short
MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
}

# if you have a tickers file (from Finnhub), load it to validate symbols
TICKERS_CSV = (ROOT/"data"/"tickers.csv")
KNOWN_TICKERS = set()
if TICKERS_CSV.exists():
    try:
        import csv
        with open(TICKERS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym = (row.get("symbol") or "").strip().upper()
                if sym:
                    KNOWN_TICKERS.add(sym)
    except Exception:
        pass

# Fall back to the same env-configurable whitelist used by the Patreon/vision
# parsers, so a signal source with no tickers.csv doesn't silently fall back
# to "any all-caps word is a ticker" (the bug that produced garbage trades
# before the whitelist was added elsewhere in the pipeline).
if not KNOWN_TICKERS:
    KNOWN_TICKERS = {
        s.strip().upper()
        for s in os.getenv(
            "WHITELIST_TICKERS",
            "AAPL,MSFT,AMD,NVDA,META,TSLA,SPY,QQQ,GOOGL,CRM,MSTR",
        ).split(",")
        if s.strip()
    }

# ---------- Utilities ----------

def _clean_text(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    # Strip thousands-separator commas ("$1,120" -> "$1120") before any of
    # the numeric regexes run - otherwise a comma splits the digits and a
    # strike like "$1,120 Call" gets misread as strike 120, resolving to a
    # completely different (and real, tradable) contract.
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _to_float(x: str) -> Optional[float]:
    try:
        x = x.replace(",", "")
        return float(x)
    except Exception:
        return None

def _nearest_friday(d: date) -> date:
    # many screenshots show weeklies by date (e.g., "May 31")
    # if the parsed date isn't Fri, nudge to nearest Fri within the same week
    wd = d.weekday()
    if wd == 4:
        return d
    # push forward to next Fri, but cap at +6
    delta = (4 - wd) % 7
    return d + timedelta(days=delta)

def _parse_numeric_date(text: str, year: int) -> Optional[date]:
    # e.g., "5/24", "5/31", "07/24/26" - the format actually used in her
    # alerts ("SPY $528.00 Put 5/24"), as opposed to the month-name format
    # ("May 31") the OCR trade-card screenshots tend to show.
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if not m:
        return None
    mm, dd = int(m.group(1)), int(m.group(2))
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    yy = m.group(3)
    if yy:
        yy = int(yy)
        y = yy + 2000 if yy < 100 else yy
    else:
        y = year
    try:
        d = date(y, mm, dd)
        if yy is None and d < date.today():
            d = date(year + 1, mm, dd)
        return d
    except Exception:
        return None

def _parse_month_day(text: str, year: int) -> Optional[date]:
    # e.g., "May 31", "Jun 7", "June 21"
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})\b", text)
    if not m:
        return _parse_numeric_date(text, year)
    mon = m.group(1).lower()
    day = int(m.group(2))
    if mon not in MONTHS:
        return _parse_numeric_date(text, year)
    mm = MONTHS[mon]
    try:
        d = date(year, mm, day)
        return _nearest_friday(d)
    except Exception:
        return None

def _best_ticker_candidate(text: str) -> Tuple[Optional[str], int]:
    # Pick something like "AAPL", "SPY", "NVDA" etc.
    # 1) strong candidates: ALLCAPS 1–5 letters
    caps = re.findall(r"\b[A-Z]{1,5}\b", text)
    if KNOWN_TICKERS:
        # fuzzy match to known set to avoid FALSE positives like "CALL", "VIEW", etc.
        best = None
        best_score = 0
        for c in caps:
            match, score, _ = process.extractOne(c, KNOWN_TICKERS, scorer=fuzz.WRatio)
            if score > best_score:
                best, best_score = match, int(score)
        return (best, best_score)
    else:
        # heuristic: ignore common words
        blacklist = {"CALL","PUT","VIEW","EVERYONE","TODAY","BUY","SELL","SPY","GME","NVDA","META","ORCL"}
        # NOTE: leaving SPY/NVDA/etc. in blacklist would remove them; remove from blacklist:
        blacklist = {"CALL","PUT","VIEW","EVERYONE","TODAY","BUY","SELL"}
        for c in caps:
            if c not in blacklist:
                return (c, 60)
        return (None, 0)

def _find_side(text: str) -> Optional[str]:
    if re.search(r"\bPUT\b", text, re.I):  return "PUT"
    if re.search(r"\bCALL\b", text, re.I): return "CALL"
    return None

def _find_strike(text: str) -> Optional[float]:
    # $487.5 Call, 528.00 Put, 1120 Call, etc.
    m = re.search(r"\$?\s?(\d{1,4}(?:\.\d{1,2})?)\s*(?:call|put)\b", text, re.I)
    if m:
        return _to_float(m.group(1))
    # sometimes “$528.00” appears near a big “SPY $528.00 Put”
    m2 = re.search(r"\b(?:\$|USD)?\s*(\d{2,4}(?:\.\d{1,2})?)\s*(?:\$)?\b", text)
    if m2:
        return _to_float(m2.group(1))
    return None

def _find_entry_price(text: str) -> Optional[float]:
    # the green price pill: "$1.74", "$2.41", "$0.62", "$2.95", etc.
    # we try to bias by nearby words like "Today", "+" %, etc., but for simplicity:
    prices = re.findall(r"\$\s?(\d{1,3}(?:\.\d{1,2})?)", text)
    # pick a plausible contract price in 0.05–100
    cand = None
    for p in prices:
        v = _to_float(p)
        if v is not None and 0.05 <= v <= 100:
            cand = v
    return cand

def _find_target(text: str) -> Optional[float]:
    # “Target $183” or “Target 526” or “Target $1,120”
    m = re.search(r"\btarget\s*\$?\s*(\d{2,5}(?:\.\d{1,2})?)", text, re.I)
    if m: return _to_float(m.group(1))
    # “Should hit $1,120” (NVDA)
    m2 = re.search(r"\b(hit|reach|to|should (?:hit|reach))\s*\$?\s*(\d{2,5}(?:\.\d{1,2})?)", text, re.I)
    if m2: return _to_float(m2.group(2))
    return None

def _find_stop(text: str) -> Optional[float]:
    # “Stop loss 528.20” or “SL 528.20”
    m = re.search(r"\b(stop|sl|stop\s*loss)\s*\$?\s*(\d{2,5}(?:\.\d{1,2})?)", text, re.I)
    if m: return _to_float(m.group(2))
    return None

def _extract(text: str, img_path: Path) -> Dict[str, Any]:
    now = datetime.now()
    side = _find_side(text)
    ticker, ticker_score = _best_ticker_candidate(text)
    strike = _find_strike(text)
    entry  = _find_entry_price(text)
    # expiry from phrases like “May 24 / May 31 / Jun 7 …”
    expiry = _parse_month_day(text, year=now.year) or _parse_month_day(text, year=now.year+1)
    target = _find_target(text)
    stop   = _find_stop(text)

    conf = 0.0
    conf += 0.25 if ticker else 0.0
    conf += 0.20 if side else 0.0
    conf += 0.20 if strike else 0.0
    conf += 0.10 if entry else 0.0
    conf += 0.10 if expiry else 0.0
    conf += 0.10 if (target or stop) else 0.0
    conf = round(min(0.99, conf), 2)

    return {
        "image": str(img_path),
        "timestamp": now.isoformat(timespec="seconds"),
        "ticker": ticker,
        "ticker_score": ticker_score,
        "side": side,
        "strike": strike,
        "expiry": expiry.isoformat() if expiry else None,
        "entry_price_est": entry,
        "target_hint": target,
        "stop_hint": stop,
        "notes_excerpt": text[:280],
        "confidence": conf,
        "needs_human_review": conf < 0.75,
    }

# ---------- OCR Pipeline ----------

def _preprocess_for_ocr(img_bgr) -> "np.ndarray":
    import cv2
    # 1) make a hi-contrast grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 2) light denoise & sharpen
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    # 3) adaptive threshold (works well on dark discord UIs)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    return thr

def ocr_image(img_path: Path) -> str:
    import cv2
    from PIL import Image
    import pytesseract
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    thr = _preprocess_for_ocr(img)
    # Save debug
    dbg = OUT_DIR / f"{img_path.stem}_bin.png"
    cv2.imwrite(str(dbg), thr)
    # pytesseract
    config = "--oem 3 --psm 6"   # PSM 6 = assume a single uniform block of text
    txt = pytesseract.image_to_string(Image.fromarray(thr), config=config)
    return _clean_text(txt)

# ---------- Public API ----------

def parse_image(img_path: Path) -> Dict[str, Any]:
    text = ocr_image(img_path)
    return _extract(text, img_path)

def parse_text(text: str, source: str = "text") -> Dict[str, Any]:
    """
    Same extraction as parse_image(), for alerts that arrive as plain text
    (e.g. a Discord message body) with no attached trade-card screenshot.
    """
    return _extract(_clean_text(text or ""), Path(source))

def parse_alert(text: str = "", img_path: Optional[Path] = None, source: str = "text") -> Dict[str, Any]:
    """
    Combine text + an optional trade-card screenshot into one signal: OCR
    the image (if given) and merge it with whatever the message text itself
    contains, preferring whichever side found a given field.
    """
    text_result = parse_text(text, source=source) if text else None
    image_result = parse_image(img_path) if img_path else None

    if text_result and image_result:
        merged = dict(text_result)
        for key in ("ticker", "side", "strike", "expiry", "entry_price_est", "target_hint", "stop_hint"):
            if not merged.get(key) and image_result.get(key):
                merged[key] = image_result[key]
        merged["confidence"] = max(text_result.get("confidence", 0.0), image_result.get("confidence", 0.0))
        merged["needs_human_review"] = merged["confidence"] < 0.75
        merged["notes_excerpt"] = (text_result.get("notes_excerpt") or image_result.get("notes_excerpt"))
        return merged

    return text_result or image_result or _extract("", Path(source))

def parse_folder(folder: Path) -> List[Dict[str, Any]]:
    results = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.heic"):
        for p in folder.glob(ext):
            try:
                results.append(parse_image(p))
            except Exception as e:
                results.append({
                    "image": str(p),
                    "error": repr(e),
                    "confidence": 0.0,
                    "needs_human_review": True
                })
    return results

def save_results(rows: List[Dict[str, Any]]):
    VISION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = VISION_DIR / "signals.csv"
    json_path = VISION_DIR / "signals.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    # CSV/Parquet (if pandas present)
    if pd:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(VISION_DIR / "signals.parquet", index=False)
        except Exception:
            pass
    print(f"✅ saved {len(rows)} signals -> {csv_path}")

if __name__ == "__main__":
    folder = RAW_DIR if (RAW_DIR.exists()) else Path.cwd()
    rows = parse_folder(folder)
    save_results(rows)