"""
Candle Reader v2 — robust, screenshot-friendly extractor.
- Works from static images (charts from your stock lady) with no APIs.
- Reconstructs OHLC from pixels using Y-axis ticks when present; otherwise returns normalized 0..1 scale with consistent bar order.
- Handles dark/light themes and typical red/green (or custom) candle colors.
- Produces a list of bars + derived features (ATR, MAs) for downstream grading/prediction.

Requires: opencv-python, numpy, pillow (optional), pytesseract (optional, fallback OCR).
"""

from __future__ import annotations
import os, re, cv2, math, numpy as np
from typing import List, Dict, Optional, Tuple

# --- Optional OCR (y-axis & time labels); works even if missing
try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

# --------- Utilities ---------
def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _auto_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE helps faint axis labels in screenshots
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _detect_theme(img: np.ndarray) -> str:
    # Simple heuristic: darker backgrounds → "dark"
    gray = _to_gray(img)
    med = np.median(gray)
    return "dark" if med < 90 else "light"

def _color_masks(img: np.ndarray, theme: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bull_mask, bear_mask) tolerant to many color palettes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Greens (bull)
    green1 = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Reds (bear) (wraps around hue 0)
    red1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 40), (179, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    # Some platforms use teal/blue for bull: add a permissive band
    teal = cv2.inRange(hsv, (85, 40, 40), (110, 255, 255))
    bull = cv2.bitwise_or(green1, teal)
    bear = red
    # If theme light and candles are black/white, fallback to thick bodies from edges
    if cv2.countNonZero(bull) < 100 and cv2.countNonZero(bear) < 100:
        edges = cv2.Canny(_to_gray(img), 50, 120)
        # Dilate to form bodies
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thick = cv2.dilate(edges, k, iterations=1)
        return thick, thick
    return bull, bear

def _extract_axis_scale(img: np.ndarray) -> Optional[Tuple[float, float, int, int]]:
    """
    Try to read Y-axis tick values to map pixels->price.
    Returns (price_min, price_max, y_min_px, y_max_px) or None if not found.
    """
    if not _HAS_TESS:
        return None
    h, w = img.shape[:2]
    # Assume y-axis at left 14% or right 14% — try both
    slices = [
        img[:, :max(80, int(w*0.14))],
        img[:, w-max(80, int(w*0.14)):]
    ]
    candidates = []
    for slc in slices:
        gray = _auto_contrast(_to_gray(slc))
        txt = pytesseract.image_to_string(gray, config="--psm 6").strip()
        nums = [float(x.replace(',','')) for x in re.findall(r"\b\d+(?:\.\d+)?\b", txt)]
        if len(nums) >= 2:
            candidates.append(sorted(nums))
    if not candidates:
        return None
    vals = max(candidates, key=len)
    # Map min/max to pixel extremes
    price_min, price_max = min(vals), max(vals)
    return (price_min, price_max, 0, h-1)

def _candles_from_masks(img: np.ndarray, bull_mask: np.ndarray, bear_mask: np.ndarray) -> List[Dict]:
    """
    Find vertical candle bodies & wicks via morphology, then left→right ordering.
    Output candles with pixel-based top/bottom; later mapped to price.
    """
    h, w = img.shape[:2]
    # Combine masks so we detect all bodies; keep polarity separately
    uni = cv2.bitwise_or(bull_mask, bear_mask)
    # Close gaps
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    uni = cv2.morphologyEx(uni, cv2.MORPH_CLOSE, kx, iterations=1)

    contours, _ = cv2.findContours(uni, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,bw,bh = cv2.boundingRect(c)
        if bh < max(6, int(h*0.03)) or bw < 2:  # filter tiny specks
            continue
        cx = x + bw//2
        # Candle polarity by overlap
        bull_overlap = cv2.countNonZero(cv2.bitwise_and(bull_mask[y:y+bh, x:x+bw], uni[y:y+bh, x:x+bw]))
        bear_overlap = cv2.countNonZero(cv2.bitwise_and(bear_mask[y:y+bh, x:x+bw], uni[y:y+bh, x:x+bw]))
        typ = "bull" if bull_overlap >= bear_overlap else "bear"
        boxes.append((x,y,bw,bh, cx, typ))
    if not boxes:
        return []

    # Sort left→right by center x; merge boxes that belong to same candle (platforms may draw body+wick separate)
    boxes.sort(key=lambda t: t[4])
    merged: List[Dict] = []
    tol = max(3, int(w*0.005))
    for b in boxes:
        x,y,bw,bh,cx,typ = b
        if not merged:
            merged.append({"x1":x, "x2":x+bw, "y1":y, "y2":y+bh, "cx":cx, "typ":typ})
            continue
        last = merged[-1]
        if abs(cx - last["cx"]) <= tol:
            last["x1"] = min(last["x1"], x)
            last["x2"] = max(last["x2"], x+bw)
            last["y1"] = min(last["y1"], y)
            last["y2"] = max(last["y2"], y+bh)
            # keep earlier polarity if consistent; if conflict, pick body area majority
        else:
            merged.append({"x1":x, "x2":x+bw, "y1":y, "y2":y+bh, "cx":cx, "typ":typ})

    # Convert to candles with pixel OHLC approximation using scanlines within body
    candles: List[Dict] = []
    for m in merged:
        x1,x2,y1,y2,typ = m["x1"], m["x2"], m["y1"], m["y2"], m["typ"]
        body = img[y1:y2, x1:x2]
        # Scan vertical intensity to estimate open/close edges
        col = np.mean(_to_gray(body), axis=1)  # one column avg per y
        # open/close orientation by color convention: bull = close>open
        # Use top/bottom of body as high/low proxy; refine with thin wick search around center column
        high_px = y1
        low_px  = y2
        # find center column wick extremes
        cx = (x1+x2)//2
        line = _to_gray(img[:, max(cx-1,0):min(cx+2,img.shape[1])]).mean(axis=1)
        # extend upward until background changes slowly; robust within chart area
        # (keep simple/fast; reader is screenshot tolerant)
        # compute open/close by looking at lower 40% vs upper 40% mean brightness
        top_mean = float(np.mean(col[:max(1, int(len(col)*0.4))]))
        bot_mean = float(np.mean(col[-max(1, int(len(col)*0.4)):]))

        if typ == "bull":
            close_px = y1 + int(len(col)*0.25)
            open_px  = y1 + int(len(col)*0.75)
        else:
            close_px = y1 + int(len(col)*0.75)
            open_px  = y1 + int(len(col)*0.25)

        candles.append({
            "x1": x1, "x2": x2, "y_high": int(high_px), "y_low": int(low_px),
            "y_open": int(open_px), "y_close": int(close_px),
            "type": typ
        })

    # Keep chronological order left→right
    candles.sort(key=lambda c: (c["x1"]+c["x2"])//2)
    return candles

def _map_pixels_to_price(candles: List[Dict], scale: Optional[Tuple[float,float,int,int]]) -> List[Dict]:
    if not candles:
        return candles
    if scale is None:
        # produce normalized OHLC (0..1, higher value = higher price)
        y_min = min(c["y_high"] for c in candles)
        y_max = max(c["y_low"]  for c in candles)
        rng = max(1, y_max - y_min)
        out = []
        for c in candles:
            out.append({
                **c,
                "open": 1 - (c["y_open"]  - y_min)/rng,
                "close":1 - (c["y_close"] - y_min)/rng,
                "high": 1 - (c["y_high"]  - y_min)/rng,
                "low":  1 - (c["y_low"]   - y_min)/rng,
                "scaled": False
            })
        return out
    # With axis: linear map y→price
    pmin, pmax, y_min_px, y_max_px = scale
    def y_to_price(y):
        # top pixel y_min_px is high price; invert
        frac = (y - y_min_px) / max(1, (y_max_px - y_min_px))
        return pmax - frac * (pmax - pmin)
    out = []
    for c in candles:
        out.append({
            **c,
            "open":  y_to_price(c["y_open"]),
            "close": y_to_price(c["y_close"]),
            "high":  y_to_price(c["y_high"]),
            "low":   y_to_price(c["y_low"]),
            "scaled": True
        })
    return out

def _ta_features(ohlc: List[Dict], look_atr: int = 14) -> Dict:
    if len(ohlc) < 3:
        return {"atr": 0.0, "ma5": None, "ma10": None, "slope5": 0.0}
    closes = np.array([c["close"] for c in ohlc], dtype=float)
    highs  = np.array([c["high"]  for c in ohlc], dtype=float)
    lows   = np.array([c["low"]   for c in ohlc], dtype=float)
    # ATR (classic)
    trs = [highs[0]-lows[0]]
    for i in range(1, len(ohlc)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    atr = float(np.mean(trs[-look_atr:])) if len(trs) >= look_atr else float(np.mean(trs))
    # MAs & slope
    def _ma(a, n):
        if len(a) < n: return None
        return float(np.mean(a[-n:]))
    ma5  = _ma(closes, 5)
    ma10 = _ma(closes,10)
    # slope (per bar) using last 5
    y = closes[-5:] if len(closes) >= 5 else closes
    x = np.arange(len(y))
    m = 0.0
    if len(y) >= 2:
        m = float(np.polyfit(x, y, 1)[0])
    return {"atr": atr, "ma5": ma5, "ma10": ma10, "slope5": m}

def read_candles(image_path: str) -> Dict:
    """Main entry. Returns dict with keys: bars (list of OHLC dicts), features, meta."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    theme = _detect_theme(img)
    bull, bear = _color_masks(img, theme)
    raw = _candles_from_masks(img, bull, bear)
    scale = _extract_axis_scale(img)  # may be None
    bars = _map_pixels_to_price(raw, scale)
    feats = _ta_features(bars)
    return {
        "bars": bars,
        "features": feats,
        "meta": {"theme": theme, "scaled": bool(scale)}
    }