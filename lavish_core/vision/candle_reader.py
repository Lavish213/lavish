# lavish_core/vision/candle_reader.py
# Wall-Street Edition: robust candle extraction from chart images.
# - Handles light/dark themes, adaptive thresholding, deskew, color calibration
# - Detects bodies & wicks; estimates OHLC per bar
# - Y-axis scale via OCR (lavish_core.vision.ocr if present; otherwise Paddle/Tesseract fallback)
# - Outputs JSONL bars; supports single-run and folder watch loop

from __future__ import annotations
import os, sys, cv2, re, json, time, math, glob, logging
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# =========[ SECTION: Logging ]=========
logger = logging.getLogger("candle_reader")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# =========[ SECTION: Stable Paths ]=========
_THIS = Path(__file__).resolve()
_LAVISH_CORE = _THIS.parent.parent
_DATA = (_LAVISH_CORE / "data").resolve()

IN_DIR  = (_DATA / "images_in").resolve()
OUT_DIR = (_DATA / "images_processed").resolve()
JSONL_OUT = (_DATA / "parsed_bars.jsonl").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========[ SECTION: Optional OCR import ]=========
def _ocr_try(image_gray: np.ndarray) -> str:
    """Try project OCR → PaddleOCR → Tesseract, return raw text (may be empty)."""
    # 1) Project-local OCR if exists
    try:
        from lavish_core.vision.ocr import run_ocr as _project_ocr
        try:
            txt = _project_ocr(None, pre_img=image_gray)  # our ocr accepts pre_img per your setup
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    except Exception:
        pass

    # 2) PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_gray, cls=True) or []
        lines = []
        for page in result:
            if not page:
                continue
            for item in page:
                # item: [ [[x1,y1],[x2,y2]...], (text,conf) , ... ]
                if isinstance(item, list) and len(item) >= 2:
                    t = item[1][0]
                    if t:
                        lines.append(t)
        return "\n".join(lines).strip()
    except Exception:
        pass

    # 3) Tesseract
    try:
        import pytesseract
        return pytesseract.image_to_string(image_gray, lang="eng") or ""
    except Exception:
        return ""

# =========[ SECTION: Data Structures ]=========
@dataclass
class CandleBar:
    index: int
    ts_hint: Optional[str]  # optional text time (if seen on x-axis)
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    theme: str              # 'dark'|'light' (detected)
    body_top_px: int
    body_bottom_px: int
    wick_top_px: int
    wick_bottom_px: int
    body_left_px: int
    body_right_px: int
    confidence: float       # heuristic 0..1

# =========[ SECTION: Preprocessing ]=========
def _deskew(gray: np.ndarray) -> np.ndarray:
    try:
        # Binary inverse for text/grid detection
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(255 - thr)
        if coords is None:
            return gray
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return gray

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _theme(img: np.ndarray) -> str:
    # Simple brightness heuristic
    gray = _to_gray(img)
    mean_val = float(np.mean(gray))
    return "dark" if mean_val < 110 else "light"

def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _prep(image_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    th = _theme(img)
    gray = _to_gray(img)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    gray = _normalize_contrast(gray)
    gray = _deskew(gray)
    return img, gray, th

# =========[ SECTION: Chart Area Detection ]=========
def _largest_rect_mask(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    # Find the largest rectangular-ish contour (chart pane)
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return np.ones_like(gray, dtype=np.uint8)*255, (0,0,w,h)  # fallback: full image
    best = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(best)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
    return mask, (x,y,w,h)

# =========[ SECTION: Body/Wick Segmentation ]=========
def _segment_bodies(img: np.ndarray, chart_roi: np.ndarray, theme: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (body_mask, wick_mask) within chart ROI using color+shape cues.
    """
    x,y,w,h = chart_roi
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros(img.shape[:2], np.uint8), np.zeros(img.shape[:2], np.uint8)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Broad ranges for green/up & red/down candles (tolerant to themes)
    green1 = cv2.inRange(hsv, (35, 25, 40), (85, 255, 255))
    red1a  = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    red1b  = cv2.inRange(hsv, (160, 40, 40), (179, 255, 255))
    red1 = cv2.bitwise_or(red1a, red1b)

    # Bodies tend to be thicker; wicks very thin vertical lines.
    body_mask_crop = cv2.bitwise_or(green1, red1)

    # Wick detection by vertical line emphasis
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    k = np.array([[0,-1,0],
                  [0, 4,0],
                  [0,-1,0]])
    highpass = cv2.filter2D(gray, -1, k)
    edges = cv2.Canny(highpass, 50, 150)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
    wick_mask_crop = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vert_kernel, iterations=1)

    # Map back to full image coords
    body_mask = np.zeros(img.shape[:2], np.uint8)
    wick_mask = np.zeros(img.shape[:2], np.uint8)
    body_mask[y:y+h, x:x+w] = body_mask_crop
    wick_mask[y:y+h, x:x+w] = wick_mask_crop
    return body_mask, wick_mask

# =========[ SECTION: Column Grouping → Candles ]=========
def _extract_columns(body_mask: np.ndarray, chart_roi: Tuple[int,int,int,int]) -> List[Tuple[int,int]]:
    """
    Returns list of (x_start, x_end) columns for each candle body cluster inside ROI.
    """
    x,y,w,h = chart_roi
    crop = body_mask[y:y+h, x:x+w]
    if crop.size == 0:
        return []

    # Reduce noise, then connected components on body blobs
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    den = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(den, connectivity=8)
    columns = []
    for i in range(1, num_labels):
        x0, y0, ww, hh, area = stats[i]
        if area < 20 or ww < 2 or hh < 5:
            continue
        # treat each blob as a candle body region
        columns.append((x + x0, x + x0 + ww - 1))
    columns = sorted(columns, key=lambda t: t[0])
    return columns

# =========[ SECTION: Per-Candle Vertical Extents ]=========
def _vert_bounds(mask: np.ndarray, x0: int, x1: int) -> Tuple[int,int]:
    """
    Returns (top_px, bottom_px) where mask is nonzero in this x-range.
    """
    col = mask[:, x0:x1+1]
    ys = np.where(np.any(col > 0, axis=1))[0]
    if len(ys) == 0:
        return -1, -1
    return int(ys.min()), int(ys.max())

# =========[ SECTION: Y-axis Scale Recovery ]=========
def _estimate_scale(gray: np.ndarray, chart_roi: Tuple[int,int,int,int]) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to OCR y-axis labels near left/right of chart to map pixels→price.
    Returns (px_per_unit, price_at_top) or (None,None) if not found.
    """
    x,y,w,h = chart_roi
    H, W = gray.shape
    # Look 0..x (left gutter) and (x+w)..W (right gutter) for numbers
    left_gutter = gray[y:y+h, max(0, x-120):x]
    right_gutter = gray[y:y+h, x+w:min(W, x+w+120)]

    candidates = []
    for gut in [left_gutter, right_gutter]:
        if gut.size == 0: 
            continue
        gut = _normalize_contrast(gut)
        gut_text = _ocr_try(gut)
        if gut_text:
            # pull numeric labels
            nums = re.findall(r"[-]?\d+(?:\.\d+)?", gut_text.replace(",", ""))
            # rough guess using first & last numeric token positions
            if len(nums) >= 2:
                candidates.append((gut_text, nums))

    if not candidates:
        return None, None

    # Heuristic: parse two extreme values and assume linear scale
    # We don’t know exact pixel positions for labels (OCR doesn’t return coords here),
    # so we fallback to unit-per-pixel from value spread vs axis height.
    # It’s approximate but serviceable for relative OHLC in price units.
    # Take the widest numeric spread we see.
    best = None
    best_spread = 0.0
    for _, nums in candidates:
        vals = [float(n) for n in nums]
        spread = max(vals) - min(vals)
        if spread > best_spread:
            best_spread = spread
            best = (spread, min(vals), max(vals))

    if not best or best_spread <= 0:
        return None, None

    spread, vmin, vmax = best
    px_per_unit = h / (vmax - vmin + 1e-9)
    price_at_top = vmax  # assume top corresponds to vmax
    return px_per_unit, price_at_top

def _px_to_price(px_y: int, chart_roi: Tuple[int,int,int,int], px_per_unit: Optional[float], price_at_top: Optional[float]) -> Optional[float]:
    if px_per_unit is None or price_at_top is None:
        return None
    _, y, _, _ = chart_roi
    dy = px_y - y  # pixels from top of chart
    price = price_at_top - (dy / (px_per_unit + 1e-9))
    return float(price)

# =========[ SECTION: Main Extraction ]=========
def extract_candles(image_path: Path) -> Dict[str, Any]:
    img, gray, theme = _prep(image_path)
    mask, roi = _largest_rect_mask(gray)
    x,y,w,h = roi
    logger.info(f"[{image_path.name}] chart ROI = (x={x}, y={y}, w={w}, h={h}) theme={theme}")

    body_mask, wick_mask = _segment_bodies(img, roi, theme)
    columns = _extract_columns(body_mask, roi)
    logger.info(f"[{image_path.name}] detected {len(columns)} candle columns")

    # Try to estimate y scale
    px_per_unit, price_top = _estimate_scale(gray, roi)

    candles: List[CandleBar] = []
    for i, (cx0, cx1) in enumerate(columns):
        btop, bbot = _vert_bounds(body_mask, cx0, cx1)
        wtop, wbot = _vert_bounds(wick_mask, cx0, cx1)
        if btop < 0 or bbot < 0:
            # Try wick-only bars as thin bodies (rare)
            if wtop >= 0 and wbot >= 0:
                btop, bbot = wtop+2, wbot-2
            else:
                continue

        # We don’t know time; ts_hint can be added in future with x-axis OCR
        ts_hint = None

        # OHLC heuristic from body/wick geometry:
        body_height = max(1, bbot - btop)
        wick_top = wtop if wtop >= 0 else btop
        wick_bottom = wbot if wbot >= 0 else bbot

        # Determine green/up vs red/down by sampling the body color mid-column
        midx = int((cx0 + cx1) / 2)
        midy = int((btop + bbot) / 2)
        color = img[min(max(midy,0), img.shape[0]-1), min(max(midx,0), img.shape[1]-1)]
        # simplistic: green if G channel is greatest, red if R greatest
        up = int(color[1]) > int(color[2])

        # Map pixels → price if possible
        open_px = bbot if up else btop
        close_px = btop if up else bbot
        high_px = min(wick_top, btop)
        low_px  = max(wick_bottom, bbot)

        open_p  = _px_to_price(open_px, roi, px_per_unit, price_top)
        close_p = _px_to_price(close_px, roi, px_per_unit, price_top)
        high_p  = _px_to_price(high_px, roi, px_per_unit, price_top)
        low_p   = _px_to_price(low_px,  roi, px_per_unit, price_top)

        # Confidence heuristic
        span = (cx1 - cx0) + 1
        conf = 0.35
        if span >= 3: conf += 0.15
        if wick_top >= 0 and wick_bottom >= 0: conf += 0.15
        if body_height >= 6: conf += 0.10
        if open_p is not None and close_p is not None: conf += 0.15
        conf = float(max(0.0, min(1.0, conf)))

        candles.append(CandleBar(
            index=i, ts_hint=ts_hint,
            open=open_p, high=high_p, low=low_p, close=close_p,
            theme=theme,
            body_top_px=int(btop), body_bottom_px=int(bbot),
            wick_top_px=int(wick_top), wick_bottom_px=int(wick_bottom),
            body_left_px=int(cx0), body_right_px=int(cx1),
            confidence=conf
        ))

    # Save a debug overlay image (optional)
    overlay = img.copy()
    for c in candles:
        cv2.rectangle(overlay, (c.body_left_px, c.body_top_px), (c.body_right_px, c.body_bottom_px), (0,255,255), 1)
        cv2.line(overlay, (int((c.body_left_px+c.body_right_px)/2), c.wick_top_px),
                           (int((c.body_left_px+c.body_right_px)/2), c.wick_bottom_px), (255,0,255), 1)
    dbg_path = (OUT_DIR / f"{image_path.stem}_overlay.png")
    try:
        cv2.imwrite(str(dbg_path), overlay)
    except Exception:
        pass

    return {
        "image": str(image_path),
        "roi": {"x":x,"y":y,"w":w,"h":h},
        "theme": theme,
        "px_per_unit": px_per_unit,
        "price_at_top": price_top,
        "bars": [asdict(c) for c in candles],
        "debug_overlay": str(dbg_path),
    }

# =========[ SECTION: IO helpers ]=========
def _append_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _list_images(folder: Path) -> List[Path]:
    exts = (".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")
    files = []
    for ext in exts:
        files += list(folder.rglob(f"*{ext}"))
    # Remove hidden / temp
    return sorted([p for p in files if not p.name.startswith(".")])

# =========[ SECTION: CLI ]=========
def run_once(in_dir: Path, out_jsonl: Path) -> None:
    imgs = _list_images(in_dir)
    if not imgs:
        logger.warning(f"No images found in {in_dir}")
        return
    recs = []
    for p in imgs:
        try:
            rec = extract_candles(p)
            recs.append(rec)
            logger.info(f"Parsed {p.name}: {len(rec['bars'])} bars  →  overlay saved")
        except Exception as e:
            logger.error(f"{p.name} failed: {e}")
    if recs:
        _append_jsonl(recs, out_jsonl)
        logger.info(f"Wrote {len(recs)} records → {out_jsonl}")

def watch_loop(in_dir: Path, out_jsonl: Path, poll_sec: int = 2) -> None:
    """Continuously watch for new images and parse them."""
    seen: set[str] = set()
    logger.info(f"Watching {in_dir} every {poll_sec}s … Ctrl+C to stop")
    while True:
        try:
            imgs = _list_images(in_dir)
            new_ones = [p for p in imgs if str(p) not in seen]
            if new_ones:
                recs=[]
                for p in new_ones:
                    try:
                        rec = extract_candles(p)
                        recs.append(rec)
                        seen.add(str(p))
                        logger.info(f"[NEW] {p.name}: {len(rec['bars'])} bars")
                    except Exception as e:
                        logger.error(f"{p.name} failed: {e}")
                if recs:
                    _append_jsonl(recs, out_jsonl)
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Lavish Candle Reader (Wall-Street Edition)")
    ap.add_argument("--in", dest="in_dir", default=str(IN_DIR), help="Folder with chart images")
    ap.add_argument("--out", dest="out_jsonl", default=str(JSONL_OUT), help="Output JSONL path")
    ap.add_argument("--watch", action="store_true", help="Watch folder continuously")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()
    in_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.watch:
        watch_loop(in_dir, out_jsonl)
    else:
        run_once(in_dir, out_jsonl)
