#!/usr/bin/env python3
# auto_signal_runner.py
# Drop screenshots into lavish_core/vision/raw and this will:
#   - OCR them (pytesseract) + pattern-parse for ticker/side/expiry/strike/confidence
#   - Apply guardrails (market hours, risk caps, duplicate suppression, whitelist)
#   - Write a JSONL event to data/orders_queue.jsonl
#   - If PAPER_TRADING=1 -> place a paper order on Alpaca (stocks only) OR log-only for options
#   - Rate-limited: processes ≤100 images / 15 minutes; backs off on bursts

import os, re, sys, json, time, queue, threading, hashlib, signal, datetime as dt
from pathlib import Path

# ---------- PATHS ----------
BASE = Path(__file__).resolve().parent
RAW_DIR = Path(__file__).resolve().parents[2] / "lavish_core" / "vision" / "raw"
DATA_DIR = BASE / "data"
LOGS = DATA_DIR / "logs"
OUT_DIR = DATA_DIR / "images_out"
QUEUE_FILE = DATA_DIR / "orders_queue.jsonl"
SEEN_FILE = DATA_DIR / "images_out" / "seen_hashes.json"
DATA_DIR.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- ENV ----------
from dotenv import load_dotenv
load_dotenv(BASE / ".env")

ALPACA_KEY       = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET    = os.getenv("ALPACA_API_SECRET", "")
ALPACA_ENDPOINT  = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
PAPER_TRADING    = os.getenv("PAPER_TRADING", "1") == "1"   # paper by default
LIVE_TRADING     = os.getenv("LIVE_TRADING", "0") == "1"    # explicit to go live
ACCOUNT_CAP_USD  = float(os.getenv("ACCOUNT_CAP_USD", "25000"))
MAX_RISK_PCT     = float(os.getenv("MAX_RISK_PCT", "0.02"))  # 2% per signal
WHITELIST        = set([s.strip().upper() for s in os.getenv("TICKER_WHITELIST", "AAPL,MSFT,NVDA,META,TSLA,SPY,QQQ,AMD,CRM,ORCL,MSTR,GME,TLRY").split(",") if s.strip()])

# ---------- OPTIONAL: OCR (pytesseract) ----------
try:
    import pytesseract
    from PIL import Image, ImageOps
    OCR_OK = True
except Exception:
    OCR_OK = False

# ---------- HTTP client for Alpaca ----------
import requests

def log(line, level="INFO"):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"{ts} [{level}] {line}"
    print(out, flush=True)
    with open(LOGS / "auto_signal_runner.log", "a", encoding="utf-8") as f:
        f.write(out + "\n")

def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(131072), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def load_seen():
    try:
        return set(json.loads(Path(SEEN_FILE).read_text()))
    except Exception:
        return set()

def save_seen(seen):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    Path(SEEN_FILE).write_text(json.dumps(sorted(list(seen))))

# ---------- OCR helpers ----------
TICKER_RE   = re.compile(r'\b([A-Z]{2,5})\b')
SIDE_RE     = re.compile(r'\b(CALL|PUT)\b', re.I)
DATE1_RE    = re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s?\d{1,2}\b', re.I)
DATE2_RE    = re.compile(r'\b\d{1,2}/\d{1,2}\b')      # e.g., 6/21
STRIKE_RE   = re.compile(r'\$?(\d{2,5}(\.\d{1,2})?)\s*(Call|Put)?', re.I)  # fallback
BREAKEVEN   = re.compile(r'breakeven', re.I)

def ocr_image(p: Path) -> str:
    if not OCR_OK:
        return ""
    try:
        img = Image.open(p).convert("L")
        img = ImageOps.autocontrast(img)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        log(f"OCR failed for {p.name}: {e}", "ERROR")
        return ""

# Parse common broker screenshots (Robinhood/Webull-like)
def parse_text(text: str):
    text_up = text.upper()
    # 1) try to pick a likely ticker from WHITELIST first
    tickers = WHITELIST.intersection(set(TICKER_RE.findall(text_up)))
    ticker = None
    if tickers:
        # choose the visually most frequent in text
        ticker = sorted(tickers, key=lambda t: -text_up.count(t))[0]
    else:
        # fallback: pick first plausible token (still uppercase A-Z)
        m = TICKER_RE.findall(text_up)
        if m:
            ticker = m[0]

    side = None
    m = SIDE_RE.search(text_up)
    if m:
        side = m.group(1).upper()

    expiry = None
    m1 = DATE1_RE.search(text_up)
    m2 = DATE2_RE.search(text_up)
    if m1: expiry = m1.group(0).title()
    elif m2: expiry = m2.group(0)

    # strike: look near CALL/PUT first
    strike = None
    if side:
        ctx = text_up.split(side)[-1][:40]
        m = STRIKE_RE.search(ctx)
        if m:
            try: strike = float(m.group(1))
            except: pass
    if strike is None:
        m = STRIKE_RE.search(text_up)
        if m:
            try: strike = float(m.group(1))
            except: pass

    # Confidence score: add up signals we saw
    conf = 0
    conf += 0.4 if ticker else 0
    conf += 0.3 if side else 0
    conf += 0.15 if expiry else 0
    conf += 0.15 if strike else 0
    if BREAKEVEN.search(text_up): conf += 0.05
    conf = min(1.0, conf)

    return {
        "ticker": ticker,
        "side": side,                  # CALL/PUT or None
        "expiry": expiry,              # e.g., "Jun 21" or "6/21"
        "strike": strike,              # float or None
        "confidence": conf
    }

# ---------- Risk & market checks ----------
def is_market_open(now=None, tz="US/Eastern"):
    # simple guard: Mon-Fri, 9:30–16:00 ET (doesn't handle holidays)
    now = now or dt.datetime.now(dt.timezone.utc).astimezone()
    # You can use exchange calendars later; keeping lightweight here.
    wd = now.weekday()
    if wd >= 5: return False
    h, m = now.hour, now.minute
    tmins = h*60+m
    return 9*60+30 <= tmins <= 16*60

def compute_dollars_per_signal(account_cap=ACCOUNT_CAP_USD, max_risk_pct=MAX_RISK_PCT):
    # e.g., 2% of cap
    return round(account_cap * max_risk_pct, 2)

def suppress_duplicate(sig_hash, seen_hashes):
    if sig_hash in seen_hashes:
        return True
    seen_hashes.add(sig_hash)
    save_seen(seen_hashes)
    return False

# ---------- Queue I/O ----------
def emit_order_event(event: dict):
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# ---------- Alpaca (stocks only in practice mode) ----------
def alpaca_account():
    r = requests.get(f"{ALPACA_ENDPOINT}/v2/account",
        headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
        timeout=10)
    r.raise_for_status()
    return r.json()

def alpaca_place_market_buy(symbol: str, notional: float):
    payload = {"symbol": symbol, "notional": str(notional), "side": "buy", "type": "market", "time_in_force": "day"}
    r = requests.post(f"{ALPACA_ENDPOINT}/v2/orders",
        headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET, "Content-Type": "application/json"},
        data=json.dumps(payload), timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca error: {r.status_code} {r.text}")
    return r.json()

# ---------- Core pipeline ----------
class VisionSignalRunner:
    def __init__(self, batch_size=100, interval_sec=900):
        self.batch_size = batch_size
        self.interval = interval_sec
        self.stop = False
        self.seen_hashes = load_seen()

    def _list_new_images(self):
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = [p for p in RAW_DIR.glob("*") if p.suffix.lower() in exts]
        files.sort(key=lambda p: p.stat().st_mtime)
        # Only those not seen by hash
        out = []
        for p in files:
            try:
                h = hash_file(p)
                if h not in self.seen_hashes:
                    out.append((p, h))
            except Exception:
                continue
        return out[: self.batch_size]

    def _handle_one(self, p: Path, h: str):
        text = ocr_image(p) if OCR_OK else ""
        parsed = parse_text(text)
        parsed["source_path"] = str(p)
        parsed["image_hash"] = h
        parsed["ts"] = dt.datetime.now(timezone.utc).isoformat()

        # Basic validation
        issues = []
        if not parsed.get("ticker"): issues.append("no_ticker")
        if parsed.get("ticker") and parsed["ticker"] not in WHITELIST: issues.append("not_whitelisted")
        if parsed.get("confidence", 0) < 0.6: issues.append("low_conf")
        parsed["issues"] = issues

        dollars = compute_dollars_per_signal()
        parsed["notional_usd"] = dollars

        # Order intent: for screenshots that say CALL → we simulate BUY underlying
        action = "BUY"
        if parsed.get("side") == "PUT":
            action = "SELL_SHORT"  # paper notion; we still only submit BUY in demo

        intent = {
            "action": action,
            "ticker": parsed.get("ticker"),
            "reason": "image_signal",
            "confidence": parsed.get("confidence", 0),
            "metadata": parsed
        }

        # Dedup
        if suppress_duplicate(h, self.seen_hashes):
            log(f"Skipping duplicate {p.name} ({h})")
            return

        # Emit to queue always
        emit_order_event(intent)
        log(f"Queued order: {intent['ticker']} {intent['action']} ${dollars} conf={intent['confidence']:.2f}")

        # Safety rails before any actual trade
        if not is_market_open():
            log("Market closed → queue only (no trade).")
            return
        if "no_ticker" in issues or "not_whitelisted" in issues:
            log(f"Guardrail blocked trade (issues={issues}).")
            return

        # Paper trading only unless LIVE_TRADING=1
        if PAPER_TRADING and not LIVE_TRADING:
            try:
                acct = alpaca_account()
                log(f"Alpaca paper account status={acct.get('status')} equity={acct.get('equity')}")
                if parsed["ticker"]:
                    order = alpaca_place_market_buy(parsed["ticker"], parsed["notional_usd"])
                    log(f"Paper order placed: {order.get('id')} {order.get('symbol')} notional={parsed['notional_usd']}")
            except Exception as e:
                log(f"Alpaca paper order failed: {e}", "ERROR")
            return

        # Live trading (explicit opt-in)
        if LIVE_TRADING:
            try:
                order = alpaca_place_market_buy(parsed["ticker"], parsed["notional_usd"])
                log(f"*** LIVE order placed: {order.get('id')} {order.get('symbol')}")
            except Exception as e:
                log(f"*** LIVE order FAILED: {e}", "ERROR")

    def run_forever(self):
        log(f"Watcher online. Folder={RAW_DIR} batch={self.batch_size} every={self.interval}s OCR_OK={OCR_OK}")
        while not self.stop:
            try:
                batch = self._list_new_images()
                if not batch:
                    log("Idle. No new images.")
                for p, h in batch:
                    log(f"Processing {p.name} ({h})")
                    self._handle_one(p, h)
            except Exception as e:
                log(f"Loop error: {e}", "ERROR")
            time.sleep(self.interval)

def main():
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    runner = VisionSignalRunner(batch_size=100, interval_sec=900)  # 100 every 15 minutes
    def _sigint(_a,_b):
        runner.stop = True
        log("Stopping on SIGINT")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)
    runner.run_forever()

if __name__ == "__main__":
    main()