#!/usr/bin/env python3
# lavish_core/superbot.py
# "All-in-one" news + OCR + memory + signals + risk + trade executor (Alpaca for execution only)
# Free sources for news; optional OCR of screenshots in raw/
# Persisted local memory via TF-IDF .pkl for chat/RAG
# Robust logging + safety rails; dry-run/live modes

from __future__ import annotations
import os, sys, re, time, json, math, csv, glob, queue, signal, logging, argparse, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

# ---------- Third-party (all pure-python, widely available) ----------
# pip install feedparser requests pandas numpy scikit-learn pillow pytesseract
import requests
import feedparser
import numpy as np
import pandas as pd

# OCR (optional)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Vector memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ---------- Logging ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("lavish.superbot")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(_ch)
_fh = logging.FileHandler("logs/superbot.log")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(_fh)

# ---------- Env / Config ----------
ENV = os.getenv("ENV", "production")
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")
BOT_REFRESH_MIN = int(os.getenv("BOT_REFRESH_INTERVAL_MIN", "60"))

# Alpaca (execution only)
ALPACA_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")  # paper by default

# Free / public news feeds (no key)
DEFAULT_FEEDS = [
    # Broad markets
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ Markets RSS
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # SEC filings feed (broad)
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=100&output=atom",
    # Crypto (free)
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    # Google News RSS (query injected at runtime for tickers)
    # e.g. https://news.google.com/rss/search?q=AAPL%20stock
]

# Optional keyed providers (you have keys in .env; we leave them here but these modules do not require them)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ---------- Storage ----------
os.makedirs("storage", exist_ok=True)
MEMORY_PKL = "storage/news_memory.pkl"
VEC_PKL = "storage/news_vectorizer.pkl"
LEDGER_CSV = "storage/trade_ledger.csv"

# ---------- Helpers ----------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def as_dollars(x: float) -> str:
    return f"${x:,.2f}"

def ensure_csv(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

# ---------- Data classes ----------
@dataclass
class NewsItem:
    id: str
    ts: str
    source: str
    title: str
    link: str
    summary: str
    symbols: List[str]

@dataclass
class TradeIntent:
    ts: str
    mode: str               # "dry" | "paper" | "live"
    symbol: str
    side: str               # "buy" | "sell"
    qty: float
    reason: str
    confidence: float       # 0..1
    max_risk_pct: float     # 0..1 per trade
    cool_down_s: int
    checks: Dict[str, Any]

@dataclass
class TradeResult:
    ok: bool
    broker_id: Optional[str]
    message: str
    intent: TradeIntent

# ---------- Memory (TF-IDF) ----------
class Memory:
    """
    Simple local vector store (TF-IDF) with persistence to .pkl
    Content sources: RSS news + OCR text + any appended notes.
    """
    def __init__(self, vec_path=VEC_PKL, mem_path=MEMORY_PKL):
        self.vec_path = vec_path
        self.mem_path = mem_path
        self.docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._load()

    def _load(self):
        if os.path.exists(self.vec_path) and os.path.exists(self.mem_path):
            try:
                with open(self.vec_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                with open(self.mem_path, "rb") as f:
                    payload = pickle.load(f)
                self.docs = payload["docs"]
                self.meta = payload["meta"]
                if self.vectorizer and self.docs:
                    self.matrix = self.vectorizer.fit_transform(self.docs)
                logger.info(f"Memory loaded: {len(self.docs)} docs")
            except Exception as e:
                logger.warning(f"Memory load failed, starting fresh: {e}")

    def _save(self):
        if self.vectorizer and self.docs is not None:
            with open(self.vec_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.mem_path, "wb") as f:
                pickle.dump({"docs": self.docs, "meta": self.meta}, f)

    def add(self, text: str, meta: Dict[str, Any]):
        if not text:
            return
        self.docs.append(text)
        self.meta.append(meta)
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(self.docs)
        self._save()

    def add_bulk(self, items: List[Tuple[str, Dict[str, Any]]]):
        if not items:
            return
        for t, m in items:
            if t:
                self.docs.append(t)
                self.meta.append(m)
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(self.docs)
        self._save()

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.vectorizer or self.matrix is None or not self.docs:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            out.append((float(sims[i]), self.meta[i]))
        return out

# ---------- News Collector (free feeds) ----------
class NewsCollector:
    def __init__(self, feeds: Optional[List[str]] = None):
        self.feeds = feeds or DEFAULT_FEEDS
        self.seen = set()   # store sha1 of link/title to avoid dupes

    def _parse_feed(self, url: str) -> List[NewsItem]:
        items: List[NewsItem] = []
        try:
            d = feedparser.parse(url)
            for e in d.entries:
                title = (e.title or "").strip()
                link = (getattr(e, "link", "") or "").strip()
                summary = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
                ts = getattr(e, "published", "") or getattr(e, "updated", "") or utcnow().isoformat()
                key = sha1(f"{url}|{title}|{link}")
                if key in self.seen:
                    continue
                self.seen.add(key)
                syms = self._extract_symbols(title + " " + summary)
                items.append(NewsItem(
                    id=key, ts=str(ts), source=url, title=title, link=link, summary=summary, symbols=syms
                ))
        except Exception as ex:
            logger.warning(f"Feed failed {url}: {ex}")
        return items

    def _extract_symbols(self, text: str) -> List[str]:
        # Simple heuristic: uppercase words 1-5 chars that look like tickers
        cands = re.findall(r"\b[A-Z]{1,5}\b", text.upper())
        # Common stop words to reduce false positives
        bad = {"THE","AND","FOR","WITH","FROM","THIS","THAT","WILL","WON'T","WERE","WAS","HAS","HAVE","USD","NEWS"}
        cands = [c for c in cands if c not in bad]
        return sorted(set(cands))

    def collect(self, symbols: Optional[List[str]] = None) -> List[NewsItem]:
        urls = list(self.feeds)
        # Add Google News RSS per symbol if provided
        if symbols:
            for s in symbols:
                q = requests.utils.quote(f"{s} stock")
                urls.append(f"https://news.google.com/rss/search?q={q}")
        all_items: List[NewsItem] = []
        for u in urls:
            all_items.extend(self._parse_feed(u))
        return all_items

# ---------- OCR Ingestor ----------
class OCRIngestor:
    def __init__(self, raw_dir="raw"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)

    def ingest(self) -> List[Tuple[str, Dict[str, Any]]]:
        if not OCR_AVAILABLE:
            logger.info("OCR not available (pytesseract/Pillow missing). Skipping.")
            return []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.raw_dir, e)))
        out = []
        for path in files:
            try:
                img = Image.open(path)
                txt = pytesseract.image_to_string(img)
                meta = {"source": "ocr", "path": path, "ts": utcnow().isoformat()}
                out.append((txt, meta))
            except Exception as e:
                logger.warning(f"OCR failed {path}: {e}")
        return out

# ---------- Simple Fourier/Cycle signal (on externally supplied prices) ----------
class FourierSignals:
    @staticmethod
    def _lowpass(x: np.ndarray, cutoff_period: int = 80) -> np.ndarray:
        n = len(x)
        X = np.fft.rfft(x.astype(float))
        freqs = np.fft.rfftfreq(n, d=1.0)
        cutoff_freq = 1.0 / max(2, cutoff_period)
        X[freqs > cutoff_freq] = 0.0
        return np.fft.irfft(X, n=n)

    @staticmethod
    def _z(a: np.ndarray) -> np.ndarray:
        mu, sd = np.nanmean(a), np.nanstd(a)
        if sd < 1e-12:
            return np.zeros_like(a)
        return (a - mu) / sd

    @staticmethod
    def quick_features(prices: List[float]) -> Dict[str, float]:
        if len(prices) < 120:
            return {"trend_z": 0.0, "risk": 0.0}
        x = np.array(prices, float)
        trend = FourierSignals._lowpass(x, cutoff_period=80)
        slope = np.gradient(trend)
        tz = FourierSignals._z(slope[-60:])[-1]
        ext = x[-1] - trend[-1]
        extz = FourierSignals._z((x[-60:] - trend[-60:]))[-1]
        # heuristic risk 0..1
        risk = float(clamp(0.15*abs(tz) + 0.85*abs(extz), 0.0, 3.0)/3.0)
        return {"trend_z": float(tz), "risk": risk}

# ---------- Risk & Policy ----------
class Policy:
    def __init__(self):
        self.cooldowns: Dict[str, datetime] = {}
        self.block_until: Optional[datetime] = None
        self.daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "0"))  # $; 0 disables
        self.max_pos_pct = float(os.getenv("MAX_POSITION_PCT", "0.15"))    # of equity
        self.max_risk_pct_default = float(os.getenv("MAX_RISK_PCT", "0.01"))

    def global_circuit(self) -> bool:
        if self.block_until and utcnow() < self.block_until:
            return True
        return False

    def trip_circuit(self, minutes=30):
        self.block_until = utcnow() + timedelta(minutes=minutes)
        logger.warning(f"[CIRCUIT] Global trading paused for {minutes} min")

    def in_cooldown(self, symbol: str) -> bool:
        t = self.cooldowns.get(symbol)
        return t is not None and utcnow() < t

    def set_cooldown(self, symbol: str, secs: int):
        self.cooldowns[symbol] = utcnow() + timedelta(seconds=secs)

# ---------- Ledger ----------
class Ledger:
    def __init__(self, csv_path=LEDGER_CSV):
        self.csv_path = csv_path
        ensure_csv(self.csv_path, ["ts","mode","symbol","side","qty","price","broker_id","ok","message","reason","confidence","checks_json"])
    def write(self, res: TradeResult, price: float):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([utcnow().isoformat(), res.intent.mode, res.intent.symbol, res.intent.side,
                        res.intent.qty, price, (res.broker_id or ""), int(res.ok), res.message,
                        res.intent.reason, res.intent.confidence, json.dumps(res.intent.checks)])

# ---------- Alpaca Executor (execution only; no data fetching here) ----------
class AlpacaExecutor:
    def __init__(self, base_url=ALPACA_BASE_URL, key=ALPACA_KEY, secret=ALPACA_SECRET, timeout=15):
        self.base = base_url.rstrip("/")
        self.key = key
        self.secret = secret
        self.timeout = timeout

    def _headers(self):
        return {"APCA-API-KEY-ID": self.key or "", "APCA-API-SECRET-KEY": self.secret or ""}

    def _orders_url(self):
        return f"{self.base}/orders"

    def place_order(self, symbol: str, side: str, qty: float, tif="day", order_type="market") -> Tuple[bool, str, Optional[str]]:
        try:
            payload = {"symbol": symbol, "side": side, "type": order_type, "time_in_force": tif, "qty": str(qty)}
            r = requests.post(self._orders_url(), headers=self._headers(), json=payload, timeout=self.timeout)
            if r.status_code in (200, 201):
                oid = r.json().get("id")
                return True, "submitted", oid
            else:
                return False, f"{r.status_code} {r.text}", None
        except Exception as e:
            return False, str(e), None

# ---------- Trader orchestrator ----------
class Trader:
    def __init__(self, mode="dry"):
        self.mode = mode  # dry | paper | live
        self.policy = Policy()
        self.ledger = Ledger()
        self.alpaca = AlpacaExecutor()

    def decide_size(self, price: float, equity: float, max_pos_pct: float) -> float:
        # simple sizing: dollar cap
        max_dollar = equity * max_pos_pct
        qty = math.floor(max(1, max_dollar / max(1e-6, price)))
        return float(qty)

    def run_checks(self, intent: TradeIntent, last_price: float, equity: float) -> Tuple[bool, Dict[str, Any]]:
        checks = dict(intent.checks)

        # global circuit
        if self.policy.global_circuit():
            checks["global_circuit"] = "blocked"
            return False, checks

        # cooldown
        if self.policy.in_cooldown(intent.symbol):
            checks["cooldown"] = "active"
            return False, checks

        # price sanity
        if last_price <= 0:
            checks["price"] = "invalid"
            return False, checks

        # position sizing
        theoretical_qty = self.decide_size(last_price, equity, max_pos_pct=checks.get("max_pos_pct", 0.15))
        if intent.qty > theoretical_qty * 1.2:  # intent qty should not wildly exceed cap
            checks["size_cap"] = f"intent {intent.qty} > cap {theoretical_qty}"
            return False, checks

        # risk allowance
        mrp = intent.max_risk_pct or self.policy.max_risk_pct_default
        if mrp <= 0 or mrp > 0.05:
            checks["max_risk_pct"] = "out_of_bounds"
            return False, checks

        # confidence gate
        if intent.confidence < 0.5:
            checks["confidence"] = f"too_low({intent.confidence:.2f})"
            return False, checks

        checks["ok"] = True
        return True, checks

    def execute(self, intent: TradeIntent, last_price: float, equity: float) -> TradeResult:
        ok, checks = self.run_checks(intent, last_price, equity)
        intent.checks = checks
        if not ok:
            msg = "checks_failed"
            logger.info(f"[SKIP] {intent.symbol} {intent.side} {intent.qty} ({msg}) | checks={checks}")
            return TradeResult(False, None, msg, intent)

        # set cooldown
        self.policy.set_cooldown(intent.symbol, intent.cool_down_s or 60)

        if self.mode == "dry":
            logger.info(f"[DRY] {intent.symbol} {intent.side} x{intent.qty} @ ~{as_dollars(last_price)} | reason={intent.reason}")
            return TradeResult(True, "DRY-RUN", "dry_ok", intent)

        if self.mode in ("paper", "live"):
            if not (ALPACA_KEY and ALPACA_SECRET):
                return TradeResult(False, None, "alpaca_keys_missing", intent)
            ok, msg, oid = self.alpaca.place_order(intent.symbol, intent.side, intent.qty)
            if ok:
                logger.info(f"[{self.mode.upper()}] Submitted {intent.side} {intent.symbol} x{intent.qty} | oid={oid}")
                return TradeResult(True, oid, "submitted", intent)
            else:
                logger.error(f"[{self.mode.upper()}] Order error: {msg}")
                # soft circuit on repeated errors
                self.policy.trip_circuit(10)
                return TradeResult(False, None, msg, intent)

        return TradeResult(False, None, "unknown_mode", intent)

# ---------- Orchestrator ----------
class SuperBot:
    def __init__(self, mode="dry"):
        self.mode = mode
        self.mem = Memory()
        self.news = NewsCollector()
        self.ocr = OCRIngestor()
        self.trader = Trader(mode=mode)

    # PUBLIC: chat retrieval (for your chat layer to call)
    def ask(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        return self.mem.search(query, k=k)

    # PUBLIC: add dev note to memory
    def note(self, text: str, tag="note"):
        self.mem.add(text, {"source": tag, "ts": utcnow().isoformat()})

    # Pipeline steps
    def step_collect_news(self, symbols: Optional[List[str]] = None) -> List[NewsItem]:
        items = self.news.collect(symbols=symbols)
        if not items:
            return []
        # push to memory
        pairs = []
        for it in items:
            pack = f"{it.title}\n{it.summary}\n{it.link}"
            pairs.append((pack, {"source": "rss", "title": it.title, "link": it.link, "ts": it.ts, "symbols": it.symbols}))
        self.mem.add_bulk(pairs)
        logger.info(f"News collected: {len(items)} items")
        return items

    def step_ingest_ocr(self) -> int:
        pairs = self.ocr.ingest()
        if not pairs:
            return 0
        self.mem.add_bulk(pairs)
        logger.info(f"OCR ingested: {len(pairs)} files")
        return len(pairs)

    def step_signals(self, symbol: str, recent_prices: List[float]) -> Dict[str, float]:
        sig = FourierSignals.quick_features(recent_prices)
        # store brief signal snapshot to memory for traceability
        self.mem.add(json.dumps({"symbol": symbol, "signal": sig}, indent=2),
                     {"source": "signal", "symbol": symbol, "ts": utcnow().isoformat()})
        return sig

    def step_decide_and_trade(self, symbol: str, side: str, price: float, equity: float,
                              confidence: float, reason: str, cool_down_s=90,
                              max_pos_pct=0.15, max_risk_pct=0.01, qty_override: Optional[float]=None) -> TradeResult:
        qty = qty_override if qty_override is not None else self.trader.decide_size(price, equity, max_pos_pct)
        intent = TradeIntent(
            ts=utcnow().isoformat(), mode=self.mode, symbol=symbol, side=side, qty=float(qty),
            reason=reason, confidence=confidence, max_risk_pct=max_risk_pct, cool_down_s=cool_down_s,
            checks={"max_pos_pct": max_pos_pct}
        )
        res = self.trader.execute(intent, last_price=price, equity=equity)
        self.trader.ledger.write(res, price)
        return res

# ---------- CLI ----------
def build_parser():
    p = argparse.ArgumentParser(description="Lavish SuperBot – news/OCR/memory + trade executor (Alpaca exec only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) Run everything in a loop (no data from Alpaca; only execution if mode != dry)
    r = sub.add_parser("run", help="Run pipeline loop")
    r.add_argument("--mode", choices=["dry","paper","live"], default="dry")
    r.add_argument("--symbols", type=str, default="", help="Comma list e.g. AAPL,MSFT; adds per-ticker RSS searches")
    r.add_argument("--interval", type=int, default=300, help="Seconds between loops")
    r.add_argument("--equity", type=float, default=25000.0, help="Account equity estimate for sizing")
    r.add_argument("--max-pos-pct", type=float, default=0.15)
    r.add_argument("--max-risk-pct", type=float, default=0.01)

    # 2) One-shot news pull
    n = sub.add_parser("news", help="Collect news once and index to memory")
    n.add_argument("--symbols", type=str, default="")

    # 3) OCR ingest from raw/
    o = sub.add_parser("ocr", help="OCR ingest images in raw/ into memory")

    # 4) Ask memory
    a = sub.add_parser("ask", help="Query memory (TF-IDF)")
    a.add_argument("--q", required=True)
    a.add_argument("--k", type=int, default=5)

    # 5) Simulate a trade decision & (possibly) execute
    t = sub.add_parser("trade", help="Decide + (dry/paper/live) execute a trade")
    t.add_argument("--mode", choices=["dry","paper","live"], default="dry")
    t.add_argument("--symbol", required=True)
    t.add_argument("--side", choices=["buy","sell"], required=True)
    t.add_argument("--price", type=float, required=True)
    t.add_argument("--equity", type=float, default=25000.0)
    t.add_argument("--confidence", type=float, required=True)
    t.add_argument("--reason", type=str, default="manual")
    t.add_argument("--cooldown", type=int, default=90)
    t.add_argument("--max-pos-pct", type=float, default=0.15)
    t.add_argument("--max-risk-pct", type=float, default=0.01)
    t.add_argument("--qty", type=float, default=None)

    # 6) Add a note to memory
    m = sub.add_parser("note", help="Store a developer/operator note into memory")
    m.add_argument("--text", required=True)

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "news":
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if getattr(args, "symbols", "") else []
        bot = SuperBot(mode="dry")
        items = bot.step_collect_news(symbols=syms)
        logger.info(f"Saved {len(items)} news items into memory")

    elif args.cmd == "ocr":
        bot = SuperBot(mode="dry")
        n = bot.step_ingest_ocr()
        logger.info(f"OCR -> {n} items")

    elif args.cmd == "ask":
        bot = SuperBot(mode="dry")
        hits = bot.ask(args.q, k=args.k)
        for score, meta in hits:
            logger.info(f"{score:.3f} :: {meta}")

    elif args.cmd == "trade":
        bot = SuperBot(mode=args.mode)
        res = bot.step_decide_and_trade(
            symbol=args.symbol, side=args.side, price=args.price, equity=args.equity,
            confidence=args.confidence, reason=args.reason, cool_down_s=args.cooldown,
            max_pos_pct=args.max_pos_pct, max_risk_pct=args.max_risk_pct, qty_override=args.qty
        )
        logger.info(f"TRADE RESULT ok={res.ok} msg={res.message} oid={res.broker_id}")

    elif args.cmd == "run":
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if getattr(args, "symbols", "") else []
        bot = SuperBot(mode=args.mode)
        logger.info(f"RUN loop start | mode={args.mode} interval={args.interval}s symbols={syms}")
        # Basic example loop: pull news/OCR, (optionally) issue toy trade if a simple keyword hits
        # You should wire this to your Patreon/“stock lady” triggers and math pipeline.
        try:
            while True:
                # 1) collect
                bot.step_collect_news(symbols=syms)
                bot.step_ingest_ocr()

                # 2) example: if “upgrade” appears for a watched symbol, place a guarded small buy (confidence 0.6)
                if syms:
                    query = " OR ".join([f"{s} upgrade OR beats OR guidance" for s in syms])
                    hits = bot.ask(query, k=3)
                    if hits and hits[0][0] > 0.25:  # crude relevance threshold
                        pick = syms[0]
                        price_guess = 100.0  # replace with your own price feed
                        bot.step_decide_and_trade(
                            symbol=pick, side="buy", price=price_guess, equity=args.equity,
                            confidence=0.6, reason="news_hit", cool_down_s=120,
                            max_pos_pct=args.max_pos_pct, max_risk_pct=args.max_risk_pct, qty_override=1.0
                        )

                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")

if __name__ == "__main__":
    main()