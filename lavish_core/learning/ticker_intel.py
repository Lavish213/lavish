# lavish_core/learning/ticker_intel.py
from __future__ import annotations
import os, re, time, json, math, threading, queue, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Optional: yfinance for profile/fundamentals fallback
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------- env ----------
BASE = Path(__file__).resolve().parents[2]  # repo root-ish
load_dotenv(BASE / ".env")

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDB   = os.getenv("PGDATABASE", "lavish")
PGUSER = os.getenv("PGUSER", "lavish")
PGPWD  = os.getenv("PGPASSWORD", "")

FINNHUB_KEY  = os.getenv("FINNHUB_API_KEY", "")
ALPACA_DATA  = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_KEY   = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET= os.getenv("ALPACA_SECRET_KEY", "")

TICKER_DIR   = Path(os.getenv("TICKER_TXT_DIR", "tickers"))
REFRESH_HRS  = int(os.getenv("TICKER_REFRESH_HOURS", "24"))
AUTORUN_MIN  = int(os.getenv("TICKER_AUTORUN_MINUTE", "5"))   # 00:05 local
CONF_FLOOR   = float(os.getenv("TICKER_MIN_CONF", "0.0"))

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "intel_learning.log"

logger = logging.getLogger("TickerIntel")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

VALID = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")

def _pg_conn():
    return psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDB, user=PGUSER, password=PGPWD
    )

# ---------- schema ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tickers (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    summary TEXT,
    volatility TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.0,
    last_refreshed TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ticker_learning_log (
    ts TIMESTAMP DEFAULT NOW(),
    symbol TEXT,
    source TEXT,
    event TEXT,
    detail TEXT
);
"""

UPSERT_SQL = """
INSERT INTO tickers (symbol, name, sector, industry, summary, volatility, confidence, last_refreshed)
VALUES (%(symbol)s, %(name)s, %(sector)s, %(industry)s, %(summary)s, %(volatility)s, %(confidence)s, NOW())
ON CONFLICT(symbol) DO UPDATE SET
    name=EXCLUDED.name,
    sector=EXCLUDED.sector,
    industry=EXCLUDED.industry,
    summary=EXCLUDED.summary,
    volatility=EXCLUDED.volatility,
    confidence=EXCLUDED.confidence,
    last_refreshed=NOW();
"""

SELECT_STALE_SQL = """
SELECT symbol, COALESCE(EXTRACT(EPOCH FROM (NOW() - last_refreshed))/3600.0, 1e9) AS hrs
FROM tickers
WHERE (NOW() - last_refreshed) > (%s || ' hours')::INTERVAL
ORDER BY last_refreshed NULLS FIRST
LIMIT %s;
"""

INSERT_LOG_SQL = """
INSERT INTO ticker_learning_log(symbol, source, event, detail)
VALUES (%s, %s, %s, %s);
"""

def ensure_schema():
    with _pg_conn() as con, con.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    logger.info("✅ ensured tickers schema")

# ---------- sources ----------
def get_from_finnhub(sym: str) -> Dict[str, Any]:
    if not FINNHUB_KEY:
        return {}
    url_profile = f"https://finnhub.io/api/v1/stock/profile2"
    url_metric  = f"https://finnhub.io/api/v1/stock/metric"
    p = {}
    try:
        r = requests.get(url_profile, params={"symbol": sym, "token": FINNHUB_KEY}, timeout=10)
        if r.status_code == 200:
            j = r.json()
            p.update({
                "name": j.get("name") or j.get("ticker") or "",
                "sector": j.get("finnhubIndustry") or "",
                "industry": j.get("gicsIndustry") or "",
            })
    except Exception as e:
        logger.warning(f"finnhub profile fail {sym}: {e}")

    try:
        r = requests.get(url_metric, params={"symbol": sym, "metric": "all", "token": FINNHUB_KEY}, timeout=10)
        if r.status_code == 200:
            m = (r.json() or {}).get("metric", {})
            # light heuristic for volatility
            vol = None
            if "beta" in m:
                b = m.get("beta")
                if isinstance(b, (int, float)):
                    if b < 0.8: vol="low"
                    elif b < 1.2: vol="medium"
                    else: vol="high"
            p["volatility"] = vol or p.get("volatility")
    except Exception as e:
        logger.warning(f"finnhub metric fail {sym}: {e}")

    return p

def get_from_yfinance(sym: str) -> Dict[str, Any]:
    if yf is None:
        return {}
    try:
        t = yf.Ticker(sym)
        info = {}
        # yfinance v0.2+ style access—robust to missing keys
        try:
            info = t.get_info()
        except Exception:
            info = {}
        out = {
            "name": info.get("shortName") or info.get("longName") or "",
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "summary": info.get("longBusinessSummary") or "",
        }
        # volatility proxy from historical std if needed
        try:
            hist = t.history(period="3mo", interval="1d")
            if not hist.empty:
                std = float(hist["Close"].pct_change().std() or 0)
                if std < 0.015: out["volatility"]="low"
                elif std < 0.03: out["volatility"]="medium"
                else: out["volatility"]="high"
        except Exception:
            pass
        return out
    except Exception as e:
        logger.warning(f"yfinance fail {sym}: {e}")
        return {}

def get_from_alpaca(sym: str) -> Dict[str, Any]:
    if not (ALPACA_KEY and ALPACA_SECRET):
        return {}
    # Alpaca fundamentals endpoints vary by plan; we use basic latest bar for sanity/vol proxy
    try:
        hdrs = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        r = requests.get(f"{ALPACA_DATA}/v2/stocks/{sym}/bars?timeframe=1Day&limit=15", headers=hdrs, timeout=10)
        if r.status_code != 200:
            return {}
        j = r.json()
        bars = j.get("bars", [])
        if not bars:
            return {}
        # quick realized vol proxy
        closes = [b.get("c") for b in bars if "c" in b]
        vol = None
        if len(closes) >= 5:
            import statistics
            series = [ (closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes)) if closes[i-1] ]
            s = abs(statistics.pstdev(series)) if series else 0.0
            if s < 0.015: vol="low"
            elif s < 0.03: vol="medium"
            else: vol="high"
        return {"volatility": vol} if vol else {}
    except Exception as e:
        logger.warning(f"alpaca data fail {sym}: {e}")
        return {}

# ---------- confidence engine ----------
def fuse_and_score(sym: str, sources: List[Dict[str, Any]], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fused: Dict[str, Any] = {
        "symbol": sym, "name":"", "sector":"", "industry":"", "summary":"", "volatility":None
    }
    # combine (prefer richer strings first)
    for key in ("name","sector","industry","summary","volatility"):
        for src in sources:
            v = src.get(key)
            if v:
                # keep longest most-informative field
                if not fused.get(key) or (isinstance(v, str) and len(v) > len(str(fused.get(key) or ""))):
                    fused[key] = v

    # base confidence from number of non-empty fields
    richness = sum(1 for k in ("name","sector","industry","summary") if fused.get(k))
    conf = 0.25 * min(4, richness)  # 0..1

    # cross-source agreement bonus (sector/industry/volatility)
    agree = 0
    keys = ("sector","industry","volatility")
    for k in keys:
        vals = [s.get(k) for s in sources if s.get(k)]
        if not vals: 
            continue
        # simple majority agreement
        maj = max(set(vals), key=vals.count)
        if vals.count(maj) >= 2:
            agree += 1
            fused[k] = maj
    conf += 0.15 * agree  # up to +0.45

    # persistence bonus if we already had a record (stability)
    if existing:
        conf += 0.10

    # clamp & floor
    conf = max(CONF_FLOOR, min(1.0, conf))
    fused["confidence"] = conf
    return fused

# ---------- ticker sourcing ----------
def load_all_tickers() -> List[str]:
    TICKER_DIR.mkdir(exist_ok=True, parents=True)
    files = sorted(TICKER_DIR.glob("*.txt"))
    found: List[str] = []
    for fp in files:
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            parts = re.split(r"[\s,]+", txt.upper())
            for p in parts:
                p = p.strip().upper()
                if p and VALID.match(p):
                    found.append(p)
        except Exception as e:
            logger.warning(f"cannot read {fp}: {e}")
    dedup = sorted(set(found))
    if not dedup:
        logger.warning(f"no tickers found in {TICKER_DIR} (put *.txt lists here)")
    else:
        logger.info(f"loaded {len(dedup)} tickers from {TICKER_DIR}")
    return dedup

def get_existing(con, sym: str) -> Optional[Dict[str, Any]]:
    with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM tickers WHERE symbol=%s", (sym,))
        row = cur.fetchone()
        return dict(row) if row else None

# ---------- worker pool ----------
class WorkerPool:
    def __init__(self, n: int = 6):
        self.n = max(1, n)
        self.q: "queue.Queue[str]" = queue.Queue()
        self.stop = threading.Event()

    def submit(self, sym: str):
        self.q.put(sym)

    def start(self):
        for _ in range(self.n):
            threading.Thread(target=self._run, daemon=True).start()

    def join(self):
        self.q.join()

    def shutdown(self):
        self.stop.set()

    def _run(self):
        with _pg_conn() as con:
            while not self.stop.is_set():
                try:
                    sym = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self._process_one(con, sym)
                except Exception as e:
                    logger.error(f"worker error {sym}: {e}")
                finally:
                    self.q.task_done()

    def _process_one(self, con, sym: str):
        existing = get_existing(con, sym)

        s_fin = get_from_finnhub(sym)
        s_yf  = get_from_yfinance(sym)
        s_alp = get_from_alpaca(sym)

        fused = fuse_and_score(sym, [s_fin, s_yf, s_alp], existing)
        with con.cursor() as cur:
            cur.execute(UPSERT_SQL, fused)
            cur.execute(INSERT_LOG_SQL, (sym, "fuse", "upsert", json.dumps(fused, ensure_ascii=False)[:2000]))
        con.commit()
        logger.info(f"{sym:<6} conf={fused['confidence']:.2f} sector={fused.get('sector','')!s} name={fused.get('name','')[:40]!s}")

# ---------- learning cycle ----------
def run_learning_batch(limit: int = 300, threads: int = 6):
    ensure_schema()
    tickers = load_all_tickers()
    if not tickers:
        return

    # pick stale or never-seen first
    with _pg_conn() as con, con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SELECT_STALE_SQL, (str(REFRESH_HRS), limit))
        stale = [r["symbol"] for r in cur.fetchall()]
    # include new tickers not in table yet
    to_learn = list(dict.fromkeys(stale + [t for t in tickers if t not in stale]))[:limit]
    if not to_learn:
        logger.info("nothing stale — skipping")
        return

    logger.info(f"learning {len(to_learn)} tickers (threads={threads}) …")
    pool = WorkerPool(n=threads); pool.start()
    for s in to_learn:
        pool.submit(s)
    pool.join()
    pool.shutdown()
    logger.info("batch complete")

# ---------- autorun nightly ----------
def _seconds_until_next_run(minute: int) -> int:
    now = datetime.now()
    nxt = now.replace(hour=0, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt = nxt + timedelta(days=1)
    return int((nxt - now).total_seconds())

def start_autorun(threads: int = 6, limit: int = 600):
    """
    Kick off a daemon thread that:
      - runs a batch immediately on boot (small)
      - then sleeps until HH:minute daily and runs a full batch
    """
    def _loop():
        logger.info("TickerIntel autorun thread started")
        try:
            run_learning_batch(limit=min(100, limit), threads=threads)  # warm-up
        except Exception as e:
            logger.warning(f"warm-up failed: {e}")

        while True:
            try:
                wait_s = _seconds_until_next_run(AUTORUN_MIN)
                logger.info(f"sleeping {wait_s//3600}h {(wait_s%3600)//60}m until nightly learn …")
                time.sleep(max(30, wait_s))
                run_learning_batch(limit=limit, threads=threads)
            except Exception as e:
                logger.error(f"autorun error: {e}")
                time.sleep(60)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

# ---------- CLI ----------
if __name__ == "__main__":
    # run once ad-hoc
    run_learning_batch()