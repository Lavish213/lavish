import os
import re
import glob
import json
import time
import traceback
import requests
import sqlite3
import pandas as pd
import yfinance as yf

from datetime import timedelta
from typing import List, Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------- constants ---------------- #

TICKER_DIR = "tickers"
YF_NEWS_LIMIT = 10
REDDIT_ENABLED = True
REDDIT_SUBS = ["stocks", "wallstreetbets", "investing"]
REDDIT_LIMIT = 10
OPEN_FETCH_MINUTES = 390
CLOSED_FETCH_MINUTES = 1440
OPEN_LOOP_SECONDS = 60 * 15
CLOSED_LOOP_SECONDS = 60 * 60
UA = "Mozilla/5.0 (compatible; LavishBot/1.0; +https://github.com/lavish)"

# ---------------- helpers ---------------- #

def connect_db(path: str = "lavish_bot.db") -> sqlite3.Connection:
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def short_summary(t: str, n: int = 120) -> str:
    t = clean_text(t)
    return t[:n] + ("…" if len(t) > n else "")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def et_now():
    return pd.Timestamp.utcnow().tz_convert("US/Eastern")

def is_market_open(et):
    return et.weekday() < 5 and (9 <= et.hour < 16)

# ---------------- schema ---------------- #

def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS bars (
        ts INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        source TEXT NOT NULL DEFAULT 'yfinance',
        PRIMARY KEY (ts, ticker, interval)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS news_feed (
        id TEXT PRIMARY KEY,
        ticker TEXT,
        source TEXT,          -- 'yf' or 'reddit'
        title TEXT,
        url TEXT,
        publisher TEXT,
        published_ts INTEGER,
        sentiment REAL,       -- compound
        summary TEXT,
        raw JSON
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ticker_logs (
        ts INTEGER,
        ticker TEXT,
        source TEXT,
        status TEXT,
        note TEXT
    );
    """)

    con.commit()

# ---------------- tickers ---------------- #

VALID = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")

def load_all_tickers() -> List[str]:
    os.makedirs(TICKER_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TICKER_DIR, "*.txt")))
    found: List[str] = []
    for fp in files:
        try:
            txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
            parts = re.split(r"[\s,]+", txt.upper())
            for p in parts:
                p = p.strip().upper()
                if p and VALID.match(p):
                    found.append(p)
        except Exception as e:
            print(f"Could not read {fp}: {e}")
    dedup = sorted(set(found))
    if not dedup:
        print(f"No tickers found in {TICKER_DIR} (put *.txt here)")
    else:
        print(f"Loaded {len(dedup)} tickers")
    return dedup

# ---------------- prices (yfinance) ---------------- #

def fetch_prices_yf(tickers: List[str], minutes: int) -> pd.DataFrame:
    if minutes <= 390:
        interval, period = "1m", "1d"
    elif minutes <= 7*24*60:
        interval, period = "5m", "5d"
    else:
        interval, period = "1h", "60d"

    out_rows = []
    for batch in chunked(tickers, 50):
        try:
            df = yf.download(batch, period=period, interval=interval, auto_adjust=False, progress=False, threads=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.stack(level=1).reset_index().rename(columns={"level_1": "Ticker"})
                df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close",
                    "Adj Close": "adj_close", "Volume": "volume", "Datetime": "Datetime"
                }, inplace=True)
            else:
                df = df.reset_index()
                df["Ticker"] = batch[0]
                df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", 
                    "Adj Close": "adj_close", "Volume": "volume",
                }, inplace=True)
            tcol = "Datetime" if "Datetime" in df.columns else "Date"
            df["ts"] = pd.to_datetime(df[tcol], utc=True).astype("int64") // 10**9
            df["ticker"] = df["Ticker"].astype(str)
            df["interval"] = interval
            take = df[["ts","ticker","interval","open","high","low","close","volume"]].dropna(how="any")
            out_rows.append(take)
        except Exception as e:
            print(f"yfinance batch failed ({len(batch)} tickers): {e}")

    if not out_rows:
        return pd.DataFrame(columns=["ts","ticker","interval","open","high","low","close","volume"])
    return pd.concat(out_rows, ignore_index=True).sort_values(["ticker","ts"])

def upsert_prices(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cur = con.cursor()
    rows = df.to_records(index=False)
    cur.executemany("""
    INSERT INTO bars (ts, ticker, interval, open, high, low, close, volume, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'yfinance')
    ON CONFLICT(ts, ticker, interval) DO UPDATE SET
       open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume
    """, list(rows))
    con.commit()
    return cur.rowcount

# ---------------- news (yfinance) ---------------- #

def fetch_yf_news_for(ticker: str) -> List[Dict]:
    out = []
    try:
        items = getattr(yf.Ticker(ticker), "news", []) or []
        for it in items[:YF_NEWS_LIMIT]:
            it = dict(it)
            it["ticker"] = ticker
            out.append(it)
    except Exception as e:
        print(f"YF news error {ticker}: {e}")
    return out

# ---------------- reddit (no auth) ---------------- #

def reddit_search(sub: str, query: str, limit: int) -> List[Dict]:
    url = f"https://www.reddit.com/r/{sub}/search.json"
    params = {"q": query, "restrict_sr": 1, "sort": "new", "limit": limit, "t": "day"}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("data", {}).get("children", [])
        out = []
        for ch in items:
            d = ch.get("data", {})
            out.append({
                "id": f"reddit_{d.get('id')}",
                "title": d.get("title"),
                "url": "https://www.reddit.com" + d.get("permalink", ""),
                "publisher": f"r/{sub}",
                "published_ts": int(d.get("created_utc", 0)),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0)
            })
        return out
    except Exception:
        return []

def fetch_reddit_for(ticker: str) -> List[Dict]:
    results = []
    q = f'"{ticker}" OR ${ticker}'
    for sub in REDDIT_SUBS:
        results.extend(reddit_search(sub, q, REDDIT_LIMIT))
    seen = set()
    uniq = []
    for item in results:
        k = (item["title"], item["url"])
        if k not in seen:
            seen.add(k)
            uniq.append(item)
    return uniq

# ---------------- sentiment ---------------- #

_VADER = SentimentIntensityAnalyzer()
POS_WORDS = {"beat","upgrade","strong","record","surge","bullish","profit","guidance raised","buyback"}
NEG_WORDS = {"miss","downgrade","weak","lawsuit","probe","fraud","dilution","guidance cut","bankruptcy"}

def score_sentiment(text: str) -> float:
    s = _VADER.polarity_scores(text or "")
    comp = s.get("compound", 0.0)
    lt = text.lower()
    for w in POS_WORDS:
        if w in lt: comp += 0.03
    for w in NEG_WORDS:
        if w in lt: comp -= 0.03
    return max(-1.0, min(1.0, comp))

# ---------------- upsert news ---------------- #

def upsert_news(con: sqlite3.Connection, ticker: str, yf_items: List[Dict], reddit_items: List[Dict]) -> int:
    cur = con.cursor()
    total = 0

    # Yahoo Finance
    for it in yf_items:
        uid = str(it.get("uuid") or it.get("id") or f"yf_{ticker}_{it.get('providerPublishTime',0)}_{hash(it.get('title',''))}")
        title = clean_text(it.get("title",""))
        url = it.get("link") or it.get("url") or ""
        pub  = it.get("publisher","Yahoo")
        ts   = int(it.get("providerPublishTime", 0))
        sent = score_sentiment(title)
        summ = short_summary(title)
        raw  = json.dumps(it)[:2000]
        try:
            cur.execute("""
            INSERT INTO news_feed (id, ticker, source, title, url, publisher, published_ts, sentiment, summary, raw)
            VALUES (?, ?, 'yf', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
               title=excluded.title, url=excluded.url, publisher=excluded.publisher,
               published_ts=excluded.published_ts, sentiment=excluded.sentiment, summary=excluded.summary
            """, (uid, ticker, title, url, pub, ts, sent, summ, raw))
            total += cur.rowcount
        except Exception as e:
            print(f"news upsert (yf) skip: {e}")

    # Reddit
    for it in reddit_items:
        uid = it["id"]
        title = clean_text(it.get("title",""))
        url = it.get("url","")
        pub = it.get("publisher","reddit")
        ts = int(it.get("published_ts", 0))
        sent = score_sentiment(title)
        summ = short_summary(title)
        raw = json.dumps(it)[:2000]
        try:
            cur.execute("""
            INSERT INTO news_feed (id, ticker, source, title, url, publisher, published_ts, sentiment, summary, raw)
            VALUES (?, ?, 'reddit', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
               title=excluded.title, url=excluded.url, publisher=excluded.publisher,
               published_ts=excluded.published_ts, sentiment=excluded.sentiment, summary=excluded.summary
            """, (uid, ticker, title, url, pub, ts, sent, summ, raw))
            total += cur.rowcount
        except Exception as e:
            print(f"news upsert (reddit) skip: {e}")

    con.commit()
    return total

# ---------------- main ingest loop ---------------- #

def ingest_once(con: sqlite3.Connection, tickers: List[str]) -> None:
    et = et_now()
    market_open = is_market_open(et)
    minutes = OPEN_FETCH_MINUTES if market_open else CLOSED_FETCH_MINUTES

    print(f"Market {'OPEN' if market_open else 'CLOSED'} → fetching last {minutes} minutes for {len(tickers)} tickers")

    df = fetch_prices_yf(tickers, minutes=minutes)
    added = upsert_prices(con, df)
    print(f"Prices upserted: {added} rows")

    news_total = 0
    for t in tickers:
        try:
            yf_items = fetch_yf_news_for(t)
            rd_items = fetch_reddit_for(t) if REDDIT_ENABLED else []
            n = upsert_news(con, t, yf_items, rd_items)
            news_total += n
        except Exception as e:
            print(f"news ingest failed for {t}: {e}")
    print(f"News upserted: {news_total} items")

def loop_forever():
    con = connect_db()
    ensure_schema(con)
    tickers = load_all_tickers()
    if not tickers:
        print("No tickers to process, exiting.")
        return