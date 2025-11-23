#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lavish Bot – Brain Master (knowledge + reasoning + enrichment)
- SQLite memory store
- News/RSS/YouTube ingestion
- Signal enrichment (vision/patreon)
- Confidence scoring with transparent features
- CLI: ingest once, watch, and/or enrich queue

Hard deps kept light; optional deps guarded.
"""

import os, sys, re, json, time, math, csv, argparse, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sqlite3
import threading

# ---------- Paths & folders ----------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

DB_PATH          = DATA / "brain.db"
QUEUE_IN_PATH    = DATA / "orders_queue.json"      # vision/patreon writes here (JSON list or JSONL)
QUEUE_OUT_PATH   = DATA / "enriched_orders.jsonl"  # trader reads this
ARTICLES_CSV     = DATA / "news_articles.csv"
VIDEOS_CSV       = DATA / "yt_videos.csv"
SCORES_CSV       = DATA / "signal_scores.csv"
YOUTUBE_WATCH_TXT= DATA / "youtube_watchlist.txt"  # list of YouTube video IDs to fetch transcripts

# ---------- ENV ----------
def getenv(key, default=None):
    v = os.getenv(key, default)
    return v if (v is not None and str(v).strip() != "") else default

# Core env config
WHITELIST = set([s.strip().upper() for s in getenv("WHITELIST_TICKERS", "AAPL,MSFT,AMD,NVDA,META,TSLA,SPY,QQQ,GOOGL,CRM,MSTR").split(",") if s.strip()])
NEWSAPI_KEY = getenv("NEWSAPI_KEY")                 # optional
RSS_URLS    = [u.strip() for u in getenv("RSS_URLS", "").split(",") if u.strip()]
TRUSTED_SOURCES = set([s.strip().lower() for s in getenv("TRUSTED_SOURCES", "wsj.com,bloomberg.com,ft.com,reuters.com,cnbc.com,seekingalpha.com").split(",") if s.strip()])
YOUTUBE_OK  = getenv("YOUTUBE_OK", "1") == "1"      # allow transcripts if present
PAPER_TRADING = getenv("PAPER_TRADING", "1") == "1" # just for reference (trader handles actual placing)
REFRESH_MIN  = int(getenv("BRAIN_REFRESH_MIN", "15"))
RECENCY_HALF_LIFE_H = float(getenv("RECENCY_HALF_LIFE_H", "12"))  # news recency decay
MAX_ARTICLES_PER_TICKER = int(getenv("MAX_ARTICLES_PER_TICKER", "200"))

# ---------- Optional imports (guarded) ----------
try:
    import requests
except Exception:
    requests = None

try:
    import feedparser  # RSS/Atom
except Exception:
    feedparser = None

# Simple sentiment (vader)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
except Exception:
    VADER = None

# YouTube transcripts (no API key needed for public transcripts)
try:
    from youtube_transcript_api import YouTubeTranscriptApi as YT
except Exception:
    YT = None

# ---------- DB ----------
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS articles(
  id INTEGER PRIMARY KEY,
  source TEXT,
  url TEXT UNIQUE,
  title TEXT,
  published_at TEXT,
  text TEXT,
  tickers TEXT,           -- comma list
  sentiment REAL,
  trust REAL,
  recency REAL,           -- 0..1 after decay
  score REAL,             -- sentiment*trust*recency
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS videos(
  id INTEGER PRIMARY KEY,
  video_id TEXT UNIQUE,
  channel TEXT,
  title TEXT,
  published_at TEXT,
  transcript TEXT,
  tickers TEXT,
  sentiment REAL,
  trust REAL,
  recency REAL,
  score REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY,
  raw_json TEXT,
  source TEXT,             -- "vision" | "patreon" | "manual"
  ticker TEXT,
  side TEXT,               -- CALL/PUT/BUY/SELL_SHORT (intent)
  expiry TEXT,
  strike REAL,
  confidence REAL,         -- parser confidence
  enriched_score REAL,     -- final decision score
  notional_usd REAL,
  reason TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def db_init():
    con = db()
    with con:
        for stmt in SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
    con.close()

# ---------- Utils ----------
TICKER_RE = re.compile(r"\b([A-Z]{1,5})(?=$|[^a-zA-Z])")
STRONG_TICKER_HINTS = {"call", "put", "strike", "exp", "exp.", "breakeven", "premium", "weekly"}

def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    cand = set()
    for m in TICKER_RE.finditer(text.upper()):
        t = m.group(1)
        if t in WHITELIST:
            cand.add(t)
    # light boost: if context contains options language, keep; else keep anyway (whitelist already guards)
    return sorted(cand)

def sent(text: str) -> float:
    if not text:
        return 0.0
    if VADER:
        s = VADER.polarity_scores(text)["compound"]
        return float(s)  # -1..1
    # fallback: weak heuristic
    pos = len(re.findall(r"\b(beat|surge|rise|upgrade|wins|strong|record|profit|above|growth|bull|breakout)\b", text.lower()))
    neg = len(re.findall(r"\b(miss|drop|fall|downgrade|loss|below|cut|bear|delay|risk)\b", text.lower()))
    return max(-1.0, min(1.0, (pos - neg) / 5.0))

def trust_for_source(url_or_domain: str) -> float:
    domain = url_or_domain.lower()
    if "://" in domain:
        domain = domain.split("://", 1)[-1]
    domain = domain.split("/", 1)[0]
    if domain in TRUSTED_SOURCES:
        return 1.0
    # light default
    return 0.6

def recency_weight(published_at_iso: str, half_life_h: float = RECENCY_HALF_LIFE_H) -> float:
    try:
        t = dt.datetime.fromisoformat(published_at_iso.replace("Z", "+00:00"))
    except Exception:
        try:
            t = dt.datetime.strptime(published_at_iso, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return 0.5
    age_h = (dt.datetime.now(dt.timezone.utc) - t.astimezone(dt.timezone.utc)).total_seconds() / 3600.0
    if age_h <= 0:
        return 1.0
    # exponential decay
    return 0.5 ** (age_h / half_life_h)

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

# ---------- News ingestion ----------
def newsapi_fetch(tickers: List[str]) -> List[Dict]:
    if not NEWSAPI_KEY or not requests:
        return []
    out = []
    # NewsAPI: https://newsapi.org (developer must have key; respects their ToS)
    # We query per ticker (simple + transparent)
    base = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": NEWSAPI_KEY}
    for t in tickers:
        params = {
            "q": f'"{t}"',
            "pageSize": 20,
            "language": "en",
            "sortBy": "publishedAt"
        }
        try:
            r = requests.get(base, params=params, headers=headers, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
            for a in data.get("articles", []):
                out.append({
                    "source": a.get("source", {}).get("name") or "",
                    "url": a.get("url") or "",
                    "title": a.get("title") or "",
                    "published_at": a.get("publishedAt") or dt.datetime.now(timezone.utc).isoformat(),
                    "text": (a.get("description") or "") + "\n" + (a.get("content") or ""),
                })
        except Exception:
            continue
    return out

def rss_fetch(urls: List[str]) -> List[Dict]:
    if not urls or not feedparser:
        return []
    out = []
    for url in urls:
        try:
            d = feedparser.parse(url)
            for e in d.entries:
                out.append({
                    "source": e.get("link", "").split("/")[2] if e.get("link") else (d.feed.get("title") or "rss"),
                    "url": e.get("link") or "",
                    "title": e.get("title") or "",
                    "published_at": e.get("published", dt.datetime.now(timezone.utc).isoformat()),
                    "text": e.get("summary", ""),
                })
        except Exception:
            continue
    return out

def upsert_articles(rows: List[Dict]):
    con = db()
    with con:
        for r in rows:
            tix = extract_tickers(f"{r.get('title','')} {r.get('text','')}")
            tix_csv = ",".join(tix)
            s = sent(f"{r.get('title','')}. {r.get('text','')}")
            tr = trust_for_source(r.get("source","") or r.get("url",""))
            rc = recency_weight(str(r.get("published_at","")))
            score = (s+1)/2 * tr * rc  # map sentiment -1..1 -> 0..1
            con.execute("""
              INSERT OR IGNORE INTO articles(source,url,title,published_at,text,tickers,sentiment,trust,recency,score)
              VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (r.get("source",""), r.get("url",""), r.get("title",""), str(r.get("published_at","")),
                  r.get("text",""), tix_csv, float(s), float(tr), float(rc), float(score)))
    con.close()

# ---------- YouTube transcripts ----------
def load_youtube_watchlist() -> List[str]:
    if not YOUTUBE_OK or not YT:
        return []
    if not YOUTUBE_WATCH_TXT.exists():
        return []
    ids = []
    for line in YOUTUBE_WATCH_TXT.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    if not YT:
        return None
    try:
        parts = YT.get_transcript(video_id, languages=["en"])
        return " ".join([p["text"] for p in parts if p.get("text")])
    except Exception:
        return None

def upsert_videos(video_ids: List[str]):
    if not video_ids:
        return
    con = db()
    with con:
        for vid in video_ids:
            # if present, skip
            cur = con.execute("SELECT 1 FROM videos WHERE video_id=?", (vid,))
            if cur.fetchone():
                continue
            tx = fetch_youtube_transcript(vid) or ""
            # we don't fetch meta via API here; let title/channel be placeholders
            title = f"YT:{vid}"
            channel = "unknown"
            published_at = dt.datetime.now(timezone.utc).isoformat()
            tix = extract_tickers(tx)
            s = sent(tx)
            tr = 0.7  # default trust for curated watchlist
            rc = recency_weight(published_at)
            score = (s+1)/2 * tr * rc
            con.execute("""
              INSERT OR IGNORE INTO videos(video_id,channel,title,published_at,transcript,tickers,sentiment,trust,recency,score)
              VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (vid, channel, title, published_at, tx, ",".join(tix), float(s), float(tr), float(rc), float(score)))
    con.close()

# ---------- Query: context for a ticker ----------
def context_for_ticker(ticker: str, limit=30) -> Dict:
    con = db()
    res = {"articles": [], "videos": [], "agg": {"avg_score":0,"n":0}}
    try:
        arts = con.execute("""
          SELECT source,url,title,published_at,score FROM articles
          WHERE instr(tickers, ?) > 0
          ORDER BY score DESC, published_at DESC LIMIT ?
        """, (ticker, limit)).fetchall()
        vids = con.execute("""
          SELECT video_id,title,published_at,score FROM videos
          WHERE instr(tickers, ?) > 0
          ORDER BY score DESC, published_at DESC LIMIT ?
        """, (ticker, limit)).fetchall()
        res["articles"] = [{"source":a[0], "url":a[1], "title":a[2], "published_at":a[3], "score":a[4]} for a in arts]
        res["videos"]   = [{"video_id":v[0], "title":v[1], "published_at":v[2], "score":v[3]} for v in vids]
        all_scores = [a[4] for a in arts] + [v[3] for v in vids]
        if all_scores:
            res["agg"]["avg_score"] = sum(all_scores) / len(all_scores)
            res["agg"]["n"] = len(all_scores)
    finally:
        con.close()
    return res

# ---------- Signal enrichment & scoring ----------
def read_queue(path: Path) -> List[dict]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    items = []
    # tolerate JSON list or JSONL
    try:
        if txt.startswith("["):
            items = json.loads(txt)
        else:
            for line in txt.splitlines():
                if line.strip():
                    items.append(json.loads(line))
    except Exception:
        # best-effort line-by-line
        items = []
        for line in txt.splitlines():
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def write_jsonl(path: Path, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")

def composite_score(signal: dict, cx: Dict) -> float:
    """
    Final 0..1 score blending:
      w_signal: parser confidence & structure (ticker/side/expiry/strike present)
      w_news:   avg score in knowledge base (news/videos)
      w_trust:  source trust if present
      w_time:   recentness of signal itself
    """
    # signal bits
    conf = float(signal.get("metadata", {}).get("confidence", signal.get("confidence", 0.0)) or 0.0)
    # structure bonus: did we parse core fields?
    bonus = 0.0
    if signal.get("ticker"): bonus += 0.1
    if signal.get("metadata", {}).get("expiry"): bonus += 0.05
    if signal.get("metadata", {}).get("strike"): bonus += 0.05
    sig_part = clamp01(conf + bonus)  # ~0..1

    # news/videos avg (already 0..1)
    news_avg = float(cx.get("agg", {}).get("avg_score", 0.5) or 0.5)

    # trust of signal source (patreon/vision)
    src = (signal.get("source") or signal.get("reason") or "").lower()
    src_trust = 0.8 if "patreon" in src else 0.7 if "vision" in src else 0.6

    # recency of signal timestamp
    ts = signal.get("metadata", {}).get("ts") or signal.get("ts") or dt.datetime.now(timezone.utc).isoformat()
    r = recency_weight(ts, half_life_h=2.0)  # signals decay faster

    # blend
    w_signal, w_news, w_trust, w_time = 0.4, 0.35, 0.15, 0.10
    final = w_signal*sig_part + w_news*news_avg + w_trust*src_trust + w_time*r
    return clamp01(final)

def enrich_signals():
    signals = read_queue(QUEUE_IN_PATH)
    if not signals:
        print("[INFO] No signals to enrich.")
        return

    con = db()
    with con:
        for s in signals:
            ticker = (s.get("ticker") or s.get("metadata", {}).get("ticker") or "").upper()
            if not ticker or ticker not in WHITELIST:
                continue
            cx = context_for_ticker(ticker)
            score = composite_score(s, cx)
            enriched = {
                "ticker": ticker,
                "action": s.get("action") or s.get("side") or "BUY",
                "notional_usd": float(s.get("notional_usd", 0)) or 0.0,
                "enriched_score": score,
                "source": s.get("source") or s.get("reason") or "unknown",
                "context": cx,
                "metadata": s.get("metadata", s),
                "ts": dt.datetime.now(timezone.utc).isoformat()
            }
            # persist to DB (signals table) + JSONL for trader
            con.execute("""
              INSERT INTO signals(raw_json,source,ticker,side,expiry,strike,confidence,enriched_score,notional_usd,reason)
              VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (json.dumps(s), enriched["source"], ticker, enriched["action"],
                  s.get("metadata", {}).get("expiry"), s.get("metadata", {}).get("strike"),
                  float(s.get("metadata", {}).get("confidence", 0.0)),
                  float(score), float(enriched["notional_usd"]), "brain_enriched"))

            write_jsonl(QUEUE_OUT_PATH, enriched)
            print(f"[OK] Enriched {ticker} → score={score:.2f}")
    con.close()

# ---------- CSV exports ----------
def export_csvs():
    con = db()
    with con:
        with open(ARTICLES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source","url","title","published_at","tickers","sentiment","trust","recency","score"])
            for row in con.execute("SELECT source,url,title,published_at,tickers,sentiment,trust,recency,score FROM articles"):
                w.writerow(row)
        with open(VIDEOS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id","title","published_at","tickers","sentiment","trust","recency","score"])
            for row in con.execute("SELECT video_id,title,published_at,tickers,sentiment,trust,recency,score FROM videos"):
                w.writerow(row)
        with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","source","ticker","side","enriched_score","notional_usd","created_at"])
            for row in con.execute("SELECT id,source,ticker,side,enriched_score,notional_usd,created_at FROM signals ORDER BY id DESC"):
                w.writerow(row)
    print(f"[CSV] Wrote {ARTICLES_CSV.name}, {VIDEOS_CSV.name}, {SCORES_CSV.name}")

# ---------- Orchestrations ----------
def ingest_once():
    # 1) NewsAPI
    print("[INGEST] NewsAPI...")
    arts = newsapi_fetch(sorted(WHITELIST))
    if arts:
        upsert_articles(arts)
        print(f"[INGEST] NewsAPI added ~{len(arts)}")

    # 2) RSS
    if RSS_URLS:
        print("[INGEST] RSS...")
        rss = rss_fetch(RSS_URLS)
        if rss:
            upsert_articles(rss)
            print(f"[INGEST] RSS added ~{len(rss)}")

    # 3) YouTube transcripts
    vids = load_youtube_watchlist()
    if vids:
        print(f"[INGEST] YouTube transcripts for {len(vids)} IDs…")
        upsert_videos(vids)

    # Optional pruning for DB size: keep last MAX_ARTICLES_PER_TICKER per ticker
    prune_articles()

def prune_articles():
    con = db()
    with con:
        # naive prune by ticker frequency
        for t in WHITELIST:
            cur = con.execute("""
              SELECT id FROM articles WHERE instr(tickers, ?) > 0
              ORDER BY published_at DESC, id DESC
            """, (t,))
            ids = [r[0] for r in cur.fetchall()]
            if len(ids) > MAX_ARTICLES_PER_TICKER:
                drop = ids[MAX_ARTICLES_PER_TICKER:]
                q = "DELETE FROM articles WHERE id IN ({})".format(",".join("?"*len(drop)))
                con.execute(q, drop)
    con.close()

def watch_forever(refresh_min=REFRESH_MIN):
    print(f"[WATCH] Brain refresh every {refresh_min} min.")
    while True:
        try:
            ingest_once()
            enrich_signals()
            export_csvs()
        except Exception as e:
            print(f"[ERROR] watch loop: {e}")
        time.sleep(refresh_min * 60)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(prog="brain_master", description="Lavish Bot – Knowledge & Reasoning Hub")
    parser.add_argument("--ingest-once", action="store_true", help="Fetch news/RSS/YouTube and update DB once")
    parser.add_argument("--enrich", action="store_true", help="Enrich current signals queue now")
    parser.add_argument("--watch", action="store_true", help="Run forever (ingest+enrich+export each cycle)")
    parser.add_argument("--refresh-min", type=int, default=REFRESH_MIN, help="Minutes between cycles in watch mode")
    args = parser.parse_args()

    db_init()

    if args.ingest_once:
        ingest_once()
    if args.enrich:
        enrich_signals()
        export_csvs()
    if args.watch or (not args.ingest_once and not args.enrich):
        watch_forever(args.refresh_min)

if __name__ == "__main__":
    main()
