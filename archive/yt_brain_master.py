#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yt_brain_master.py — Wall-Street grade YouTube Intelligence (single file)
------------------------------------------------------------------------
- Topic/channel search (YouTube Data API v3)
- Transcript ingest (youtube_transcript_api)
- Sentiment (VADER if available, else TextBlob, else neutral)
- Ticker/keyword extraction + credibility-weighted scoring
- Parallel fetch w/ robust retries & rate limiting
- Persistence: CSV + SQLite (dedup by videoId)
- CLI + watch mode (cronless loop)
- All side-effects contained; safe to run repeatedly

ENV (.env or process):
  YOUTUBE_API_KEY=...
  YT_CHANNEL_WHITELIST="Bloomberg Television,CNBC Television,The Plain Bagel"
  YT_MAX_RESULTS=10
  YT_DEFAULT_TOPICS="SPY,NVDA,AAPL,TSLA,AMD,rate hikes,earnings,AI stocks"
  YT_STORE_CSV="yt_intel_memory.csv"
  YT_STORE_SQLITE="yt_intel_memory.db"

Install (once):
  pip install python-dotenv google-api-python-client youtube-transcript-api
  pip install vaderSentiment textblob # optional; one of them is fine

Run examples:
  python yt_brain_master.py --topics NVDA,SPY --max 12 --days 7 --enrich
  python yt_brain_master.py --channels "Bloomberg Television,CNBC Television" --max 5 --enrich
  python yt_brain_master.py --watch --refresh-min 15 --topics TSLA
"""

from __future__ import annotations
import os, re, csv, sys, time, json, math, queue, sqlite3, signal, random
import argparse
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---------- Optional deps & fallbacks ----------
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = lambda *a, **k: None

try:
    from googleapiclient.discovery import build as ytb_build   # YouTube Data API
except Exception:
    ytb_build = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi    # transcripts
except Exception:
    YouTubeTranscriptApi = None

# Sentiment backends (prefer VADER -> TextBlob -> neutral)
_SENT_BACKEND = "neutral"
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _SENT_BACKEND = "vader"
except Exception:
    try:
        from textblob import TextBlob
        _SENT_BACKEND = "textblob"
    except Exception:
        _SENT_BACKEND = "neutral"

# ---------- Paths & ENV ----------
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

CSV_PATH  = Path(os.getenv("YT_STORE_CSV", "yt_intel_memory.csv")).resolve()
DB_PATH   = Path(os.getenv("YT_STORE_SQLITE", "yt_intel_memory.db")).resolve()
MAX_RESULTS_DEFAULT = int(os.getenv("YT_MAX_RESULTS", "10"))
DEFAULT_TOPICS = [t.strip() for t in os.getenv(
    "YT_DEFAULT_TOPICS",
    "SPY,NVDA,AAPL,TSLA,AMD,rate hikes,earnings,AI stocks"
).split(",") if t.strip()]

CHANNEL_WHITELIST = [c.strip() for c in os.getenv(
    "YT_CHANNEL_WHITELIST",
    "Bloomberg Television,CNBC Television,The Plain Bagel"
).split(",") if c.strip()]

# ---------- Utility ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def backoff_sleep(attempt: int, base: float = 0.75, cap: float = 12.0):
    time.sleep(min(cap, base * (2 ** attempt) + random.uniform(0, 0.25)))

def safe_print(*a, **k):
    try:
        print(*a, **k)
    except Exception:
        try:
            s = " ".join(str(x) for x in a)
            sys.stdout.buffer.write((s + "\n").encode("utf-8", "replace"))
        except Exception:
            pass

def log(msg: str, level="INFO"):
    safe_print(f"{utcnow_iso()} [{level}] {msg}")

# ---------- Storage (CSV + SQLite) ----------
CSV_HEADER = [
    "timestamp","videoId","published","channel","title","url",
    "sentiment","keywords","tickers","topic","score","credibility","duration_s"
]

def ensure_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS yt_intel (
        videoId TEXT PRIMARY KEY,
        timestamp TEXT,
        published TEXT,
        channel TEXT,
        title TEXT,
        url TEXT,
        sentiment REAL,
        keywords TEXT,
        tickers TEXT,
        topic TEXT,
        score REAL,
        credibility REAL,
        duration_s INTEGER
      )
    """)
    conn.commit()
    conn.close()

def sqlite_upsert(row: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO yt_intel(videoId,timestamp,published,channel,title,url,
                           sentiment,keywords,tickers,topic,score,credibility,duration_s)
      VALUES(:videoId,:timestamp,:published,:channel,:title,:url,
             :sentiment,:keywords,:tickers,:topic,:score,:credibility,:duration_s)
      ON CONFLICT(videoId) DO UPDATE SET
        timestamp=excluded.timestamp,
        published=excluded.published,
        channel=excluded.channel,
        title=excluded.title,
        url=excluded.url,
        sentiment=excluded.sentiment,
        keywords=excluded.keywords,
        tickers=excluded.tickers,
        topic=excluded.topic,
        score=excluded.score,
        credibility=excluded.credibility,
        duration_s=excluded.duration_s
    """, row)
    conn.commit()
    conn.close()

def csv_append(row: Dict[str, Any]):
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})

# ---------- NLP helpers ----------
# Stock tickers: conservative regex (2–5 uppercase letters)
TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
# Finance keywords
FIN_KEYS = [
    "earnings","guidance","revenue","profit","loss","merger","acquisition","AI","chip",
    "semiconductor","inflation","recession","dividend","buyback","FED","rates","hike",
    "cut","CPI","PPI","jobs","NFP","GDP","options","calls","puts","breakout","support",
    "resistance","target","upgrade","downgrade","PT","valuation","P/E","multiple"
]

def extract_tickers(title: str, transcript: str) -> List[str]:
    text = f"{title} {transcript}"
    found = set()
    for m in TICKER_RE.finditer(text):
        t = m.group(0)
        # filter obvious non-tickers
        if t in {"USD","EPS","PPE","CEO","CFO","ETF","IPO","AI","PE"}: 
            continue
        found.add(t)
    return sorted(found)

def extract_keywords(text: str) -> List[str]:
    found = []
    up = text.lower()
    for k in FIN_KEYS:
        if k.lower() in up:
            found.append(k)
    return sorted(set(found))

def sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    try:
        if _SENT_BACKEND == "vader":
            return round(_vader.polarity_scores(text)["compound"], 4)
        elif _SENT_BACKEND == "textblob":
            from textblob import TextBlob
            return round(TextBlob(text).sentiment.polarity, 4)
        else:
            return 0.0
    except Exception:
        return 0.0

def credibility_for(channel: str) -> float:
    # 0.0–1.0 cred factor; whitelist gets a boost
    base = 0.6
    if channel in CHANNEL_WHITELIST:
        return 0.95
    # light heuristic for “news/finance” sounding channels
    if re.search(r"(bloomberg|cnbc|wsj|financial|economics|market|invest)", channel, re.I):
        return 0.8
    return base

def final_score(sent: float, kw: List[str], tickers: List[str], cred: float) -> float:
    # weight sentiment + relevancy + credibility
    rel = min(1.0, (len(kw)/8.0 + len(tickers)/5.0) / 2.0)
    raw = 0.55 * ((sent + 1) / 2) + 0.30 * rel + 0.15 * cred
    return round(min(1.0, max(0.0, raw)), 4)

# ---------- YouTube API wrappers ----------
class YTApi:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("YOUTUBE_API_KEY missing. Put it in .env or env.")
        if ytb_build is None:
            raise RuntimeError("google-api-python-client not installed.")
        self.client = ytb_build("youtube", "v3", developerKey=api_key)

    def search(self, query: str, max_results: int = 10, published_after: Optional[str]=None) -> List[Dict[str, Any]]:
        params = dict(
            q=query, part="snippet", type="video", order="date", maxResults=max_results
        )
        if published_after:
            params["publishedAfter"] = published_after
        for attempt in range(6):
            try:
                res = self.client.search().list(**params).execute()
                items = res.get("items", [])
                out = []
                for it in items:
                    sn = it["snippet"]
                    out.append({
                        "videoId": it["id"]["videoId"],
                        "title": sn["title"],
                        "channel": sn["channelTitle"],
                        "publishedAt": sn.get("publishedAt"),
                    })
                return out
            except Exception as e:
                log(f"YT search retry {attempt+1}/6: {e}", "WARN")
                backoff_sleep(attempt)
        return []

    def channel_recent(self, channel_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        # search by channel name then list recent
        q = f"{channel_name}"
        vids = self.search(q, max_results=max_results)
        # filter to channel (best effort)
        out = [v for v in vids if v["channel"].lower() == channel_name.lower()]
        return out[:max_results]

def fetch_transcript(video_id: str) -> Tuple[str, int]:
    """Returns (transcript_text, duration_s[best effort])"""
    if YouTubeTranscriptApi is None:
        return "", 0
    for attempt in range(6):
        try:
            segs = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join(s.get("text","") for s in segs)
            dur = int(sum(s.get("duration", 0) for s in segs))
            return text, dur
        except Exception as e:
            # Often transcript not available—just return empty after a couple tries
            if "Could not retrieve a transcript" in str(e) or "not available" in str(e).lower():
                return "", 0
            log(f"Transcript retry {attempt+1}/6 for {video_id}: {e}", "WARN")
            backoff_sleep(attempt)
    return "", 0

# ---------- Worker pipeline (multithread) ----------
class WorkerPool:
    def __init__(self, n: int = 6):
        self.n = max(1, n)
        self.q: "queue.Queue[Tuple[str,Dict[str,Any]]]" = queue.Queue()
        self.threads: List[threading.Thread] = []
        self.stop = threading.Event()
        ensure_sqlite()

    def submit(self, topic: str, item: Dict[str, Any]):
        self.q.put((topic, item))

    def _work(self):
        while not self.stop.is_set():
            try:
                topic, v = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(topic, v)
            except Exception as e:
                log(f"Worker error: {e}", "ERROR")
            finally:
                self.q.task_done()

    def _process(self, topic: str, v: Dict[str, Any]):
        vid = v["videoId"]
        title = v["title"]
        channel = v["channel"]
        published = v.get("publishedAt") or ""
        url = f"https://youtu.be/{vid}"

        # Transcript & analysis
        transcript, duration_s = fetch_transcript(vid)
        text_for_nlp = f"{title}. {transcript}"
        sent = sentiment_score(text_for_nlp)
        kws = extract_keywords(text_for_nlp)
        tix = extract_tickers(title, transcript)
        cred = credibility_for(channel)
        score = final_score(sent, kws, tix, cred)

        row = {
            "timestamp": utcnow_iso(),
            "videoId": vid,
            "published": published,
            "channel": channel,
            "title": title,
            "url": url,
            "sentiment": sent,
            "keywords": ",".join(kws),
            "tickers": ",".join(tix),
            "topic": topic,
            "score": score,
            "credibility": cred,
            "duration_s": duration_s
        }

        # Persist (dedup handled by SQLite PK)
        sqlite_upsert(row)
        csv_append(row)
        log(f"YT[{topic}] {channel}: {title[:80]!r} → score={score} sent={sent} tix={tix}")

    def start(self):
        for _ in range(self.n):
            t = threading.Thread(target=self._work, daemon=True)
            t.start()
            self.threads.append(t)

    def join(self):
        self.q.join()

    def shutdown(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=0.2)

# ---------- Orchestrator ----------
class YTBrain:
    def __init__(self, api_key: str, max_results: int = MAX_RESULTS_DEFAULT, days: int = 7, threads: int = 6):
        self.api = YTApi(api_key)
        self.max_results = max_results
        self.days = days
        self.pool = WorkerPool(n=threads)
        self.pool.start()

    def _published_after_iso(self) -> str:
        dt_from = datetime.now(timezone.utc) - timedelta(days=self.days)
        return dt_from.isoformat()

    def run_topics(self, topics: List[str]):
        since = self._published_after_iso()
        for topic in topics:
            items = self.api.search(topic, max_results=self.max_results, published_after=since)
            for it in items:
                self.pool.submit(topic, it)
        self.pool.join()

    def run_channels(self, channels: List[str]):
        for ch in channels:
            items = self.api.channel_recent(ch, max_results=self.max_results)
            for it in items:
                self.pool.submit(f"channel:{ch}", it)
        self.pool.join()

    def shutdown(self):
        self.pool.shutdown()

# ---------- CLI / Watch loop ----------
def parse_args():
    p = argparse.ArgumentParser(description="Lavish YT Brain Master (single file)")
    p.add_argument("--topics", type=str, default=",".join(DEFAULT_TOPICS),
                   help="Comma-sep topics/tickers to search")
    p.add_argument("--channels", type=str, default="",
                   help="Comma-sep channel names (exact) to crawl recent")
    p.add_argument("--max", type=int, default=MAX_RESULTS_DEFAULT, help="Max results per query")
    p.add_argument("--days", type=int, default=7, help="Lookback window (days)")
    p.add_argument("--threads", type=int, default=6, help="Worker threads")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--watch", action="store_true", help="Run forever (loop)")
    p.add_argument("--refresh-min", type=int, default=15, help="Watch loop refresh minutes")
    return p.parse_args()

def main():
    args = parse_args()

    if not YOUTUBE_API_KEY:
        log("YOUTUBE_API_KEY missing — set it in .env", "ERROR")
        sys.exit(2)

    topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()]
    channels = [c.strip() for c in (args.channels or "").split(",") if c.strip()]

    def _cycle():
        brain = YTBrain(YOUTUBE_API_KEY, max_results=args.max, days=args.days, threads=args.threads)
        try:
            if topics:
                brain.run_topics(topics)
            if channels:
                brain.run_channels(channels)
        finally:
            brain.shutdown()

    if args.once or not args.watch:
        _cycle()
        log(f"Done. CSV={CSV_PATH} DB={DB_PATH}")
        return

    # watch mode
    stop = {"flag": False}
    def _sigint(_a,_b):
        stop["flag"] = True
        log("Stopping watch loop…")
    signal.signal(signal.SIGINT, _sigint)

    refresh = max(3, int(args.refresh_min))
    log(f"Watch loop started. Refresh={refresh} min — CSV={CSV_PATH} DB={DB_PATH}")
    while not stop["flag"]:
        try:
            _cycle()
        except Exception as e:
            log(f"Cycle error: {e}", "ERROR")
        for i in range(refresh * 60):
            if stop["flag"]: break
            time.sleep(1)
    log("Exiting cleanly.")

if __name__ == "__main__":
    main()