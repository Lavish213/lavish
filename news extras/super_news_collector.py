# lavish_core/news/super_news_collector.py
# 🔥 One-file, production-ready news collector (free-first, API-aware, rate-limit safe)
# - Async (aiohttp) with keyed backoff + jitter
# - Pluggable collectors: Yahoo Finance RSS (free), CryptoPanic (token), Finnhub, AlphaVantage, NewsAPI
# - Optional key rotation (e.g., FINNHUB_API_KEY, FINNHUB_API_KEY_2, FINNHUB_API_KEYS="key1,key2")
# - Dedup + merge + ticker filtering
# - Writes snapshots to state/ and a rolling cache file
# - CLI:
#   python -m lavish_core.news.super_news_collector --since 24h --tickers AAPL,MSFT,NVDA --max-per 40 --quiet
#
# ENV it will read (use .env or exported):
#  FINNHUB_API_KEY, FINNHUB_API_KEY_2, FINNHUB_API_KEYS
#  ALPHA_VANTAGE_API_KEY
#  NEWSAPI_KEY
#  CRYPTOPANIC_TOKEN   (a.k.a. developer auth_token)
#  TIMEZONE (fallback: America/Los_Angeles)
#  LOG_LEVEL (fallback: INFO)

from __future__ import annotations

import asyncio
import aiohttp
import argparse
import os
import re
import json
import random
import logging
import pathlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Iterable, Tuple
from contextlib import asynccontextmanager

try:
    # optional but nice to have
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --------------------------- Logging ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("super_news")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

# --------------------------- Paths ---------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if "__file__" in globals() else pathlib.Path(".").resolve()
STATE_DIR = ROOT / "lavish_core" / "news" / "state"
CACHE_DIR = ROOT / "data" / "cache"
STATE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_CACHE = STATE_DIR / "news_master.json"

# --------------------------- Time helpers ---------------------------
TZ_NAME = os.getenv("TIMEZONE", "America/Los_Angeles")

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def parse_since_arg(since: str) -> datetime:
    # "24h", "3d", "7d", "1h"
    m = re.fullmatch(r"(\d+)\s*(h|d)", since.strip().lower())
    if not m:
        raise ValueError("--since expects like '24h' or '7d'")
    n, unit = int(m.group(1)), m.group(2)
    delta = timedelta(hours=n) if unit == "h" else timedelta(days=n)
    return utcnow() - delta

def to_iso(dt: datetime) -> str:
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

# --------------------------- Key utils ---------------------------
def split_keys(val: Optional[str]) -> List[str]:
    if not val:
        return []
    # Accept comma-separated or separate envs (e.g. FINNHUB_API_KEY_2)
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [p for p in parts if p and p.lower() != "none"]

def gather_keys(prefix: str) -> List[str]:
    # Collect PREFIX, PREFIX_2, PREFIX_3, or PREFIXS (plural) list
    keys: List[str] = []
    direct = os.getenv(prefix)
    if direct:
        keys.append(direct.strip())
    for i in range(2, 10):
        alt = os.getenv(f"{prefix}_{i}")
        if alt:
            keys.append(alt.strip())
    plural = os.getenv(f"{prefix}S")
    keys.extend(split_keys(plural))
    # dedupe preserve order
    seen = set()
    uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq

# --------------------------- HTTP helpers ---------------------------
UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=15, connect=10)
CONCURRENCY = 6  # global limit
RATE_429_SLEEP = (7, 18)  # jitter window on 429

@asynccontextmanager
async def http_session():
    conn = aiohttp.TCPConnector(limit_per_host=4, ssl=False)
    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, connector=conn) as s:
        yield s

async def _sleep_jitter(base: float, spread: float = 0.4):
    # Sleep ~ base +/- spread%
    jitter = base * random.uniform(1 - spread, 1 + spread)
    await asyncio.sleep(jitter)

async def fetch_json(session: aiohttp.ClientSession, url: str, headers: Optional[Dict[str, str]] = None,
                     max_retries: int = 4, backoff: float = 1.8) -> Optional[Dict[str, Any]]:
    tries = 0
    hdrs = {"User-Agent": random.choice(UA_POOL)}
    if headers:
        hdrs.update(headers)

    while True:
        tries += 1
        try:
            async with session.get(url, headers=hdrs) as resp:
                if resp.status == 429:
                    wait = random.uniform(*RATE_429_SLEEP)
                    logger.warning(f"[429] rate-limited for {url} -> sleeping {wait:.1f}s")
                    await asyncio.sleep(wait)
                    if tries <= max_retries:
                        continue
                    return None
                if resp.status >= 500:
                    if tries <= max_retries:
                        await _sleep_jitter(backoff)
                        backoff *= 1.8
                        continue
                    logger.error(f"[{resp.status}] server error for {url}")
                    return None
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning(f"[{resp.status}] {url} -> {txt[:140]}")
                    return None
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype or "json" in ctype:
                    return await resp.json()
                # fallback: try parse JSON if text is JSON-ish
                txt = await resp.text()
                try:
                    return json.loads(txt)
                except Exception:
                    logger.debug(f"Non-JSON response for {url[:80]}")
                    return None
        except asyncio.TimeoutError:
            if tries <= max_retries:
                await _sleep_jitter(backoff)
                backoff *= 1.8
                continue
            logger.error(f"Timeout for {url}")
            return None
        except Exception as e:
            if tries <= max_retries:
                await _sleep_jitter(backoff)
                backoff *= 1.8
                continue
            logger.error(f"HTTP error for {url}: {e}")
            return None

async def fetch_text(session: aiohttp.ClientSession, url: str, headers: Optional[Dict[str, str]] = None,
                     max_retries: int = 4, backoff: float = 1.8) -> Optional[str]:
    tries = 0
    hdrs = {"User-Agent": random.choice(UA_POOL)}
    if headers:
        hdrs.update(headers)

    while True:
        tries += 1
        try:
            async with session.get(url, headers=hdrs) as resp:
                if resp.status == 429:
                    wait = random.uniform(*RATE_429_SLEEP)
                    logger.warning(f"[429] rate-limited for {url} -> sleeping {wait:.1f}s")
                    await asyncio.sleep(wait)
                    if tries <= max_retries:
                        continue
                    return None
                if resp.status >= 500:
                    if tries <= max_retries:
                        await _sleep_jitter(backoff)
                        backoff *= 1.8
                        continue
                    logger.error(f"[{resp.status}] server error for {url}")
                    return None
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning(f"[{resp.status}] {url} -> {txt[:140]}")
                    return None
                return await resp.text()
        except asyncio.TimeoutError:
            if tries <= max_retries:
                await _sleep_jitter(backoff)
                backoff *= 1.8
                continue
            logger.error(f"Timeout for {url}")
            return None
        except Exception as e:
            if tries <= max_retries:
                await _sleep_jitter(backoff)
                backoff *= 1.8
                continue
            logger.error(f"HTTP error for {url}: {e}")
            return None

# --------------------------- Normalization ---------------------------
def norm_article(
    *,
    source: str,
    title: str,
    url: str,
    published: Optional[str],
    tickers: Optional[List[str]] = None,
    summary: Optional[str] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "published": published,
        "tickers": [t.upper() for t in (tickers or []) if t],
        "summary": summary or "",
        "raw": raw or {},
    }

def dedupe(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for a in articles:
        key = (a["url"].lower(), a["title"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

def filter_since(articles: List[Dict[str, Any]], since_dt: datetime) -> List[Dict[str, Any]]:
    out = []
    for a in articles:
        ts = a.get("published")
        if not ts:
            out.append(a)  # keep unknowns
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt >= since_dt:
                out.append(a)
        except Exception:
            out.append(a)
    return out

def filter_focus(articles: List[Dict[str, Any]], tickers: List[str]) -> List[Dict[str, Any]]:
    if not tickers:
        return articles
    fset = set([t.upper() for t in tickers])
    out = []
    for a in articles:
        at = set(a.get("tickers") or [])
        if at & fset:
            out.append(a)
        else:
            # simple heuristic: TICKER in title
            ttl = (a.get("title") or "").upper()
            if any(t in ttl for t in fset):
                out.append(a)
    return out

# --------------------------- Collectors ---------------------------

class BaseCollector:
    name = "Base"
    semaphore = asyncio.Semaphore(CONCURRENCY)

    def __init__(self, since: datetime, focus: List[str], max_per: int = 40):
        self.since = since
        self.focus = [t.upper() for t in focus]
        self.max_per = max_per

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        raise NotImplementedError

# 1) Yahoo Finance RSS (free; per-ticker feed)
class YahooRSSCollector(BaseCollector):
    name = "YahooRSS"

    def _feed_urls(self) -> List[str]:
        # Yahoo RSS variants; try a couple variants that commonly work
        urls = []
        for t in self.focus:
            urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US")
            urls.append(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}")
        return urls

    @staticmethod
    def _parse_rss(xml_text: str, ticker: str) -> List[Dict[str, Any]]:
        # Extremely light RSS parsing (no external deps)
        # We’ll parse item<title>, <link>, <pubDate>
        items = []
        for m in re.finditer(r"<item>(.*?)</item>", xml_text, flags=re.S | re.I):
            block = m.group(1)
            def _tag(tag):
                mt = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.S | re.I)
                return mt.group(1).strip() if mt else ""
            title = re.sub("<.*?>", "", _tag("title"))
            link = re.sub("<.*?>", "", _tag("link"))
            pub = _tag("pubDate")
            # convert to ISO if possible
            iso = None
            try:
                # e.g. Tue, 02 Jul 2024 10:00:00 +0000
                from email.utils import parsedate_to_datetime
                iso = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
            except Exception:
                iso = None
            items.append(norm_article(
                source="yahoo_rss",
                title=title,
                url=link,
                published=iso,
                tickers=[ticker],
                raw={"rss": block[:1000]}
            ))
        return items

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        for url in self._feed_urls():
            ticker = re.search(r"s=([A-Za-z.\-]+)", url)
            tk = (ticker.group(1).upper() if ticker else None) or "UNK"
            txt = await fetch_text(session, url)
            await asyncio.sleep(random.uniform(0.3, 0.9))
            if not txt:
                continue
            items = self._parse_rss(txt, tk)
            articles.extend(items[: self.max_per])
        return articles

# 2) CryptoPanic (free dev token; public=true)
class CryptoPanicCollector(BaseCollector):
    name = "CryptoPanic"

    def __init__(self, since: datetime, focus: List[str], max_per: int = 40):
        super().__init__(since, focus, max_per)
        self.token = os.getenv("CRYPTOPANIC_TOKEN")

    def enabled(self) -> bool:
        return bool(self.token)

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not self.enabled():
            return []
        base = "https://cryptopanic.com/api/v1/posts/"
        url = f"{base}?auth_token={self.token}&public=true&filter=hot"
        data = await fetch_json(session, url)
        out: List[Dict[str, Any]] = []
        for it in (data or {}).get("results", [])[: self.max_per]:
            title = (it.get("title") or "").strip()
            link = ((it.get("source") or {}).get("url") or "").strip() or (it.get("url") or "")
            published = it.get("published_at")
            out.append(norm_article(
                source="cryptopanic",
                title=title,
                url=link,
                published=published,
                tickers=[],  # crypto; not mapping to equities here
                raw=it,
            ))
        return out

# 3) Finnhub company-news
class FinnhubCollector(BaseCollector):
    name = "Finnhub"

    def __init__(self, since: datetime, focus: List[str], max_per: int = 40):
        super().__init__(since, focus, max_per)
        self.keys = gather_keys("FINNHUB_API_KEY")

    def enabled(self) -> bool:
        return bool(self.keys)

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not self.enabled() or not self.focus:
            return []
        out: List[Dict[str, Any]] = []
        start = (self.since - timedelta(days=1)).date().isoformat()
        end = utcnow().date().isoformat()
        ks = list(self.keys)

        async def one(tk: str) -> List[Dict[str, Any]]:
            nonlocal ks
            k = ks[0]
            ks = ks[1:] + ks[:1]  # rotate
            url = f"https://finnhub.io/api/v1/company-news?symbol={tk}&from={start}&to={end}&token={k}"
            data = await fetch_json(session, url)
            res: List[Dict[str, Any]] = []
            for it in (data or [])[: self.max_per]:
                dt = it.get("datetime")
                pub = None
                if isinstance(dt, (int, float)):
                    pub = datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
                res.append(norm_article(
                    source="finnhub",
                    title=(it.get("headline") or ""),
                    url=(it.get("url") or ""),
                    published=pub,
                    tickers=[tk],
                    summary=(it.get("summary") or ""),
                    raw=it,
                ))
            return res

        sem = asyncio.Semaphore(4)
        async def guarded(tk: str):
            async with sem:
                return await one(tk)

        gathered = await asyncio.gather(*(guarded(t) for t in self.focus))
        for arr in gathered:
            out.extend(arr)
        return out

# 4) Alpha Vantage NEWS_SENTIMENT (free/limited)
class AlphaVantageCollector(BaseCollector):
    name = "AlphaVantage"

    def __init__(self, since: datetime, focus: List[str], max_per: int = 40):
        super().__init__(since, focus, max_per)
        self.key = os.getenv("ALPHA_VANTAGE_API_KEY")

    def enabled(self) -> bool:
        return bool(self.key)

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not self.enabled() or not self.focus:
            return []
        tickers = ",".join(self.focus[:10])
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={self.key}"
        data = await fetch_json(session, url)
        out: List[Dict[str, Any]] = []
        for it in (data or {}).get("feed", [])[: self.max_per]:
            ts = it.get("time_published")
            pub = None
            if ts:
                # format: "20240618T132800"
                try:
                    dt = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                    pub = dt.isoformat()
                except Exception:
                    pub = None
            out.append(norm_article(
                source="alphavantage",
                title=(it.get("title") or ""),
                url=(it.get("url") or ""),
                published=pub,
                tickers=[t.get("ticker", "").upper() for t in (it.get("ticker_sentiment") or []) if t.get("ticker")],
                summary=(it.get("summary") or ""),
                raw=it,
            ))
        return out

# 5) NewsAPI.org (paid/free limited) — if key present; query per ticker
class NewsAPICollector(BaseCollector):
    name = "NewsAPI"

    def __init__(self, since: datetime, focus: List[str], max_per: int = 40):
        super().__init__(since, focus, max_per)
        self.key = os.getenv("NEWSAPI_KEY")

    def enabled(self) -> bool:
        return bool(self.key)

    async def collect(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        if not self.enabled() or not self.focus:
            return []
        frm = self.since.date().isoformat()
        url_tpl = "https://newsapi.org/v2/everything?q={q}&from={frm}&sortBy=publishedAt&language=en&pageSize={ps}&apiKey={k}"
        out: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(4)

        async def one(tk: str) -> List[Dict[str, Any]]:
            u = url_tpl.format(q=tk, frm=frm, ps=min(self.max_per, 50), k=self.key)
            data = await fetch_json(session, u, headers={"Accept": "application/json"})
            res: List[Dict[str, Any]] = []
            for a in (data or {}).get("articles", []):
                pub = a.get("publishedAt")
                res.append(norm_article(
                    source="newsapi",
                    title=(a.get("title") or ""),
                    url=(a.get("url") or ""),
                    published=pub,
                    tickers=[tk],
                    summary=(a.get("description") or ""),
                    raw=a,
                ))
            return res

        async def guarded(tk: str):
            async with sem:
                await asyncio.sleep(random.uniform(0.3, 1.0))
                return await one(tk)

        gathered = await asyncio.gather(*(guarded(t) for t in self.focus))
        for arr in gathered:
            out.extend(arr[: self.max_per])
        return out

# --------------------------- Master Runner ---------------------------

async def run_master(since: datetime, tickers: List[str], max_per: int, quiet: bool) -> Dict[str, Any]:
    collectors: List[BaseCollector] = [
        YahooRSSCollector(since, tickers, max_per),
        CryptoPanicCollector(since, tickers, max_per),
        FinnhubCollector(since, tickers, max_per),
        AlphaVantageCollector(since, tickers, max_per),
        NewsAPICollector(since, tickers, max_per),
    ]

    # Filter disabled collectors silently if key missing
    collectors = [c for c in collectors if getattr(c, "enabled", lambda: True)()]

    if not quiet:
        enabled_names = ", ".join(c.name for c in collectors)
        logger.info(f"Enabled collectors: {enabled_names or 'None'}")

    all_articles: List[Dict[str, Any]] = []
    started = time.monotonic()

    async with http_session() as session:
        results = await asyncio.gather(*(c.collect(session) for c in collectors), return_exceptions=True)

    for c, r in zip(collectors, results):
        if isinstance(r, Exception):
            logger.error(f"[{c.name}] crashed: {r}")
            continue
        n = len(r)
        if not quiet:
            logger.info(f"✅ {c.name}: {n} items")
        all_articles.extend(r)

    # Dedup & filter
    before = len(all_articles)
    all_articles = dedupe(all_articles)
    if not quiet:
        logger.info(f"Deduped {before} ➜ {len(all_articles)}")

    all_articles = filter_since(all_articles, since)
    all_articles = filter_focus(all_articles, tickers)

    # Trim per ticker if requested (soft cap by sorting recent first)
    def _key(a: Dict[str, Any]):
        p = a.get("published")
        try:
            return datetime.fromisoformat((p or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    all_articles.sort(key=_key, reverse=True)

    # Persist
    snapshot = {
        "meta": {
            "generated_at": to_iso(utcnow()),
            "count": len(all_articles),
            "focus": tickers,
            "since": to_iso(since),
            "duration_sec": round(time.monotonic() - started, 2),
        },
        "articles": all_articles,
    }

    ts_name = f"news_snapshot_{utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_ts = STATE_DIR / ts_name
    out_roll = ROLLING_CACHE

    with out_ts.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with out_roll.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    if not quiet:
        logger.info(f"Saved snapshot: {out_ts}")
        logger.info(f"Saved rolling:  {out_roll}")

    return snapshot

# --------------------------- CLI ---------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="super_news_collector",
        description="Lavish super news collector (async, free-first, rate-limit safe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--since", type=str, default="24h", help="Lookback window like '24h' or '7d'")
    p.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (e.g., AAPL,MSFT,NVDA,SPY)")
    p.add_argument("--max-per", type=int, default=40, help="Max items per collector/ticker")
    p.add_argument("--quiet", action="store_true", help="Less logging")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    since_dt = parse_since_arg(args.since)
    focus = [t.strip().upper() for t in (args.tickers or "").split(",") if t.strip()]
    if not focus:
        focus = ["AAPL", "MSFT", "NVDA", "SPY"]
    if not args.quiet:
        logger.info(f"Starting super news: since={args.since} (~{to_iso(since_dt)}), focus={','.join(focus)}, max_per={args.max_per}")
    asyncio.run(run_master(since_dt, focus, args.max_per, args.quiet))

if __name__ == "__main__":
    main()

"""
=========================== HOW TO RUN ===========================

# From repo root (Lavish_bot)
PYTHONPATH=. python3 -m lavish_core.news.super_news_collector \
  --since 24h \
  --tickers AAPL,MSFT,NVDA,SPY \
  --max-per 40

# Quiet mode (CI/cron friendly)
PYTHONPATH=. python3 -m lavish_core.news.super_news_collector --since 24h --tickers AAPL,MSFT --quiet

# Outputs:
#  - lavish_core/news/state/news_snapshot_YYYYMMDD_HHMMSS.json (timestamped)
#  - lavish_core/news/state/news_master.json                 (rolling)

# Env you can set (examples):
#  FINNHUB_API_KEY=XXXX
#  FINNHUB_API_KEY_2=YYYY
#  FINNHUB_API_KEYS="ZZZZ,WWWW"
#  ALPHA_VANTAGE_API_KEY=AAAA
#  NEWSAPI_KEY=BBBB
#  CRYPTOPANIC_TOKEN=cccc
#  TIMEZONE=America/Los_Angeles
#  LOG_LEVEL=INFO

# Notes:
# - YahooRSS is fully free; others activate only if the key is set.
# - Built-in jitter, backoff, and 429 sleeps minimize bans and keep runs stable.
# - Dedup is URL+title; you can extend to use content hash if needed.
# - This file is self-contained and safe to 'drop in' as-is.
==================================================================
"""