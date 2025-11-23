# lavish_core/super_pipeline_2.py
# Wall Street Edition — Multi-Agent Async News & Signal Engine
# ------------------------------------------------------------
# Highlights:
# • Async collectors with per-source rate-limiters and 429 circuit breaker rotation
# • Sources: Yahoo Finance RSS (free), Finnhub (if key), NewsAPI (if key), lightweight fallback HTML headline scrapers
# • Sentiment: VADER (if available) or finance-keyword fallback
# • Per-ticker aggregation with freshness weights, boost from headline count
# • Outputs (under lavish_core/state2):
#     - news_stream.jsonl         (everything we see, de-duplicated by URL/title hash)
#     - signals.json              (ticker → score, count, last_ts, sample titles)
#     - learn_queue.jsonl         (clean text payloads for brain_vector ingestion)
#     - dashboard.html            (quick human view)
# • Clean API for run.py: run(mode="dry-run"| "paper"| "live", symbols=[...])
# • CLI:  python -m lavish_core.super_pipeline_2 --since 24h --tickers AAPL,MSFT --max-per 120
#
# Safe dependencies only: stdlib + aiohttp (optional), requests, VADER (optional)
# Falls back gracefully if aiohttp/VADER are missing.

from __future__ import annotations
import os, re, json, math, time, html, hashlib, asyncio, logging, textwrap, random, pathlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone

# ---------- Optional fast loop ----------
try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

# ---------- Optional deps ----------
try:
    import aiohttp  # async HTTP
    _HAVE_AIO = True
except Exception:
    _HAVE_AIO = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAVE_VADER = True
except Exception:
    _HAVE_VADER = False

import requests  # kept for RSS/HTML fallback where needed

ROOT = pathlib.Path(__file__).resolve().parent
STATE = ROOT / "state2"
STATE.mkdir(parents=True, exist_ok=True)

# ---------- Logging ----------
log = logging.getLogger("lavish.super2")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ---------- Env ----------
def getenv(k: str, default: str = "") -> str:
    return (os.getenv(k) or default).strip()

FINNHUB_API_KEY   = getenv("FINNHUB_API_KEY") or getenv("FINNHUB_API_KEY_2")
NEWSAPI_KEY       = getenv("NEWSAPI_KEY")
USER_AGENT        = getenv("USER_AGENT", "LavishBot/2.0 (+https://example.invalid)")
HTTP_TIMEOUT      = float(getenv("HTTP_TIMEOUT", "12"))
MAX_PER_SOURCE    = int(getenv("SP2_MAX_PER_SOURCE", "120"))

# ---------- Helpers ----------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def parse_http_date(s: str) -> Optional[datetime]:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def canon(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
JUNK = {"THE","AND","FOR","WITH","FROM","ABOUT","HTTPS","HTTP","WWW","NEWS","MARKET"}

def infer_tickers(text: str) -> List[str]:
    found = {t for t in TICKER_RE.findall(text.upper()) if t not in JUNK}
    return sorted(list(found))[:12]

def write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def append_jsonl(path: pathlib.Path, obj) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- Data Models ----------
@dataclass
class NewsItem:
    id: str
    source: str
    url: str
    title: str
    summary: str
    tickers: List[str]
    published: str
    score: float = 0.0
    raw: dict = field(default_factory=dict)

@dataclass
class Signal:
    symbol: str
    score: float
    n_headlines: int
    last_ts: str
    top_titles: List[str]

# ---------- Sentiment ----------
class Sentim:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if _HAVE_VADER else None
        # finance-keyword fallback
        self.pos = {"beats","beat","surge","soar","record","upgrade","outperform",
                    "raise","raised","profit","growth","acceleration","bullish","guidance raised"}
        self.neg = {"miss","misses","plunge","fall","drop","cut","cuts","downgrade","underperform",
                    "loss","decline","slowdown","bearish","lawsuit","probe","investigation","guidance cut"}

    def score(self, text: str) -> float:
        t = (text or "").strip()
        if not t: return 0.0
        if self.vader:
            try:
                return float(self.vader.polarity_scores(t)["compound"])
            except Exception:
                pass
        lo = t.lower()
        p = sum(1 for w in self.pos if w in lo)
        n = sum(1 for w in self.neg if w in lo)
        return float(max(-1.0, min(1.0, (p - n) / max(1, p + n))))

# ---------- Circuit Breaker Ring ----------
class SourceRing:
    """
    Maintains a rotating list of collectors; if one throws or returns 429 repeatedly,
    we rotate to the next. Each collector exposes: name, collect_async(...)
    """
    def __init__(self, collectors: List["BaseCollector"]):
        self.cols = collectors
        self.bad_until: Dict[str, float] = {}  # name -> epoch-until
        self.cooldown = 30.0  # seconds on 429
        self.fail_cd = 10.0   # seconds on errors

    def _ok(self, name: str) -> bool:
        return time.time() >= self.bad_until.get(name, 0.0)

    def mark_429(self, name: str):
        self.bad_until[name] = time.time() + self.cooldown

    def mark_fail(self, name: str):
        self.bad_until[name] = time.time() + self.fail_cd

    async def run_all(self, since: datetime, tickers: List[str], max_per: int) -> List[NewsItem]:
        # Try each collector that isn't cooling down
        results: List[NewsItem] = []
        for c in self.cols:
            if not self._ok(c.name):
                log.debug("skip collector %-10s (cooling down)", c.name); continue
            try:
                items, info = await c.collect_async(since, tickers, max_per)
                results.extend(items)
                if info.get("429"):
                    self.mark_429(c.name)
                    log.warning("collector %-10s hit 429 (cooldown %.0fs)", c.name, self.cooldown)
                # short stagger between sources
                await asyncio.sleep(0.2 + random.random()*0.3)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.mark_fail(c.name)
                log.warning("collector %-10s failed: %s", c.name, e)
        return results

# ---------- Collectors ----------
class BaseCollector:
    name = "base"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/*, */*"}
    async def collect_async(self, since: datetime, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict]:
        raise NotImplementedError

class YahooRSS(BaseCollector):
    name = "yahoo_rss"
    FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={tickers}&region=US&lang=en-US"

    async def collect_async(self, since: datetime, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict]:
        # requests is fine here; RSS returns quick & consistent
        syms = ",".join(tickers[:24]) if tickers else "AAPL,MSFT,NVDA,TSLA,SPY,AMZN,META,GOOGL"
        url = self.FEED.format(tickers=syms)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept":"application/rss+xml"}, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                return ([], {"429": True})
            if r.status_code != 200:
                return ([], {"status": r.status_code})
            xml = r.text
            entries = re.split(r"<item>", xml)[1:]
            out: List[NewsItem] = []
            for raw in entries:
                try:
                    title = canon(re.search(r"<title>(.*?)</title>", raw, re.S).group(1))
                    link  = canon(re.search(r"<link>(.*?)</link>", raw, re.S).group(1))
                    pub   = canon(re.search(r"<pubDate>(.*?)</pubDate>", raw, re.S).group(1))
                    dt    = parse_http_date(pub) or now_utc()
                    if dt < since: 
                        continue
                    tags = list({*(tickers or []), *infer_tickers(title)})[:12]
                    out.append(NewsItem(
                        id=sha1(f"{title}|{link}"),
                        source=self.name,
                        url=link,
                        title=title,
                        summary="",
                        tickers=tags,
                        published=iso(dt),
                        raw={"rss": True}
                    ))
                    if len(out) >= max_per: break
                except Exception:
                    continue
            return (out, {})
        except requests.RequestException as e:
            return ([], {"error": str(e)})

class FinnhubNews(BaseCollector):
    name = "finnhub"
    def __init__(self, key: str): self.key = key

    async def collect_async(self, since: datetime, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict]:
        if not self.key:
            return ([], {"skip": "no_key"})
        end = now_utc().date()
        start = since.date()
        out: List[NewsItem] = []
        # small concurrency with aiohttp if available
        symbols = (tickers or ["AAPL","MSFT","NVDA","TSLA","SPY"])[:16]
        async def one(session, sym):
            url = "https://finnhub.io/api/v1/company-news"
            params = {"symbol": sym, "from": start.isoformat(), "to": end.isoformat(), "token": self.key}
            try:
                if _HAVE_AIO:
                    async with session.get(url, params=params, timeout=HTTP_TIMEOUT) as resp:
                        if resp.status == 429: return "429"
                        if resp.status != 200: return "bad"
                        data = await resp.json()
                else:
                    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
                    if r.status_code == 429: return "429"
                    if r.status_code != 200: return "bad"
                    data = r.json()
            except Exception:
                return "err"
            cnt = 0
            for it in (data or []):
                dt = datetime.fromtimestamp(it.get("datetime", 0) or 0, tz=timezone.utc)
                if dt < since: 
                    continue
                title = canon(it.get("headline",""))
                link  = it.get("url") or ""
                out.append(NewsItem(
                    id=sha1(f"{title}|{link}"),
                    source=self.name,
                    url=link,
                    title=title,
                    summary=canon(it.get("summary","")),
                    tickers=list({sym, *infer_tickers(title)})[:12],
                    published=iso(dt),
                    raw=it
                ))
                cnt += 1
                if len(out) >= max_per: break
            return "ok" if cnt else "none"

        info = {}
        if _HAVE_AIO:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT+6)
            async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
                rets = await asyncio.gather(*(one(session, s) for s in symbols), return_exceptions=True)
            if any(r == "429" for r in rets): info["429"] = True
        else:
            rets = [await one(None, s) for s in symbols]  # type: ignore
            if any(r == "429" for r in rets): info["429"] = True
        return (out[:max_per], info)

class NewsAPIEverything(BaseCollector):
    name = "newsapi"
    def __init__(self, key: str): self.key = key

    async def collect_async(self, since: datetime, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict]:
        if not self.key:
            return ([], {"skip": "no_key"})
        q = " OR ".join((tickers or ["stocks","market","earnings","fed","rate"]))[:512]
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q, "language":"en", "sortBy":"publishedAt",
            "pageSize": min(100, max_per), "from": (since - timedelta(minutes=5)).isoformat(timespec="seconds")
        }
        headers = dict(self.headers); headers["X-Api-Key"] = self.key
        try:
            if _HAVE_AIO:
                timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT+4)
                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429: return ([], {"429": True})
                        if resp.status != 200: return ([], {"status": resp.status})
                        data = await resp.json()
            else:
                r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
                if r.status_code == 429: return ([], {"429": True})
                if r.status_code != 200: return ([], {"status": r.status_code})
                data = r.json()
        except Exception as e:
            return ([], {"error": str(e)})

        out: List[NewsItem] = []
        for a in (data or {}).get("articles", []):
            dt = parse_http_date(a.get("publishedAt") or "") or now_utc()
            if dt < since: 
                continue
            title = canon(a.get("title",""))
            link  = a.get("url") or ""
            out.append(NewsItem(
                id=sha1(f"{title}|{link}"),
                source=self.name,
                url=link,
                title=title,
                summary=canon(a.get("description","")),
                tickers=infer_tickers(title)[:12],
                published=iso(dt),
                raw=a
            ))
            if len(out) >= max_per: break
        return (out, {})

class FallbackScraper(BaseCollector):
    """
    Extremely light HTML headline grabber for resilience.
    Targets a couple of public pages (structure-agnostic via <a> and <h*> pick).
    """
    name = "fallback_html"
    PAGES = [
        "https://www.marketwatch.com/",  # headlines
        "https://www.investopedia.com/markets-news-4427704",  # markets news hub
    ]

    async def collect_async(self, since: datetime, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict]:
        out: List[NewsItem] = []
        for url in self.PAGES:
            try:
                r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept":"text/html"}, timeout=HTTP_TIMEOUT)
                if r.status_code == 429:
                    return (out, {"429": True})
                if r.status_code != 200:
                    continue
                html_text = r.text
                # naive headline anchors
                cand = set()
                for m in re.finditer(r"<a[^>]+>(.*?)</a>", html_text, re.I|re.S):
                    t = canon(m.group(1))
                    if 30 <= len(t) <= 150 and "cookie" not in t.lower() and "privacy" not in t.lower():
                        cand.add(t)
                # Also h1/h2
                for m in re.finditer(r"<h[12][^>]*>(.*?)</h[12]>", html_text, re.I|re.S):
                    cand.add(canon(m.group(1)))
                for title in list(cand)[:max_per]:
                    tags = list({*(tickers or []), *infer_tickers(title)})[:12]
                    if not tags: 
                        continue
                    out.append(NewsItem(
                        id=sha1(f"{title}|{url}"),
                        source=self.name,
                        url=url,
                        title=title,
                        summary="",
                        tickers=tags,
                        published=iso(now_utc()),
                        raw={"page": url}
                    ))
                    if len(out) >= max_per: break
            except Exception:
                continue
        return (out, {})

# ---------- Dedup ----------
def dedup(items: List[NewsItem]) -> List[NewsItem]:
    seen: set[str] = set()
    out: List[NewsItem] = []
    for it in sorted(items, key=lambda x: x.published, reverse=True):
        if it.id in seen: 
            continue
        seen.add(it.id)
        out.append(it)
    return out

# ---------- Signals ----------
def build_signals(items: List[NewsItem], focus: List[str]) -> Dict[str, Signal]:
    buckets: Dict[str, List[NewsItem]] = {}
    for it in items:
        targets = (focus or it.tickers or [])
        for sym in targets:
            if sym.isalpha() and 2 <= len(sym) <= 6:
                buckets.setdefault(sym, []).append(it)

    sigs: Dict[str, Signal] = {}
    for sym, arr in buckets.items():
        if not arr: continue
        s = 0.0; W = 0.0
        for it in arr:
            age_min = max(3.0, (now_utc() - datetime.fromisoformat(it.published)).total_seconds()/60.0)
            w = 1.0 / math.log1p(age_min)  # recency weighting
            s += it.score * w; W += w
        base = s / max(1e-6, W)
        count_boost = min(0.30, 0.02 * len(arr))
        final = max(-1.0, min(1.0, base + math.copysign(count_boost, base)))
        titles = [it.title for it in sorted(arr, key=lambda x: x.score, reverse=True)[:6]]
        last_ts = max(it.published for it in arr)
        sigs[sym] = Signal(symbol=sym, score=float(final), n_headlines=len(arr), last_ts=last_ts, top_titles=titles)
    return sigs

# ---------- Dashboard ----------
def render_dashboard(items: List[NewsItem], signals: Dict[str, Signal]) -> str:
    rows = []
    for sym, sig in sorted(signals.items(), key=lambda kv: -kv[1].score):
        rows.append(f"<tr><td>{sym}</td><td>{sig.score:+.2f}</td><td>{sig.n_headlines}</td>"
                    f"<td>{html.escape('; '.join(sig.top_titles))}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Lavish — Wall Street Signals</title>
    <style>
    body{{font-family:-apple-system,Segoe UI,Roboto,Arial;margin:22px}}
    table{{border-collapse:collapse;width:100%}}
    th,td{{border:1px solid #ddd;padding:8px;font-size:14px}}th{{background:#111;color:#fff}}
    .muted{{color:#666;font-size:12px}}
    </style></head><body>
    <h2>LavishBot — Super Pipeline 2</h2>
    <div class="muted">Updated {iso(now_utc())} • Unique items: {len(items)}</div>
    <h3>Signals</h3>
    <table><tr><th>Symbol</th><th>Score</th><th>Count</th><th>Top headlines</th></tr>
    {''.join(rows)}
    </table>
    </body></html>"""

# ---------- Main Orchestration ----------
async def collect_all_async(since: timedelta, tickers: List[str], max_per: int) -> Tuple[List[NewsItem], Dict[str,int]]:
    since_dt = now_utc() - since
    collectors: List[BaseCollector] = [YahooRSS()]
    if FINNHUB_API_KEY:
        collectors.append(FinnhubNews(FINNHUB_API_KEY))
    if NEWSAPI_KEY:
        collectors.append(NewsAPIEverything(NEWSAPI_KEY))
    collectors.append(FallbackScraper())  # always keep a last-resort

    ring = SourceRing(collectors)
    items: List[NewsItem] = []
    # one pass through the ring is usually enough; repeat once to bypass temporary 429s
    for _ in range(2):
        batch = await ring.run_all(since_dt, tickers, max_per=max_per)
        items.extend(batch)
        # small pause between passes
        await asyncio.sleep(0.4 + random.random()*0.4)

    # Dedup & sentiment
    uniq = dedup(items)
    snt = Sentim()
    for it in uniq:
        it.score = snt.score(f"{it.title}. {it.summary}")

    # Persist stream + learning queue
    news_stream = STATE / "news_stream.jsonl"
    learn_queue = STATE / "learn_queue.jsonl"
    for it in uniq:
        append_jsonl(news_stream, asdict(it))
        append_jsonl(learn_queue, {
            "ts": iso(now_utc()),
            "symbol_hints": it.tickers,
            "title": it.title,
            "summary": it.summary,
            "source": it.source,
            "url": it.url
        })

    meta = {
        "generated": iso(now_utc()),
        "sources": sorted({it.source for it in uniq}),
        "n_raw": len(items),
        "n_unique": len(uniq)
    }
    write_json(STATE / "summary.json", meta)
    return uniq, meta

def _parse_since(spec: str) -> timedelta:
    m = re.fullmatch(r"(\d+)([mhd])", spec.strip().lower())
    if not m: return timedelta(hours=24)
    n = int(m.group(1)); u = m.group(2)
    return timedelta(minutes=n) if u=="m" else timedelta(hours=n) if u=="h" else timedelta(days=n)

async def run_async(mode: str, symbols: List[str], since_spec: str = "24h", max_per: int = MAX_PER_SOURCE, quiet: bool=False) -> Dict[str, Dict]:
    since = _parse_since(since_spec)
    if not quiet:
        log.info("SuperPipeline2 start | mode=%s | since=%s | tickers=%s | max_per=%d",
                 mode, since_spec, ",".join(symbols) if symbols else "(auto)", max_per)
    uniq, meta = await collect_all_async(since, symbols, max_per=max_per)
    signals = build_signals(uniq, symbols)
    write_json(STATE / "signals.json", {k: asdict(v) for k,v in signals.items()})
    (STATE / "dashboard.html").write_text(render_dashboard(uniq, signals), encoding="utf-8")
    if not quiet:
        log.info("SuperPipeline2 done | %d unique | %d tickers with signals", len(uniq), len(signals))
    return {"meta": meta, "signals": {k: asdict(v) for k,v in signals.items()}}

# Public entry for run.py autoloader
def run(mode: str = "dry-run", symbols: Optional[List[str]] = None, since_spec: str = "24h", max_per: int = MAX_PER_SOURCE, quiet: bool=False):
    symbols = [s.strip().upper() for s in (symbols or []) if s.strip()]
    if _HAVE_AIO:
        return asyncio.run(run_async(mode, symbols, since_spec=since_spec, max_per=max_per, quiet=quiet))
    # fallback without aiohttp: still works (collectors use requests), just sync-ify
    return asyncio.run(run_async(mode, symbols, since_spec=since_spec, max_per=max_per, quiet=quiet))

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Lavish Super Pipeline 2 — Wall Street Edition")
    ap.add_argument("--since", type=str, default="24h", help="Lookback (e.g., 90m, 6h, 2d)")
    ap.add_argument("--tickers", type=str, default="", help="Comma list (AAPL,MSFT,NVDA,SPY)")
    ap.add_argument("--max-per", type=int, default=MAX_PER_SOURCE, help="Max items per collector")
    ap.add_argument("--mode", type=str, default=os.getenv("MODE","dry-run"), help="dry-run | paper | live")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.tickers.split(",") if s.strip()] if args.tickers else []
    if _HAVE_AIO:
        asyncio.run(run_async(args.mode, syms, since_spec=args.since, max_per=args.max_per, quiet=args.quiet))
    else:
        # Still executes collectors via requests; we keep same API for simplicity.
        asyncio.run(run_async(args.mode, syms, since_spec=args.since, max_per=args.max_per, quiet=args.quiet))