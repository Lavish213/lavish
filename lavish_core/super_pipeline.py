"""
Super Pipeline v1 — Multi-source News & Market Ingest (Wall Street Edition)

• Rotating sources with 429-aware circuit breaker
• Quorum goal: stop once we have enough unique items
• Dedup + light sentiment + per-ticker aggregation
• Saves: state/news_raw.jsonl, state/news_unique.jsonl, state/signals.json
"""

from __future__ import annotations
import os, sys, time, json, math, random, re, hashlib, html, argparse, pathlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

import requests

ROOT = pathlib.Path(__file__).resolve().parent
STATE = ROOT / "state"
STATE.mkdir(parents=True, exist_ok=True)

# ── Logging (simple) ──────────────────────────────────────────────────────────
def log(msg: str): print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")

UA = "LavishBot/WS-Edition"
TIMEOUT = 12

def http_get(url: str, params: Optional[dict]=None, headers: Optional[dict]=None) -> requests.Response:
    h = {"User-Agent": UA, "Accept": "application/json, text/*, */*"}
    if headers: h.update(headers)
    return requests.get(url, params=params, headers=h, timeout=TIMEOUT)

# ── Circuit breaker + rotation ────────────────────────────────────────────────
class Circuit:
    def __init__(self, name: str): self.name=name; self.fail=0; self.open_until=0.0
    def allow(self) -> bool: return time.time() >= self.open_until
    def record(self, ok: bool):
        if ok: self.fail = 0
        else:
            self.fail += 1
            self.open_until = time.time() + min(60, 2 ** min(self.fail, 5))

# ── Models ────────────────────────────────────────────────────────────────────
@dataclass
class NewsItem:
    id: str; src: str; url: str; title: str; summary: str; tickers: List[str]
    published: str; score: float = 0.0; raw: dict = field(default_factory=dict)

def sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()
def now() -> datetime: return datetime.now(timezone.utc)
def iso(dt: datetime) -> str: return dt.astimezone(timezone.utc).isoformat()

TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
STOP = {"THE","AND","FOR","WITH","FROM","ABOUT","HTTPS","HTTP","WWW","NEWS","MARKET"}

def infer_tickers(text: str) -> List[str]:
    return sorted(list({t for t in TICKER_RE.findall(text.upper()) if t not in STOP}))[:8]

# ── Sources ───────────────────────────────────────────────────────────────────
def pull_yahoo_rss(since: datetime, focus: List[str], limit: int) -> List[NewsItem]:
    syms = ",".join(focus[:20]) if focus else "AAPL,MSFT,NVDA,SPY,TSLA,AMZN"
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={syms}&region=US&lang=en-US"
    r = http_get(url, headers={"Accept":"application/rss+xml"})
    if r.status_code != 200: raise RuntimeError(f"yahoo rss {r.status_code}")
    items: List[NewsItem] = []
    parts = re.split(r"<item>", r.text)[1:]
    for raw in parts:
        try:
            title = re.search(r"<title>(.*?)</title>", raw, re.S).group(1)
            link  = re.search(r"<link>(.*?)</link>", raw, re.S).group(1)
            pub   = re.search(r"<pubDate>(.*?)</pubDate>", raw, re.S).group(1)
            dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").astimezone(timezone.utc)
            if dt < since: continue
            tickers = list({*(infer_tickers(title)), *(focus or [])})[:10]
            nid = sha1(f"{title}|{link}")
            items.append(NewsItem(id=nid, src="yahoo", url=link, title=html.unescape(title).strip(),
                                  summary="", tickers=tickers, published=iso(dt)))
            if len(items) >= limit: break
        except Exception:
            continue
    return items

def pull_finnhub_company(since: datetime, focus: List[str], limit: int) -> List[NewsItem]:
    key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_API")
    if not key: return []
    out: List[NewsItem] = []
    start, end = since.date().isoformat(), now().date().isoformat()
    for sym in (focus or ["AAPL","MSFT","NVDA","SPY"]):
        r = http_get("https://finnhub.io/api/v1/company-news",
                     params={"symbol": sym, "from": start, "to": end, "token": key})
        if r.status_code == 429: raise RuntimeError("finnhub 429")
        if r.status_code != 200: continue
        for it in r.json():
            dt = datetime.fromtimestamp(it.get("datetime",0), tz=timezone.utc)
            if dt < since: continue
            title = it.get("headline","")
            nid = sha1(f"{title}|{it.get('url','')}")
            out.append(NewsItem(id=nid, src="finnhub", url=it.get("url",""),
                                title=title, summary=it.get("summary",""),
                                tickers=list({sym, *infer_tickers(title)})[:10],
                                published=iso(dt), raw=it))
            if len(out) >= limit: break
    return out

def pull_newsapi(since: datetime, focus: List[str], limit: int) -> List[NewsItem]:
    key = os.getenv("NEWSAPI_KEY")
    if not key: return []
    q = " OR ".join((focus or ["stocks","market","earnings","fed"]))[:512]
    r = http_get("https://newsapi.org/v2/everything",
                 params={"q": q, "language":"en","from": since.isoformat(timespec="seconds"),
                         "sortBy":"publishedAt","pageSize": min(100,limit)},
                 headers={"X-Api-Key": key})
    if r.status_code == 429: raise RuntimeError("newsapi 429")
    if r.status_code != 200: return []
    out: List[NewsItem] = []
    for a in r.json().get("articles", []):
        dt = datetime.fromisoformat(a.get("publishedAt","").replace("Z","+00:00")) if a.get("publishedAt") else now()
        if dt < since: continue
        title = a.get("title","")
        nid = sha1(f"{title}|{a.get('url','')}")
        out.append(NewsItem(id=nid, src="newsapi", url=a.get("url",""),
                            title=title, summary=a.get("description",""),
                            tickers=infer_tickers(title), published=iso(dt), raw=a))
        if len(out) >= limit: break
    return out

# ── Sentiment (very light, keyword fallback) ──────────────────────────────────
POS = {"beats","beat","surge","soar","record","upgrade","outperform","raise","growth","bullish"}
NEG = {"miss","plunge","fall","drop","cut","downgrade","loss","decline","bearish","probe","lawsuit"}

def score_sentiment(title: str, summary: str) -> float:
    t = f"{title}. {summary}".lower()
    p = sum(1 for w in POS if w in t); n = sum(1 for w in NEG if w in t)
    return max(-1.0, min(1.0, (p-n)/max(1,p+n)))

# ── Dedup, signals, IO ────────────────────────────────────────────────────────
def dedup(items: List[NewsItem]) -> List[NewsItem]:
    seen=set(); out=[]
    for it in sorted(items, key=lambda x: x.published, reverse=True):
        if it.id in seen: continue
        seen.add(it.id); out.append(it)
    return out

def save_json(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def append_jsonl(path: pathlib.Path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_signals(items: List[NewsItem], focus: List[str]) -> Dict[str, dict]:
    buckets: Dict[str, List[NewsItem]] = {}
    for it in items:
        for s in (focus or it.tickers or []):
            if s.isalpha() and 2 <= len(s) <= 6:
                buckets.setdefault(s, []).append(it)

    out: Dict[str, dict] = {}
    for sym, arr in buckets.items():
        s, W = 0.0, 0.0
        for it in arr:
            age_min = max(5.0, (now()-datetime.fromisoformat(it.published)).total_seconds()/60.0)
            w = 1.0 / math.log1p(age_min)
            s += it.score*w; W += w
        raw = s/max(1e-6, W)
        boost = min(0.25, 0.02*len(arr))
        final = max(-1.0, min(1.0, raw + (boost if raw>=0 else -boost)))
        out[sym] = {
            "symbol": sym, "score": round(final, 4), "count": len(arr),
            "last_ts": max(a.published for a in arr),
            "top_titles": [a.title for a in sorted(arr, key=lambda x: x.score, reverse=True)[:5]],
        }
    return out

# ── Runner ────────────────────────────────────────────────────────────────────
def run_pipeline(since_hours: int = 24, symbols: List[str] | None = None, max_items: int = 80):
    since = now() - timedelta(hours=since_hours)
    focus = [s.strip().upper() for s in (symbols or []) if s.strip()]
    goal = max_items

    circuits = {
        "yahoo":   Circuit("yahoo"),
        "finnhub": Circuit("finnhub"),
        "newsapi": Circuit("newsapi"),
    }

    sources = [
        ("yahoo",   lambda: pull_yahoo_rss(since, focus, max_items//2)),
        ("finnhub", lambda: pull_finnhub_company(since, focus, max_items//2)),
        ("newsapi", lambda: pull_newsapi(since, focus, max_items//2)),
    ]

    collected: List[NewsItem] = []
    rotation = 0
    while len(collected) < goal and rotation < 12:
        for name, fn in sources:
            c = circuits[name]
            if not c.allow(): continue
            try:
                items = fn()
                for it in items:
                    # sentiment
                    it.score = score_sentiment(it.title, it.summary)
                    append_jsonl(STATE / "news_raw.jsonl", asdict(it))
                collected.extend(items)
                c.record(ok=True)
                log(f"√ {name:<7} +{len(items):3d} (total={len(collected)})")
                if len(collected) >= goal: break
            except Exception as e:
                c.record(ok=False)
                log(f"⚠ {name} failed: {e} (cooldown active)")
                time.sleep(1.0)
        rotation += 1
        time.sleep(0.5)

    unique = dedup(collected)
    log(f"Dedup → {len(unique)} unique")
    # rewrite unique jsonl
    with (STATE / "news_unique.jsonl").open("w", encoding="utf-8") as f:
        for it in unique:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    signals = build_signals(unique, focus)
    save_json(STATE / "signals.json", {"generated": iso(now()), "n_unique": len(unique), "signals": signals})
    log("Signals saved.")

def main():
    ap = argparse.ArgumentParser(description="Super Pipeline v1")
    ap.add_argument("--since", type=int, default=24, help="Lookback hours")
    ap.add_argument("--symbols", type=str, default="", help="Comma list of tickers")
    ap.add_argument("--max-items", type=int, default=80)
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_pipeline(since_hours=args.since, symbols=syms, max_items=args.max_items)

if __name__ == "__main__":
    main()