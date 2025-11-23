from __future__ import annotations
import os, time, logging, json
from typing import Dict, List, Any, Optional
from pathlib import Path

import requests
from lavish_core.config import get_settings

log = logging.getLogger("news_failover")

DATA_DIR = Path("data")
LOG_DIR  = Path("logs")
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

NEWS_OUT_JSON = DATA_DIR / "news.json"
NEWS_OUT_CSV  = DATA_DIR / "news.csv"

def _write_json(rows: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def _append_csv(rows: List[Dict[str, Any]], path: Path):
    import csv
    cols = ["datetime","source","headline","url","summary"]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

def _req(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        if r.status_code == 429:
            log.warning("429 from %s — failover to next source", url)
            return None
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning("request error %s -> %s", url, e)
        return None

class NewsFailover:
    """
    Pulls from multiple sources until we hit a target row count.
    Circuit-breaker: if a source errors >= NEWS_FAIL_CUTOFF, it is skipped this run.
    """
    def __init__(self):
        s = get_settings()
        self.timeout = s.NEWS_TIMEOUT
        self.fail_cutoff = s.NEWS_FAIL_CUTOFF
        self.sources = self._build_sources()

    def _build_sources(self):
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        polygon_key = os.getenv("POLYGON_API_KEY", "")

        out = []
        if finnhub_key:
            out.append({
                "name": "finnhub",
                "fn": self._finnhub,
                "fails": 0
            })
        if polygon_key:
            out.append({
                "name": "polygon",
                "fn": self._polygon,
                "fails": 0
            })
        # always-available fallbacks
        out.append({"name":"yahoo_rss", "fn": self._yahoo_rss, "fails": 0})
        out.append({"name":"reddit_stocks", "fn": self._reddit_stocks, "fails": 0})

        return out

    # ---- SOURCE IMPLEMENTATIONS ---------------------------------------------
    def _finnhub(self, want: int) -> List[Dict[str, Any]]:
        url = "https://finnhub.io/api/v1/news"
        key = os.getenv("FINNHUB_API_KEY","")
        r = _req(url, {"category":"general", "token": key}, timeout=self.timeout)
        if not r: return []
        data = r.json() or []
        rows = []
        for n in data[:want]:
            rows.append({
                "datetime": n.get("datetime",""),
                "source": n.get("source","finnhub"),
                "headline": n.get("headline",""),
                "url": n.get("url",""),
                "summary": n.get("summary","")
            })
        return rows

    def _polygon(self, want: int) -> List[Dict[str, Any]]:
        key = os.getenv("POLYGON_API_KEY","")
        url = "https://api.polygon.io/v2/reference/news"
        r = _req(url, {"limit": min(50, want), "apiKey": key}, timeout=self.timeout)
        if not r: return []
        data = (r.json() or {}).get("results", [])
        rows = []
        for n in data[:want]:
            rows.append({
                "datetime": n.get("published_utc",""),
                "source": (n.get("publisher") or {}).get("name","polygon"),
                "headline": n.get("title",""),
                "url": n.get("article_url",""),
                "summary": n.get("description",""),
            })
        return rows

    def _yahoo_rss(self, want: int) -> List[Dict[str, Any]]:
        # Simple RSS pulls that rarely rate-limit
        import xml.etree.ElementTree as ET
        urls = [
            "https://finance.yahoo.com/rss/topstories",
            "https://finance.yahoo.com/news/rssindex"
        ]
        out: List[Dict[str, Any]] = []
        for url in urls:
            r = _req(url, timeout=self.timeout)
            if not r: 
                continue
            try:
                root = ET.fromstring(r.text)
                for item in root.findall(".//item"):
                    out.append({
                        "datetime": item.findtext("pubDate",""),
                        "source": "yahoo_finance",
                        "headline": item.findtext("title",""),
                        "url": item.findtext("link",""),
                        "summary": item.findtext("description",""),
                    })
            except Exception:
                continue
            if len(out) >= want:
                break
        return out[:want]

    def _reddit_stocks(self, want: int) -> List[Dict[str, Any]]:
        urls = [
            "https://www.reddit.com/r/stocks/new.json?limit=50",
            "https://www.reddit.com/r/wallstreetbets/new.json?limit=50",
        ]
        out: List[Dict[str, Any]] = []
        headers = {"User-Agent": "LavishBot/1.0"}
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    continue
                r.raise_for_status()
                js = r.json()
                for post in (js.get("data",{}).get("children",[]) or []):
                    d = post.get("data",{})
                    out.append({
                        "datetime": d.get("created_utc",""),
                        "source": "reddit",
                        "headline": d.get("title",""),
                        "url": "https://www.reddit.com" + d.get("permalink",""),
                        "summary": d.get("selftext","")[:400],
                    })
            except Exception:
                continue
            if len(out) >= want:
                break
        return out[:want]

    # ---- PUBLIC --------------------------------------------------------------
    async def fetch_and_save_min(self, min_rows: int = 120) -> List[Dict[str, Any]]:
        """
        Walk sources until min_rows is reached. Circuit-break sources that exceed failure cutoff.
        """
        collected: List[Dict[str, Any]] = []
        for src in self.sources:
            if src["fails"] >= self.fail_cutoff:
                log.info("⛔ skipping %s (circuit-open)", src["name"])
                continue
            try:
                need = max(0, min_rows - len(collected))
                if need <= 0:
                    break
                log.info("📰 %s fetching (need %d)…", src["name"], need)
                rows = src["fn"](need)
                if rows:
                    collected.extend(rows)
                else:
                    src["fails"] += 1
            except Exception as e:
                src["fails"] += 1
                log.warning("%s failed: %s", src["name"], e)

        if not collected:
            log.warning("All primary news sources failed — falling back to data/feeds_news.py if present.")
            try:
                import subprocess
                subprocess.run(["python", "data/feeds_news.py"], check=False)
            except Exception:
                pass

        # persist
        if collected:
            _write_json(collected, NEWS_OUT_JSON)
            _append_csv(collected, NEWS_OUT_CSV)
            log.info("🗞️ saved %d news rows -> %s / %s", len(collected), NEWS_OUT_JSON, NEWS_OUT_CSV)
        return collected


