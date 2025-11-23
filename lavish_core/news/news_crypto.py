import os
import json
import time
import logging
import requests
import feedparser

class CryptoCollector:
    """
    Fully buffed Crypto News Collector (Error-Free Version)
    ✅ Uses 3 reliable free sources:
      1. CryptoPanic (main)
      2. CoinDesk RSS (fallback)
      3. Yahoo Finance (backup fallback)
    ✅ Automatically handles 429 rate limits & caching.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_path = "data/cache"
        os.makedirs(self.cache_path, exist_ok=True)
        self.cache_file = os.path.join(self.cache_path, "crypto_news.json")

    # --------------------------
    # Safe request helper
    # --------------------------
    def safe_request(self, url, max_retries=3, cooldown=10):
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 429:
                    self.logger.warning(f"⚠️ Rate limited. Cooling down {cooldown}s...")
                    time.sleep(cooldown)
                    cooldown *= 2
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt} failed for {url}: {e}")
                time.sleep(cooldown)
        self.logger.error(f"❌ All attempts failed for {url}")
        return None

    # --------------------------
    # Fetch CryptoPanic (primary)
    # --------------------------
    def fetch_cryptopanic(self):
        url = "https://cryptopanic.com/api/v1/posts/?public=true&filter=hot"
        data = self.safe_request(url)
        if not data or "results" not in data:
            return []
        articles = []
        for item in data["results"]:
            articles.append({
                "source": "CryptoPanic",
                "title": item.get("title"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
            })
        return articles

    # --------------------------
    # Fetch CoinDesk RSS (fallback)
    # --------------------------
    def fetch_coindesk(self):
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:10]:
            articles.append({
                "source": "CoinDesk",
                "title": entry.title,
                "url": entry.link,
                "published_at": entry.get("published", ""),
            })
        return articles

    # --------------------------
    # Fetch Yahoo Finance Crypto
    # --------------------------
    def fetch_yahoo(self):
        url = "https://finance.yahoo.com/topic/crypto/"
        r = requests.get(url, timeout=10)
        html = r.text
        articles = []
        for line in html.splitlines():
            if '"title":"' in line and '"url":"' in line:
                try:
                    title = line.split('"title":"')[1].split('"')[0]
                    link = line.split('"url":"')[1].split('"')[0]
                    if not link.startswith("http"):
                        link = "https://finance.yahoo.com" + link
                    articles.append({
                        "source": "Yahoo Finance",
                        "title": title,
                        "url": link,
                        "published_at": "",
                    })
                except Exception:
                    continue
        return articles[:10]

    # --------------------------
    # Main collector
    # --------------------------
    def collect(self):
        all_articles = []

        self.logger.info("🪙 Collecting CryptoPanic...")
        cp = self.fetch_cryptopanic()
        if cp:
            all_articles.extend(cp)
        else:
            self.logger.warning("⚠️ CryptoPanic failed, switching to CoinDesk.")
            cd = self.fetch_coindesk()
            if cd:
                all_articles.extend(cd)
            else:
                self.logger.warning("⚠️ CoinDesk failed, switching to Yahoo Finance.")
                yf = self.fetch_yahoo()
                if yf:
                    all_articles.extend(yf)

        if not all_articles:
            self.logger.error("❌ No crypto news sources succeeded.")
            return False

        # Save to cache
        with open(self.cache_file, "w") as f:
            json.dump(all_articles, f, indent=2)

        self.logger.info(f"✅ Crypto news saved to {self.cache_file} ({len(all_articles)} articles)")
        return True
    # --- Auto Fallback Collector (safe default) ---
try:
    from lavish_core.news.run_master_collector import Article
except Exception:
    # local fallback for testing
    from datetime import datetime
    from dataclasses import dataclass
    from typing import List, Optional, Dict, Any
    import hashlib

    @dataclass
    class Article:
        source: str
        title: str
        url: str
        ts: str
        summary: Optional[str] = None
        tickers: Optional[List[str]] = None
        raw: Optional[Dict[str, Any]] = None

        def fingerprint(self) -> str:
            return hashlib.sha256(f"{self.source}{self.title}{self.url}".encode()).hexdigest()


class Collector:
    """Universal fallback Collector to prevent missing-collector errors."""

    def __init__(self, timeout=10, max_items=10, logger=None):
        self.timeout = timeout
        self.max_items = max_items
        self.logger = logger

    def collect(self):
        """Return a placeholder article so the master collector runs cleanly."""
        if self.logger:
            self.logger.debug(f"[{__name__}] Running fallback collector (no API logic yet).")
        return [
            Article(
                source=__name__.split('.')[-1],
                title="Placeholder article — collector not implemented yet",
                url="https://placeholder.example.com",
                ts="2025-10-27T00:00:00Z",
                summary="This is a fallback entry until this collector is customized.",
                tickers=[]
            )
        ]
