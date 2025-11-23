"""🔥 Lavish AI — Final Buffed Crypto News Collector (Free, No API Key)
------------------------------------------------------------------
Features:
  ✅ No paid API keys required.
  ✅ Pulls from multiple free reliable sources:
       - CoinDesk RSS
       - Yahoo Finance Crypto News
       - CryptoSlate RSS
       - CoinTelegraph RSS
  ✅ Logs every action, warning, and result.
  ✅ Handles network drops, site errors, and bad responses automatically.
  ✅ Saves to cache (data/cache/crypto_news.json)
  ✅ Uses local cache when offline or sources fail.
"""

import os
import json
import time
import logging
import requests
import feedparser
from datetime import datetime, timezone

# -------------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------------
logger = logging.getLogger("LavishCryptoCollector")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# -------------------------------------------------------------------
# Collector Class
# -------------------------------------------------------------------
class LavishCryptoCollector:
    def __init__(self):
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "crypto_news.json")

    # ---------------------------------------------------------------
    # Safe Request Wrapper
    # ---------------------------------------------------------------
    def safe_get(self, url, expect_json=False):
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "LavishNewsBot/1.0"})
            if response.status_code == 200:
                return response.json() if expect_json else response.text
            else:
                logger.warning(f"⚠️ Bad response from {url}: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Error requesting {url}: {e}")
            return None

    # ---------------------------------------------------------------
    # Fetch CoinDesk RSS
    # ---------------------------------------------------------------
    def fetch_coindesk(self):
        logger.info("📡 Fetching from CoinDesk RSS...")
        feed_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            for entry in feed.entries[:15]:
                articles.append({
                    "source": "CoinDesk",
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                })
            logger.info(f"✅ CoinDesk: {len(articles)} articles.")
            return articles
        except Exception as e:
            logger.error(f"❌ CoinDesk failed: {e}")
            return []

    # ---------------------------------------------------------------
    # Fetch Yahoo Finance Crypto
    # ---------------------------------------------------------------
    def fetch_yahoo_finance(self):
        logger.info("📡 Fetching from Yahoo Finance Crypto...")
        url = "https://finance.yahoo.com/topic/crypto/"
        html = self.safe_get(url)
        if not html:
            return []
        articles = []
        seen = set()
        for line in html.splitlines():
            if '"title":"' in line and '"url":"' in line:
                try:
                    title = line.split('"title":"')[1].split('"')[0]
                    link = line.split('"url":"')[1].split('"')[0]
                    if not link.startswith("http"):
                        link = "https://finance.yahoo.com" + link
                    if title not in seen:
                        articles.append({
                            "source": "Yahoo Finance",
                            "title": title,
                            "url": link,
                            "published_at": datetime.now(timezone.utc).isoformat(),
                        })
                        seen.add(title)
                except Exception:
                    continue
        logger.info(f"✅ Yahoo Finance: {len(articles)} articles.")
        return articles[:15]

    # ---------------------------------------------------------------
    # Fetch CryptoSlate RSS
    # ---------------------------------------------------------------
    def fetch_cryptoslate(self):
        logger.info("📡 Fetching from CryptoSlate RSS...")
        url = "https://cryptoslate.com/feed/"
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:15]:
                articles.append({
                    "source": "CryptoSlate",
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                })
            logger.info(f"✅ CryptoSlate: {len(articles)} articles.")
            return articles
        except Exception as e:
            logger.error(f"❌ CryptoSlate failed: {e}")
            return []

    # ---------------------------------------------------------------
    # Fetch CoinTelegraph RSS
    # ---------------------------------------------------------------
    def fetch_cointelegraph(self):
        logger.info("📡 Fetching from CoinTelegraph RSS...")
        url = "https://cointelegraph.com/rss"
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:15]:
                articles.append({
                    "source": "CoinTelegraph",
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "published_at": entry.get("published", ""),
                })
            logger.info(f"✅ CoinTelegraph: {len(articles)} articles.")
            return articles
        except Exception as e:
            logger.error(f"❌ CoinTelegraph failed: {e}")
            return []

    # ---------------------------------------------------------------
    # Merge & Save Cache
    # ---------------------------------------------------------------
    def save_cache(self, articles):
        if not articles:
            logger.warning("⚠️ No articles to save. Keeping old cache.")
            return False

        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(articles)} articles → {self.cache_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to write cache: {e}")
            return False

    # ---------------------------------------------------------------
    # Load Cache (Offline Mode)
    # ---------------------------------------------------------------
    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    # ---------------------------------------------------------------
    # Main Runner
    # ---------------------------------------------------------------
    def collect_all(self):
        logger.info("🚀 Starting Lavish Crypto News Collector...")
        all_articles = []

        # Priority order (fastest + most stable first)
        sources = [
            self.fetch_coindesk,
            self.fetch_yahoo_finance,
            self.fetch_cryptoslate,
            self.fetch_cointelegraph
        ]

        for fetch_fn in sources:
            try:
                fetched = fetch_fn()
                if fetched:
                    all_articles.extend(fetched)
            except Exception as e:
                logger.error(f"⚠️ Source {fetch_fn.__name__} failed: {e}")

        # Deduplicate by title
        unique_articles = {a["title"]: a for a in all_articles}.values()
        unique_articles = sorted(unique_articles, key=lambda x: x.get("published_at", ""), reverse=True)

        if not unique_articles:
            logger.warning("⚠️ All sources failed, attempting cache load.")
            cached = self.load_cache()
            if cached:
                logger.info(f"🗂 Loaded {len(cached)} cached articles.")
                return cached
            else:
                logger.error("❌ No live or cached data available.")
                return []

        self.save_cache(list(unique_articles))
        logger.info(f"✅ Finished — {len(unique_articles)} total unique crypto articles.")
        return list(unique_articles)

# -------------------------------------------------------------------
# Run if main
# -------------------------------------------------------------------
if __name__ == "__main__":
    collector = LavishCryptoCollector()
    collector.collect_all()
    # --- Auto Fallback Collector (safe default) ---
try:
    from lavish_core.news.run_master_collector import Article
except Exception:
    # local fallback for testing
    from datetime import datetime, timezone
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
