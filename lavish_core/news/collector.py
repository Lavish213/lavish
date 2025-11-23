import json
import logging
from lavish_core.news.sentiment import analyze_news_sentiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BaseCollector:
    """
    Base class for all data collectors (Alpha, Finnhub, Crypto).
    Provides shared utilities for logging, sentiment, and JSON serialization.
    """

    name = "base"

    def __init__(self):
        logging.info(f"🛰️ Initializing collector: {self.name}")

    def fetch(self):
        """Override in subclass to implement data fetching logic."""
        raise NotImplementedError("Subclasses must implement fetch()")

    def analyze_sentiment(self, news_items):
        """Run sentiment analysis on a batch of news items."""
        try:
            sentiment = analyze_news_sentiment(news_items)
            return sentiment
        except Exception as e:
            logging.error(f"❌ Sentiment analysis error in {self.name}: {e}")
            return "neutral"

    def to_json(self, data):
        """Convert data safely to JSON string."""
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"❌ Failed to serialize {self.name} data: {e}")
            return "{}"

    def collect(self):
        """Fetch + analyze + serialize data."""
        logging.info(f"⚙️ Collecting data for {self.name}...")
        data = self.fetch()
        sentiment = self.analyze_sentiment(data)
        result = {"collector": self.name, "sentiment": sentiment, "items": data}
        return self.to_json(result)

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
