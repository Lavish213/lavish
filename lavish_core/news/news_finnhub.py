import os
import requests
import logging
from lavish_core.news.collector import BaseCollector

class FinnhubCollector(BaseCollector):
    """Collects financial news from Finnhub API."""
    name = "Finnhub"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self.url = "https://finnhub.io/api/v1/news?category=general"

    def fetch(self):
        if not self.api_key:
            logging.warning("⚠️ Missing FINNHUB_API_KEY, returning mock data.")
            return [{"headline": "No API key set for Finnhub."}]

        headers = {"X-Finnhub-Token": self.api_key}

        try:
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"headline": item.get("headline", "")} for item in data[:5]]
        except Exception as e:
            logging.error(f"❌ FinnhubCollector failed: {e}")
            return [{"headline": f"Finnhub fetch failed: {e}"}]
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
