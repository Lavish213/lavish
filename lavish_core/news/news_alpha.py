import os
import requests
import logging
from lavish_core.news.collector import BaseCollector

class AlphaCollector(BaseCollector):
    """Collects financial news from Alpha Vantage API."""
    name = "Alpha Vantage"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.url = "https://www.alphavantage.co/query"

    def fetch(self):
        if not self.api_key:
            logging.warning("⚠️ Missing ALPHAVANTAGE_API_KEY, returning mock data.")
            return [{"headline": "No API key set for Alpha Vantage."}]

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "topics": "financial_markets",
        }

        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"headline": item.get("title", "")} for item in data.get("feed", [])[:5]]
        except Exception as e:
            logging.error(f"❌ AlphaCollector failed: {e}")
            return [{"headline": f"Alpha Vantage fetch failed: {e}"}]
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
