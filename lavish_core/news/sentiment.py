from transformers import pipeline
import logging

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize the model once globally for performance
try:
    _sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logging.info("✅ Sentiment model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load sentiment model: {e}")
    _sentiment = None


def analyze_news_sentiment(news_items):
    """
    Analyze sentiment of multiple news items.
    
    Args:
        news_items (list[dict]): List of news articles with 'headline' keys.

    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    if not news_items or not isinstance(news_items, list):
        logging.warning("⚠️ No valid news items provided.")
        return "neutral"

    if _sentiment is None:
        logging.warning("⚠️ Sentiment model not available.")
        return "neutral"

    # Collect top headlines and join them into a single string
    headlines = [item.get("headline", "") for item in news_items if "headline" in item]
    if not headlines:
        return "neutral"

    text = " ".join(headlines[:5])  # Analyze top 5 for broader tone

    try:
        results = _sentiment(text)
        if not results:
            return "neutral"

        label = results[0].get("label", "").lower()

        # Normalize label variations
        if "pos" in label:
            return "positive"
        elif "neg" in label:
            return "negative"
        else:
            return "neutral"

    except Exception as e:
        logging.error(f"❌ Sentiment analysis failed: {e}")
        return "neutral"
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
