#!/usr/bin/env python3
"""
Lavish — News Collectors Runner + Safe Fallback Collector

- Runs all configured news collectors (Alpha, Finnhub, Crypto)
- Writes outputs into data/cache as JSON
- Handles collectors that return:
    • raw JSON string
    • Python dict/list
    • list of Article objects
- Provides a generic Article dataclass + Collector fallback so
  other modules can import from here without blowing up.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import hashlib

# ─────────────────────────── Paths ────────────────────────────
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # repo root
if ROOT not in sys.path:
    sys.path.append(ROOT)

CACHE_DIR = os.path.join(ROOT, "data", "cache")

# ─────────────────────── Fallback models ──────────────────────
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
        return hashlib.sha256(f"{self.source}{self.title}{self.url}".encode("utf-8")).hexdigest()


class Collector:
    """
    Universal fallback Collector to prevent missing-collector errors.

    Real collectors should subclass this or just implement .collect()
    returning a list[Article] / dict / list / JSON string.
    """

    def __init__(self, timeout: int = 10, max_items: int = 10, logger: logging.Logger | None = None):
        self.timeout = timeout
        self.max_items = max_items
        self.logger = logger

    def collect(self):
        """Return a placeholder article so the master collector runs cleanly."""
        if self.logger:
            self.logger.debug(f"[{__name__}] Running fallback collector (no API logic yet).")
        return [
            Article(
                source=__name__.split(".")[-1],
                title="Placeholder article — collector not implemented yet",
                url="https://placeholder.example.com",
                ts="2025-10-27T00:00:00Z",
                summary="This is a fallback entry until this collector is customized.",
                tickers=[],
            )
        ]


# ─────────────────── Try importing real Article ───────────────
# If run_master_collector defines its own Article, we prefer it,
# but if the import fails we keep our local dataclass above.
try:
    from lavish_core.news.run_master_collector import Article as MasterArticle  # type: ignore

    Article = MasterArticle  # override with master version if available
except Exception:
    # keep local Article definition
    pass

# ───────────────────── Real collectors import ─────────────────
from lavish_core.news.news_alpha import AlphaCollector
from lavish_core.news.news_finnhub import FinnhubCollector
from lavish_core.news.news_crypto import CryptoCollector

# ───────────────────────── Logging setup ──────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lavish.news.run_collectors")

# ───────────────────── Cache helper(s) ────────────────────────
def _normalize_to_json(data: Any) -> str:
    """
    Accepts:
      - str (assumed to already be JSON or text)
      - dict / list
      - list[Article] or Article
    and returns a pretty JSON string for disk.
    """
    # Already a string
    if isinstance(data, str):
        return data

    # One Article
    if isinstance(data, Article):
        return json.dumps(asdict(data), ensure_ascii=False, indent=2)

    # List of Articles
    if isinstance(data, list) and data and isinstance(data[0], Article):
        return json.dumps([asdict(a) for a in data], ensure_ascii=False, indent=2)

    # dict / list / other JSON-friendly types
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dict__") else str(o))
    except TypeError:
        # absolute fallback – stringify
        return json.dumps({"value": str(data)}, ensure_ascii=False, indent=2)


def save_to_cache(filename: str, data: Any) -> None:
    """Save data to JSON file in cache folder (safe for multiple types)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    try:
        payload = _normalize_to_json(data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
        log.info("✅ Saved to cache: %s", path)
    except Exception as e:
        log.error("❌ Failed to save cache file %s: %s", filename, e)


# ───────────────────── Main collector runner ──────────────────
def run_all_collectors() -> None:
    """Run all active data collectors safely."""
    print("\n🚀 Starting Lavish Bot Data Collectors...\n")

    collectors = [
        ("AlphaCollector", AlphaCollector),
        ("FinnhubCollector", FinnhubCollector),
        ("CryptoCollector", CryptoCollector),
    ]

    results: Dict[str, str] = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, collector_cls in collectors:
        try:
            log.info("⚙️  Running %s…", name)
            collector = collector_cls()
            data = collector.collect()
            fname = f"{name.lower()}_{ts}.json"
            save_to_cache(fname, data)
            results[name] = "✅ Success"
        except Exception as e:
            results[name] = f"❌ Failed: {e}"
            log.exception("%s crashed: %s", name, e)

    print("\n✨ All collectors finished.\n")
    for name, status in results.items():
        print(f"   {status} → {name}")

    print("\n🗂️  Cache files stored in:", os.path.abspath(CACHE_DIR))
    print("--------------------------------------------------")


# ───────────────────────── Entrypoint ─────────────────────────
if __name__ == "__main__":
    run_all_collectors()