#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lavish Super Vector — single-file, SQLite + Alpaca (IEX) live data ingestor,
feature computer, and simple strategy runner.

Key fixes vs your previous version:
  • Always requests FREE IEX feed by setting feed=DataFeed.IEX on the request
    (avoids “subscription does not permit querying recent SIP data”).
  • No datetime.now(timezone.utc) deprecation — uses timezone-aware datetime.now(timezone.utc).
  • Tenacity retries around network calls.
  • Clean env-config, with sane defaults.
  • Works for Python 3.10+ and alpaca-py>=0.23.

ENV you can export (examples):
  ALPACA_KEY_ID=xxx
  ALPACA_SECRET_KEY=xxx
  ALPACA_PAPER=true              # true=sandbox, false=live
  SV_DB_PATH=lavish_bot.db
  SV_TICKERS=AAPL,MSFT,TSLA,AMD,NVDA
  SV_STRATEGY=roc                # roc | sma | pairs
  SV_PAIRS=AAPL:MSFT
  SV_INGEST_SECONDS=5
  SV_AUTO_TRAIN_HOURS=3
"""

from __future__ import annotations

import os
import time
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Tuple

import pandas as pd
from sqlalchemy import (
    create_engine, text,
    Column, String, Float, DateTime, Integer, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Alpaca v2 (alpaca-py >= 0.23)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.common.exceptions import APIError


# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("lavish-super-vector")


# ---------- configuration ----------
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")

@dataclass(frozen=True)
class CFG:
    db_path: str = os.getenv("SV_DB_PATH", "lavish_bot.db")
    tickers: Tuple[str, ...] = tuple([t.strip().upper() for t in os.getenv(
        "SV_TICKERS", "AAPL,MSFT,TSLA,AMD,NVDA"
    ).split(",") if t.strip()])

    # strategy options
    strategy: str = os.getenv("SV_STRATEGY", "roc").strip().lower()  # roc | sma | pairs
    pairs: Tuple[str, str] = tuple(
        [s.strip().upper() for s in os.getenv("SV_PAIRS", "AAPL:MSFT").split(":")]
    ) if os.getenv("SV_PAIRS") else tuple()

    # ingest cadence
    ingest_seconds: int = int(os.getenv("SV_INGEST_SECONDS", "5"))
    auto_train_hours: int = int(os.getenv("SV_AUTO_TRAIN_HOURS", "3"))

    # alpaca creds
    alpaca_key: str = os.getenv("ALPACA_KEY_ID", "")
    alpaca_secret: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_paper: bool = _env_bool("ALPACA_PAPER", True)

CFG = CFG()
log.info("Config: %s", CFG)

if not CFG.alpaca_key or not CFG.alpaca_secret:
    raise SystemExit("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY environment variables.")

# base url (only needed if you also use trading — for data client it’s fine)
if CFG.alpaca_paper and not os.getenv("APCA_API_BASE_URL"):
    # Not strictly required for HistoricalDataClient, but harmless to set.
    os.environ["APCA_API_BASE_URL"] = "https://data.sandbox.alpaca.markets"

# ---------- DB ----------
Base = declarative_base()

class Bar(Base):
    __tablename__ = "bars"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "ts", name="uq_symbol_ts"),
    )

engine = create_engine(f"sqlite:///{CFG.db_path}", future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# ---------- Alpaca data client ----------
alpaca_data = StockHistoricalDataClient(api_key=CFG.alpaca_key, secret_key=CFG.alpaca_secret)

@retry(
    reraise=True,
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
)
def _get_bars(symbols: Iterable[str],
              start: datetime | None = None,
              end: datetime | None = None,
              limit: int | None = None) -> pd.DataFrame:
    """
    Fetch minute bars using the FREE IEX feed (no SIP).
    IMPORTANT: The 'feed' is set on the request (DataFeed.IEX), not on the client.
    """
    if start is None and limit is None:
        # default to last 180 minutes if not specified
        end = datetime.now(timezone.utc) if end is None else end
        start = end - timedelta(minutes=180)

    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,           # <—— FIX: force IEX (free)
        limit=limit
    )
    # Alpaca returns a BarsResponse mapping -> convert to pandas in one go
    resp = alpaca_data.get_stock_bars(req)
    df = resp.df  # Multi-index (symbol, timestamp)
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol","ts","open","high","low","close","volume"])
    df = df.reset_index().rename(columns={"timestamp": "ts", "symbol": "symbol"})
    # Ensure timezone-aware timestamps
    if df["ts"].dtype.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(timezone.utc)
    return df[["symbol","ts","open","high","low","close","volume"]].sort_values(["symbol","ts"])


def fetch_recent_1m_bars(symbols: Iterable[str], minutes: int = 180) -> pd.DataFrame:
    """Convenience wrapper pulling the last `minutes` minutes with IEX feed."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return _get_bars(symbols, start=start, end=end, limit=None)


def upsert_bars(df: pd.DataFrame) -> int:
    """Upsert bars into SQLite."""
    if df is None or df.empty:
        return 0
    with engine.begin() as conn:
        # Use INSERT OR IGNORE; then UPDATE for changed rows (SQLite doesn't have real upsert pre-3.24).
        # For our use, minute bars are immutable, so IGNORE is enough.
        rows = 0
        for chunk_start in range(0, len(df), 1000):
            chunk = df.iloc[chunk_start: chunk_start + 1000]
            conn.exec_driver_sql(
                """
                INSERT OR IGNORE INTO bars(symbol, ts, open, high, low, close, volume)
                VALUES (:symbol, :ts, :open, :high, :low, :close, :volume)
                """,
                chunk.to_dict(orient="records"),
            )
            rows += len(chunk)
    return rows


# ---------- feature engineering ----------
def load_bars(symbol: str, lookback_minutes: int = 1000) -> pd.DataFrame:
    """Load recent bars from DB for one symbol."""
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes+5)
    q = text("""
        SELECT symbol, ts, open, high, low, close, volume
        FROM bars
        WHERE symbol = :symbol AND ts >= :since
        ORDER BY ts ASC
    """)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"symbol": symbol, "since": since})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    return df

def compute_roc(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    return df["close"].pct_change(window) * 100.0

def compute_sma(df: pd.DataFrame, n: int) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    return df["close"].rolling(n).mean()

def compute_pairs_spread(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.Series:
    if a_df.empty or b_df.empty: return pd.Series(dtype=float)
    # align on timestamps
    joined = a_df[["close"]].rename(columns={"close": "A"}).join(
        b_df[["close"]].rename(columns={"close": "B"}), how="inner"
    )
    return joined["A"] - joined["B"]


# ---------- strategies (signals only; no live trading here) ----------
def run_strategy_roc(symbols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        df = load_bars(s, lookback_minutes=600)
        if df.empty or len(df) < 30:
            continue
        roc = compute_roc(df, window=20)
        last = roc.dropna().iloc[-1] if not roc.dropna().empty else float("nan")
        rows.append((s, float(last)))
    out = pd.DataFrame(rows, columns=["symbol","roc20"]).sort_values("roc20", ascending=False)
    return out

def run_strategy_sma(symbols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        df = load_bars(s, lookback_minutes=600)
        if df.empty or len(df) < 50:
            continue
        sma_fast = compute_sma(df, 20)
        sma_slow = compute_sma(df, 50)
        cross = (sma_fast.iloc[-1] > sma_slow.iloc[-1]) if not math.isnan(sma_fast.iloc[-1]) and not math.isnan(sma_slow.iloc[-1]) else False
        rows.append((s, cross, float(sma_fast.iloc[-1]), float(sma_slow.iloc[-1])))
    return pd.DataFrame(rows, columns=["symbol","bullish","sma20","sma50"]).sort_values("bullish", ascending=False)

def run_strategy_pairs(a: str, b: str) -> pd.DataFrame:
    a_df = load_bars(a, lookback_minutes=600)
    b_df = load_bars(b, lookback_minutes=600)
    spread = compute_pairs_spread(a_df, b_df)
    if spread.empty:
        return pd.DataFrame(columns=["ts","spread","z"])
    z = (spread - spread.rolling(100).mean()) / (spread.rolling(100).std())
    out = pd.DataFrame({"ts": spread.index, "spread": spread.values, "z": z.values})
    return out.tail(5)


# ---------- orchestration ----------
def ensure_tickers() -> Tuple[str, ...]:
    tickers = tuple([t for t in CFG.tickers if t])
    if CFG.strategy == "pairs":
        if len(CFG.pairs) != 2:
            raise SystemExit("SV_STRATEGY=pairs requires SV_PAIRS like AAPL:MSFT")
        for p in CFG.pairs:
            if p not in tickers:
                tickers += (p,)
    return tickers

def ingest_once() -> None:
    """Pull latest few minutes for all tickers and upsert to DB."""
    tickers = ensure_tickers()
    try:
        df = fetch_recent_1m_bars(tickers, minutes=30)
        n = upsert_bars(df)
        log.info("Ingested %s rows.", n)
    except APIError as e:
        # If you ever see SIP again, this makes it obvious:
        log.error("Alpaca APIError: %s", e)

def train_if_due(last_trained: List[datetime]) -> None:
    """Dummy hook to represent 'auto-train' every N hours."""
    now = datetime.now(timezone.utc)
    if not last_trained or (now - last_trained[0]) >= timedelta(hours=CFG.auto_train_hours):
        last_trained[:] = [now]
        # put your ML/retraining code here if you want
        log.info("Auto-train hook executed (every %d h).", CFG.auto_train_hours)

def run_signals_preview() -> None:
    """Just prints top-of-book signals so you can see something running."""
    if CFG.strategy == "roc":
        table = run_strategy_roc(CFG.tickers)
        log.info("ROC snapshot:\n%s", table.head(10).to_string(index=False))
    elif CFG.strategy == "sma":
        table = run_strategy_sma(CFG.tickers)
        log.info("SMA snapshot:\n%s", table.head(10).to_string(index=False))
    elif CFG.strategy == "pairs":
        a, b = CFG.pairs
        table = run_strategy_pairs(a, b)
        log.info("Pairs snapshot (%s-%s):\n%s", a, b, table.to_string(index=False))
    else:
        log.warning("Unknown strategy '%s'", CFG.strategy)


def main() -> None:
    tickers = ensure_tickers()
    log.info("Starting ingest for: %s", ", ".join(tickers))

    while True:
        now_et = datetime.now(timezone.utc)

        # ✅ Intelligent scheduling based on market hours (roughly 9AM–6PM ET)
        if 13 <= now_et.hour <= 22:
            log.info("Market is open → fetching recent 3 hours of data.")
            df = fetch_recent_1m_bars(tickers, minutes=180)
            upsert_bars(df)
            run_signals_preview()
            train_if_due([now_et])

            # ⏰ Update every 30 minutes when market is open
            log.info("Sleeping 30 minutes until next market update...")
            time.sleep(1800)
        else:
            log.info("Market closed → pulling last 24 hours of data for backfill.")
            df = fetch_recent_1m_bars(tickers, minutes=1440)
            upsert_bars(df)
            run_signals_preview()

            # ⏰ Check once per hour when market is closed
            log.info("Sleeping 60 minutes while market is closed...")
            time.sleep(3600)


if __name__ == "__main__":
    main()