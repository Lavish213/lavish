# Covers the PDT day-trade counting and blocking logic verified manually
# earlier this project: open positions don't count, overnight holds don't
# count, multiple real day trades across symbols sum correctly, small
# accounts get blocked, large accounts bypass, and a broker error fails
# open rather than silently blocking every trade.
from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.trade import pdt_guard


@pytest.fixture
def store(tmp_path):
    return HybridStore(duckdb_path=tmp_path / "test.duckdb")


def _fill(store, symbol, side, ts, price=100.0, qty=1.0):
    store.execute(
        "INSERT INTO fills (id, ts, order_id, symbol, side, qty, price, fee, venue) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), ts, str(uuid.uuid4()), symbol, side, qty, price, 0.0, "test"),
    )


def test_open_position_not_a_day_trade(store):
    today = date.today()
    _fill(store, "AAPL", "buy", today)
    assert pdt_guard.count_recent_day_trades(store, as_of=today) == 0


def test_overnight_hold_not_a_day_trade(store):
    today = date.today()
    yesterday = today - timedelta(days=1)
    _fill(store, "AAPL", "buy", yesterday)
    _fill(store, "AAPL", "sell", today)
    assert pdt_guard.count_recent_day_trades(store, as_of=today) == 0


def test_same_day_round_trip_counts_as_one(store):
    today = date.today()
    _fill(store, "AAPL", "buy", today)
    _fill(store, "AAPL", "sell", today)
    assert pdt_guard.count_recent_day_trades(store, as_of=today) == 1


def test_multiple_symbols_sum_correctly(store):
    today = date.today()
    for sym in ("AAPL", "TSLA", "SPY"):
        _fill(store, sym, "buy", today)
        _fill(store, sym, "sell", today)
    assert pdt_guard.count_recent_day_trades(store, as_of=today) == 3


def test_trade_outside_lookback_window_not_counted(store):
    today = date.today()
    old = today - timedelta(days=30)
    _fill(store, "AAPL", "buy", old)
    _fill(store, "AAPL", "sell", old)
    assert pdt_guard.count_recent_day_trades(store, as_of=today) == 0


def test_small_account_blocks_after_max_day_trades(store):
    today = date.today()
    for _ in range(pdt_guard.PDT_MAX_DAY_TRADES):
        sym = f"T{uuid.uuid4().hex[:4]}"
        _fill(store, sym, "buy", today)
        _fill(store, sym, "sell", today)
    with patch.object(pdt_guard, "get_account", return_value={"equity": "10000"}):
        ok, reason = pdt_guard.check_pdt_ok(store)
    assert ok is False
    assert "pdt_guard" in reason


def test_large_account_bypasses_check(store):
    today = date.today()
    for _ in range(pdt_guard.PDT_MAX_DAY_TRADES + 2):
        sym = f"T{uuid.uuid4().hex[:4]}"
        _fill(store, sym, "buy", today)
        _fill(store, sym, "sell", today)
    with patch.object(pdt_guard, "get_account", return_value={"equity": "100000"}):
        ok, _ = pdt_guard.check_pdt_ok(store)
    assert ok is True


def test_broker_error_fails_open(store):
    with patch.object(pdt_guard, "get_account", side_effect=RuntimeError("broker down")):
        ok, reason = pdt_guard.check_pdt_ok(store)
    assert ok is True
    assert "unavailable" in reason


def test_explicit_opt_out_via_zero_threshold(store, monkeypatch):
    monkeypatch.setattr(pdt_guard, "PDT_EQUITY_THRESHOLD", 0)
    ok, reason = pdt_guard.check_pdt_ok(store)
    assert ok is True
    assert "disabled" in reason
