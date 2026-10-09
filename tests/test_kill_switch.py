# Covers the kill switch: a real, remotely-triggerable "stop all new
# entries" control (see kill_switch.py), reachable without a redeploy.
# Blocks new BUY entries immediately; exits/sells are never affected -
# a kill switch that could trap you IN a position would be worse than no
# kill switch at all.
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import lavish_core.trading.trade_handler as th
from lavish_core.db.hybrid_store import HybridStore


def test_kill_switch_defaults_to_off(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    assert store.get_kill_switch() == (False, "")


def test_kill_switch_on_off_round_trip(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "manual review")
    assert store.get_kill_switch() == (True, "manual review")
    store.set_kill_switch(False, "")
    assert store.get_kill_switch() == (False, "")


def test_kill_switch_update_overwrites_previous_reason(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "first reason")
    store.set_kill_switch(True, "second reason")
    assert store.get_kill_switch() == (True, "second reason")


def test_equity_buy_is_blocked_when_kill_switch_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "testing a halt")

    signal = {"source": "patreon", "action": "BUY", "symbol": "AAPL", "confidence": 0.9}
    with patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(signal)

    assert handled is True  # deterministic, deliberate skip
    mock_place.assert_not_called()


def test_equity_sell_is_never_blocked_by_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "testing a halt")

    signal = {"source": "patreon", "action": "SELL", "symbol": "AAPL", "confidence": 0.9}
    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", return_value={"qty": "10"}), \
         patch.object(th, "place_trade", return_value={"status": "filled", "order_id": "x", "legs": []}) as mock_place:
        handled = th._execute_equity_trade(signal)

    assert handled is True
    mock_place.assert_called_once()  # exit went through despite the kill switch


def test_options_buy_is_blocked_when_kill_switch_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "testing a halt")

    expiry = (date.today() + timedelta(days=5)).isoformat()
    signal = {"ticker": "AAPL", "side": "CALL", "strike": 200.0, "expiry": expiry, "confidence": 0.9}
    with patch.object(th, "place_option_order") as mock_place:
        handled = th._execute_option_trade(signal)

    assert handled is True
    mock_place.assert_not_called()


def test_trades_proceed_normally_once_kill_switch_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.set_kill_switch(True, "testing")
    store.set_kill_switch(False, "")  # turned back off

    signal = {"source": "patreon", "action": "BUY", "symbol": "AAPL", "confidence": 0.9}
    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "check_correlation_ok", return_value=(True, "ok")), \
         patch.object(th, "check_pdt_ok", return_value=(True, "ok")), \
         patch.object(th, "place_trade", return_value={"status": "filled", "order_id": "x", "legs": []}) as mock_place:
        handled = th._execute_equity_trade(signal)

    assert handled is True
    mock_place.assert_called_once()
