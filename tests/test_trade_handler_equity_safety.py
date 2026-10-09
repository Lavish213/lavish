# Covers two real money-path bugs in _execute_equity_trade, confirmed
# directly in the code before fixing:
#   - a missing/failed real quote fell back to _dry_price() (a hash of the
#     ticker string) and sized a REAL order off it in paper/live mode
#   - a SELL's qty was computed purely from the configured dollar amount,
#     never checked against what's actually held, so it could short by
#     accident
from __future__ import annotations

from unittest.mock import patch

import lavish_core.trading.trade_handler as th


def _base_signal(action="BUY", symbol="AAPL", confidence=0.9):
    return {"source": "patreon", "action": action, "symbol": symbol, "confidence": confidence}


def test_paper_mode_refuses_trade_when_no_real_quote(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "latest_quote", return_value=None), \
         patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(_base_signal())

    assert handled is False  # retryable, not a terminal business-rule skip
    mock_place.assert_not_called()


def test_paper_mode_refuses_trade_when_quote_lookup_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "latest_quote", side_effect=RuntimeError("Alpaca down")), \
         patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(_base_signal())

    assert handled is False
    mock_place.assert_not_called()


def test_dry_mode_still_uses_dry_price_fallback(tmp_path, monkeypatch):
    # Dry mode never submits a real order - the fallback is safe there,
    # and this confirms the fix didn't break that legitimate path.
    monkeypatch.setattr(th, "TRADE_MODE", "dry")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "latest_quote", return_value=None), \
         patch.object(th, "place_trade", return_value={"status": "filled", "order_id": "x", "legs": []}) as mock_place:
        handled = th._execute_equity_trade(_base_signal())

    assert handled is True
    mock_place.assert_called_once()


def test_sell_with_nothing_held_is_rejected_not_shorted(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", return_value=None), \
         patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(_base_signal(action="SELL"))

    assert handled is True  # deterministic business-rule skip, not retryable
    mock_place.assert_not_called()


def test_sell_qty_is_capped_to_what_is_actually_held(tmp_path, monkeypatch):
    # amount_usd sizes this to 10 shares at $150 ref price (1500/150), but
    # only 3 are actually held - must cap to 3, never submit a sell for
    # more than is owned (which is how an accidental short happened).
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    signal = _base_signal(action="SELL")
    signal["amount_usd"] = 1500.0
    captured = {}

    def _fake_place_trade(**kwargs):
        captured["qty"] = kwargs["qty"]
        return {"status": "filled", "order_id": "x", "legs": []}

    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", return_value={"qty": "3"}), \
         patch.object(th, "place_trade", side_effect=_fake_place_trade):
        handled = th._execute_equity_trade(signal)

    assert handled is True
    assert captured["qty"] == 3.0


def test_sell_within_held_quantity_is_not_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    signal = _base_signal(action="SELL")
    signal["amount_usd"] = 150.0  # 1 share at ref 150
    captured = {}

    def _fake_place_trade(**kwargs):
        captured["qty"] = kwargs["qty"]
        return {"status": "filled", "order_id": "x", "legs": []}

    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", return_value={"qty": "10"}), \
         patch.object(th, "place_trade", side_effect=_fake_place_trade):
        handled = th._execute_equity_trade(signal)

    assert handled is True
    assert captured["qty"] == 1.0


def test_position_check_failure_is_retryable_not_a_silent_short(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", side_effect=RuntimeError("Alpaca down")), \
         patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(_base_signal(action="SELL"))

    assert handled is False
    mock_place.assert_not_called()
