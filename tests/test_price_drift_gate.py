# Covers the price-drift-since-alert gate: a reference price is captured
# as early as possible in execute_trade_from_post() (before dedup/parsing/
# risk checks run), and compared against a fresh quote at the actual
# moment of execution - a late-processed alert shouldn't trade at a price
# meaningfully different from what she actually saw when she posted it.
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import lavish_core.trading.trade_handler as th


def test_price_drift_reason_none_when_within_tolerance():
    assert th._price_drift_reason(100.0, 101.0) is None  # 1% move, default limit 3%


def test_price_drift_reason_fires_when_exceeded():
    reason = th._price_drift_reason(100.0, 105.0)  # 5% move
    assert reason is not None
    assert "price_drift" in reason


def test_price_drift_reason_none_when_either_price_missing():
    assert th._price_drift_reason(None, 105.0) is None
    assert th._price_drift_reason(100.0, None) is None


def test_price_drift_reason_disabled_when_threshold_is_zero(monkeypatch):
    monkeypatch.setattr(th, "MAX_PRICE_DRIFT_SINCE_ALERT_PCT", 0)
    assert th._price_drift_reason(100.0, 200.0) is None


def test_equity_buy_rejected_when_drifted_too_far(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    signal = {"source": "patreon", "action": "BUY", "symbol": "AAPL", "confidence": 0.9,
              "_alert_time_price": 100.0}
    with patch.object(th, "latest_quote", return_value=110.0), \
         patch.object(th, "place_trade") as mock_place:
        handled = th._execute_equity_trade(signal)
    assert handled is True  # deterministic skip
    mock_place.assert_not_called()


def test_equity_buy_proceeds_when_within_drift_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    signal = {"source": "patreon", "action": "BUY", "symbol": "AAPL", "confidence": 0.9,
              "_alert_time_price": 100.0}
    with patch.object(th, "latest_quote", return_value=100.5), \
         patch.object(th, "check_correlation_ok", return_value=(True, "ok")), \
         patch.object(th, "check_pdt_ok", return_value=(True, "ok")), \
         patch.object(th, "place_trade", return_value={"status": "filled", "order_id": "x", "legs": []}) as mock_place:
        handled = th._execute_equity_trade(signal)
    assert handled is True
    mock_place.assert_called_once()


def test_equity_sell_is_never_blocked_by_drift(tmp_path, monkeypatch):
    # Exits must never be gated by price drift, same philosophy as every
    # other entry-only risk check.
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    signal = {"source": "patreon", "action": "SELL", "symbol": "AAPL", "confidence": 0.9,
              "_alert_time_price": 100.0}
    with patch.object(th, "latest_quote", return_value=150.0), \
         patch.object(th, "broker_get_position", return_value={"qty": "10"}), \
         patch.object(th, "place_trade", return_value={"status": "filled", "order_id": "x", "legs": []}) as mock_place:
        handled = th._execute_equity_trade(signal)
    assert handled is True
    mock_place.assert_called_once()


def _options_signal(alert_time_price=200.0):
    expiry = (date.today() + timedelta(days=5)).isoformat()
    return {"ticker": "AAPL", "side": "CALL", "strike": 200.0, "expiry": expiry,
            "confidence": 0.9, "_alert_time_price": alert_time_price}


def test_options_buy_rejected_when_underlying_drifted_too_far(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "resolve_and_price_contract",
                       return_value={"symbol": "AAPL260101C00200000", "mid_price": 5.0}), \
         patch.object(th, "latest_quote", return_value=220.0), \
         patch.object(th, "place_option_order") as mock_place:
        handled = th._execute_option_trade(_options_signal(alert_time_price=200.0))
    assert handled is True
    mock_place.assert_not_called()
