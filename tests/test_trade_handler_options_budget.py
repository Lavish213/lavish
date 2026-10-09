# Covers the options budget-cap bug: qty = max(1, dollars // (mid*100))
# always bought at least 1 contract even when it cost far more than the
# configured budget. Also covers _execute_option_trade's return value
# (True/False) used for dedup timing (see test_dedup_timing.py).
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import lavish_core.trading.trade_handler as th


def _contract(symbol="AAPL260101C00200000", mid_price=50.0, bid=None, ask=None):
    c = {"symbol": symbol, "mid_price": mid_price}
    if bid is not None:
        c["bid"] = bid
    if ask is not None:
        c["ask"] = ask
    return c


def _options_signal(ticker="AAPL", side="CALL", strike=200.0, confidence=0.9, amount_usd=None):
    expiry = (date.today() + timedelta(days=5)).isoformat()
    sig = {"ticker": ticker, "side": side, "strike": strike, "expiry": expiry, "confidence": confidence}
    if amount_usd is not None:
        sig["amount_usd"] = amount_usd
    return sig


def test_contract_over_budget_is_rejected_not_bought_anyway(tmp_path, monkeypatch):
    # mid_price=50 -> $5000/contract; budget $200 (default) can't afford 1.
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "resolve_and_price_contract", return_value=_contract(mid_price=50.0)), \
         patch.object(th, "place_option_order") as mock_place:
        handled = th._execute_option_trade(_options_signal(amount_usd=200.0))

    assert handled is True  # deterministic business-rule skip, not retryable
    mock_place.assert_not_called()


def test_affordable_contract_still_trades(tmp_path, monkeypatch):
    # mid_price=1.50 -> $150/contract; budget $500 affords 3.
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "resolve_and_price_contract", return_value=_contract(mid_price=1.50)), \
         patch.object(th, "place_option_order",
                       return_value={"id": "o1", "status": "filled", "filled_qty": "3", "filled_avg_price": "1.50"}), \
         patch.object(th, "circuit_breaker_check_ok", return_value=(True, "ok")), \
         patch.object(th, "check_correlation_ok", return_value=(True, "ok")), \
         patch.object(th, "check_pdt_ok", return_value=(True, "ok")), \
         patch.object(th, "watch_and_exit_async"):
        handled = th._execute_option_trade(_options_signal(amount_usd=500.0))

    assert handled is True


def test_no_contract_found_is_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "resolve_and_price_contract", return_value=None):
        handled = th._execute_option_trade(_options_signal())
    assert handled is False


def test_order_submission_failure_is_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "resolve_and_price_contract", return_value=_contract(mid_price=1.50)), \
         patch.object(th, "circuit_breaker_check_ok", return_value=(True, "ok")), \
         patch.object(th, "check_correlation_ok", return_value=(True, "ok")), \
         patch.object(th, "check_pdt_ok", return_value=(True, "ok")), \
         patch.object(th, "place_option_order", side_effect=RuntimeError("Alpaca 503")):
        handled = th._execute_option_trade(_options_signal(amount_usd=500.0))
    assert handled is False


def test_confidence_floor_skip_is_terminal_not_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TRADE_MODE", "paper")
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    handled = th._execute_option_trade(_options_signal(confidence=0.1))
    assert handled is True
