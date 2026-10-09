# Covers a real gap found during the "ready for live money" pass: all
# three new-entry risk gates (circuit breaker, correlation check, and PDT
# guard - see test_pdt_guard.py for that one) failed OPEN on a broker/
# account read error, silently allowing a trade through exactly when the
# bot could least verify it was safe. Default flipped to fail closed;
# RISK_CHECK_FAIL_OPEN=true restores the old behavior explicitly.
from __future__ import annotations

from unittest.mock import patch

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.trade import circuit_breaker, portfolio_risk


def test_circuit_breaker_fails_closed_on_account_read_error(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    with patch.object(circuit_breaker, "get_account", side_effect=RuntimeError("broker down")):
        ok, reason = circuit_breaker.check_ok(store)
    assert ok is False
    assert "unavailable" in reason


def test_circuit_breaker_fails_open_with_explicit_opt_out(tmp_path, monkeypatch):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    monkeypatch.setattr(circuit_breaker, "RISK_CHECK_FAIL_OPEN", True)
    with patch.object(circuit_breaker, "get_account", side_effect=RuntimeError("broker down")):
        ok, reason = circuit_breaker.check_ok(store)
    assert ok is True


def test_circuit_breaker_still_works_normally_when_account_is_readable(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    with patch.object(circuit_breaker, "get_account", return_value={"equity": "100000"}):
        ok, reason = circuit_breaker.check_ok(store)
    assert ok is True
    assert reason == "ok"


def test_correlation_check_fails_closed_on_broker_read_error():
    def _raise():
        raise RuntimeError("broker down")
    ok, reason = portfolio_risk.check_correlation_ok("AAPL", get_positions_fn=_raise)
    assert ok is False
    assert "unavailable" in reason


def test_correlation_check_fails_open_with_explicit_opt_out(monkeypatch):
    monkeypatch.setattr(portfolio_risk, "RISK_CHECK_FAIL_OPEN", True)

    def _raise():
        raise RuntimeError("broker down")
    ok, reason = portfolio_risk.check_correlation_ok("AAPL", get_positions_fn=_raise)
    assert ok is True


def test_correlation_check_still_works_normally_when_positions_are_readable():
    def _positions():
        return [{"symbol": "MSFT", "qty": "10"}, {"symbol": "GOOGL", "qty": "5"}]
    ok, reason = portfolio_risk.check_correlation_ok("AAPL", get_positions_fn=_positions)
    assert ok is True  # only 2 correlated (MSFT, GOOGL), limit is 3
