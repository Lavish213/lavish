# Covers the dedup-timing bug: a signal used to be recorded as "seen"
# BEFORE validation, risk approval, or broker submission - so a transient
# failure (quote/broker error) could suppress a legitimate retry from the
# other alert source (Patreon/Discord) for the whole dedup window. Fixed
# by recording only once the execution path reports a final, deterministic
# outcome (_execute_option_trade/_execute_equity_trade returning True),
# not on a retryable technical failure (returning False).
from __future__ import annotations

from unittest.mock import patch

import lavish_core.trading.trade_handler as th


def _equity_signal(action="BUY", symbol="AAPL"):
    return {"source": "patreon", "action": action, "symbol": symbol, "confidence": 0.9}


def test_signal_is_recorded_when_execution_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "_execute_equity_trade", return_value=True) as mock_exec:
        th.execute_trade_from_post(_equity_signal())
    mock_exec.assert_called_once()

    store = th.HybridStore(duckdb_path=tmp_path / "test.duckdb")
    rows = store.fetchall("SELECT symbol, side FROM signals WHERE symbol = 'AAPL'")
    assert len(rows) == 1


def test_signal_is_not_recorded_on_a_retryable_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "_execute_equity_trade", return_value=False) as mock_exec:
        th.execute_trade_from_post(_equity_signal())
    mock_exec.assert_called_once()

    store = th.HybridStore(duckdb_path=tmp_path / "test.duckdb")
    rows = store.fetchall("SELECT symbol, side FROM signals WHERE symbol = 'AAPL'")
    assert len(rows) == 0  # nothing recorded - the other source can still retry this


def test_signal_is_not_recorded_on_an_unhandled_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "_execute_equity_trade", side_effect=RuntimeError("boom")):
        th.execute_trade_from_post(_equity_signal())  # must not propagate

    store = th.HybridStore(duckdb_path=tmp_path / "test.duckdb")
    rows = store.fetchall("SELECT symbol, side FROM signals WHERE symbol = 'AAPL'")
    assert len(rows) == 0


def test_second_source_is_still_deduped_against_a_successful_first_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "_execute_equity_trade", return_value=True) as mock_exec:
        th.execute_trade_from_post(_equity_signal())          # source A, succeeds
        th.execute_trade_from_post(_equity_signal())           # source B, same alert
    assert mock_exec.call_count == 1  # second call was deduped before dispatch


def test_second_source_gets_a_real_chance_after_a_retryable_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_DB", tmp_path / "test.duckdb")
    with patch.object(th, "_execute_equity_trade", return_value=False) as mock_exec:
        th.execute_trade_from_post(_equity_signal())           # source A, fails technically
        th.execute_trade_from_post(_equity_signal())           # source B, same alert
    assert mock_exec.call_count == 2  # NOT deduped - source A never actually recorded anything
