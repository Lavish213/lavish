# Covers the third exit-safety fix: her original stated target/stop is
# persisted in the entry order's meta (trade_handler.py) and recovered on
# reconcile (reconcile.py), instead of a crash/restart always falling back
# to our generic default guardrails. Tests the real HybridStore read/write
# round trip, not a mock - this is exactly what trade_handler.py writes
# and what reconcile.py reads back.
from __future__ import annotations

import pytest

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.trade.reconcile import _find_original_target_stop


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    monkeypatch.setattr("lavish_core.trade.reconcile.DEFAULT_DB", tmp_path / "test.duckdb")
    return s


def test_recovers_persisted_target_and_stop(store):
    store.submit_order(
        symbol="TSLA260101C00100000", side="buy", qty=1, order_type="limit",
        limit_price=4.0, venue="paper", status="filled",
        meta={"ticker": "TSLA", "target_underlying": 250.0, "stop_underlying": 220.0},
    )
    target, stop = _find_original_target_stop("TSLA260101C00100000")
    assert target == 250.0
    assert stop == 220.0


def test_returns_none_none_when_no_order_found(store):
    target, stop = _find_original_target_stop("NOPE260101C00100000")
    assert target is None
    assert stop is None


def test_returns_none_none_when_levels_were_never_given(store):
    # A real alert with no stated target/stop - trade_handler.py still
    # writes the meta keys, just as None (json round-trips None -> null).
    store.submit_order(
        symbol="SPY260101P00500000", side="buy", qty=1, order_type="limit",
        limit_price=2.0, venue="paper", status="filled",
        meta={"ticker": "SPY", "target_underlying": None, "stop_underlying": None},
    )
    target, stop = _find_original_target_stop("SPY260101P00500000")
    assert target is None
    assert stop is None


def test_uses_the_most_recent_buy_order_for_the_symbol(store):
    store.submit_order(
        symbol="AMD260101C00200000", side="buy", qty=1, order_type="limit",
        limit_price=3.0, venue="paper", status="rejected",
        meta={"target_underlying": 999.0, "stop_underlying": 888.0},
    )
    import time
    time.sleep(0.01)
    store.submit_order(
        symbol="AMD260101C00200000", side="buy", qty=1, order_type="limit",
        limit_price=3.5, venue="paper", status="filled",
        meta={"target_underlying": 240.0, "stop_underlying": 210.0},
    )
    target, stop = _find_original_target_stop("AMD260101C00200000")
    assert target == 240.0
    assert stop == 210.0
