# lavish_core/trade/account_monitor.py
# HybridStore already had an account_snapshots table and a snapshot()
# method (see db/hybrid_store.py) - but nothing outside its own CLI demo
# ever called it. That means there was no historical equity/cash/day-P&L
# series stored anywhere: the only way to see how the account was doing
# over time was Alpaca's own dashboard, not this bot's own audit trail.
# This periodically snapshots the real account so track_record.py and any
# future dashboard has actual equity-over-time data to plot, independent
# of individual trade fills.
from __future__ import annotations
import os, time, logging
from typing import Optional

from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB
from lavish_core.trade.broker_alpaca import get_account

log = logging.getLogger("account_monitor")

ACCOUNT_SNAPSHOT_INTERVAL_SECONDS = float(os.getenv("ACCOUNT_SNAPSHOT_INTERVAL_SECONDS", "900"))


def _first_recorded_equity(store: HybridStore) -> Optional[float]:
    rows = store.fetchall("SELECT equity FROM account_snapshots ORDER BY ts ASC LIMIT 1")
    return float(rows[0][0]) if rows else None


def snapshot_once(store: Optional[HybridStore] = None) -> Optional[dict]:
    store = store or HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
    try:
        acct = get_account()
    except Exception as e:
        log.warning("account_monitor: could not read account for snapshot: %s", e)
        return None

    equity = float(acct.get("equity") or 0.0)
    cash = float(acct.get("cash") or 0.0)
    buying_power = float(acct.get("buying_power") or 0.0)
    last_equity = float(acct.get("last_equity") or equity)
    day_pl = equity - last_equity

    baseline = _first_recorded_equity(store)
    cumulative_pl = (equity - baseline) if baseline is not None else 0.0

    snap = store.snapshot(equity=equity, cash=cash, buying_power=buying_power,
                           day_pl=day_pl, cumulative_pl=cumulative_pl)
    log.info("account snapshot: equity=%.2f cash=%.2f buying_power=%.2f day_pl=%.2f cumulative_pl=%.2f",
              equity, cash, buying_power, day_pl, cumulative_pl)
    return snap


def run_periodic_snapshot() -> None:
    """Blocking loop - run in its own background thread from run_ingest.py."""
    log.info("Account snapshot loop started (every %.0fs).", ACCOUNT_SNAPSHOT_INTERVAL_SECONDS)
    store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
    while True:
        try:
            snapshot_once(store)
        except Exception as e:
            log.error("account_monitor: unexpected error: %s", e)
        time.sleep(ACCOUNT_SNAPSHOT_INTERVAL_SECONDS)
