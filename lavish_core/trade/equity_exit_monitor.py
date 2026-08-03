# lavish_core/trade/equity_exit_monitor.py
# Equity entries go out as a real Alpaca bracket order (take_profit/
# stop_loss) - unlike options, Alpaca DOES manage the OCO exit itself
# server-side (whichever leg fills first auto-cancels the other), so this
# doesn't submit anything the way options_exit_monitor.py has to. It only
# watches the bracket's child leg orders and logs whichever one fills back
# into this bot's own audit trail. Without this, an equity buy's DB order
# just sat "filled" forever with no linked exit fill, even though the
# position closed correctly on Alpaca's side - track_record.py's "open
# entries with no recorded exit fill" bucket exists specifically because
# of this gap.
from __future__ import annotations
import os, time, threading, logging
from typing import Optional, Dict, Any, Callable, List

from lavish_core.trade.broker_alpaca import get_order
from lavish_core.trade.circuit_breaker import record_trade_outcome
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB

log = logging.getLogger("equity_exit_monitor")

POLL_INTERVAL_SEC = float(os.getenv("EQUITY_EXIT_POLL_INTERVAL_SEC", "15"))
# Longer than the options default (8h) - equity brackets are typically
# wider/slower-moving than short-dated option premium, and a day order's
# own TIF already bounds this to at most one trading session anyway.
MAX_MONITOR_HOURS = float(os.getenv("EQUITY_EXIT_MAX_MONITOR_HOURS", "12"))

_TERMINAL_NO_FILL = {"canceled", "expired", "rejected"}


def watch_bracket(
    order_id: str,
    symbol: str,
    qty: float,
    entry_price: float,
    leg_ids: List[str],
    venue: str = "paper",
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Blocking poll loop over a bracket order's take_profit/stop_loss child
    legs. Returns as soon as either leg reaches "filled", or once every
    known leg has independently reached a terminal non-fill state (which
    means the position closed some other way - manual intervention, a
    day-order expiring unfilled - without an exit fill for us to record).
    """
    if not leg_ids:
        log.warning("watch_bracket called for %s with no leg ids - nothing to watch.", symbol)
        return {"status": "no_legs"}

    deadline = time.time() + MAX_MONITOR_HOURS * 3600
    log.info("Watching equity bracket exit for %s x%s (entry=%.2f) legs=%s", symbol, qty, entry_price, leg_ids)

    last_status: Dict[str, str] = {leg_id: "unknown" for leg_id in leg_ids}

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        if on_exit:
            on_exit(result)
        return result

    while time.time() < deadline:
        for leg_id in leg_ids:
            try:
                leg = get_order(leg_id)
            except Exception as e:
                log.warning("equity_exit_monitor: poll failed for leg %s (%s): %s", leg_id, symbol, e)
                continue

            status = leg.get("status", "unknown")
            last_status[leg_id] = status

            if status == "filled":
                exit_price = float(leg.get("filled_avg_price") or 0.0)
                filled_qty = float(leg.get("filled_qty") or qty)
                pnl = (exit_price - entry_price) * filled_qty
                store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
                store.log_fill(order_id=order_id, symbol=symbol, side="sell",
                                qty=filled_qty, price=exit_price, fee=0.0, venue=venue)
                record_trade_outcome(store, pnl)
                log.info("Equity bracket exit: %s closed via %s leg, entry=%.2f exit=%.2f pnl=%.2f",
                          symbol, leg.get("type", "?"), entry_price, exit_price, pnl)
                return _finish({"status": "exited", "leg_id": leg_id, "leg_type": leg.get("type"),
                                "exit_price": exit_price, "pnl": pnl})

        if all(s in _TERMINAL_NO_FILL for s in last_status.values()):
            log.warning("equity_exit_monitor: every tracked leg for %s ended without filling (%s) - "
                        "position may have closed some other way; no exit fill recorded.", symbol, last_status)
            return _finish({"status": "no_fill", "leg_statuses": dict(last_status)})

        time.sleep(POLL_INTERVAL_SEC)

    log.warning("equity_exit_monitor: %s bracket exit not observed within %.1fh - stopping watch "
                "(position may still be open and unmanaged by this monitor going forward).",
                symbol, MAX_MONITOR_HOURS)
    return _finish({"status": "monitor_timeout"})


def watch_bracket_async(*args, **kwargs) -> threading.Thread:
    """Fire-and-forget version of watch_bracket() for callers that can't block."""
    t = threading.Thread(target=watch_bracket, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
