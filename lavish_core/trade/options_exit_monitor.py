# lavish_core/trade/options_exit_monitor.py
# Alpaca doesn't support bracket/OCO orders on options positions, so a
# target/stop on an options trade only means something if this actively
# watches for it and submits a closing order when a level is hit.
#
# Important: alert callers state target/stop as a level on the UNDERLYING
# ("Target $526. Stop loss $528.20" on SPY), not the option's own premium -
# that's how a discretionary trader thinks about levels on a chart. So this
# watches the underlying's price, not the contract's, and closes the
# contract when the underlying crosses the caller's stated level.
from __future__ import annotations
import os, time, logging, threading
from typing import Optional, Dict, Any, Callable

from lavish_core.trade.broker_alpaca import latest_quote
from lavish_core.trade.options_broker import latest_option_quote, place_option_order

log = logging.getLogger("options_exit_monitor")

POLL_INTERVAL_SEC = float(os.getenv("OPTIONS_EXIT_POLL_INTERVAL_SEC", "5"))
MAX_MONITOR_HOURS = float(os.getenv("OPTIONS_EXIT_MAX_MONITOR_HOURS", "8"))


def _option_limit_price(contract_symbol: str) -> Optional[float]:
    q = latest_option_quote(contract_symbol)
    if not q or not q.get("bid") or not q.get("ask"):
        return None
    return round((q["bid"] + q["ask"]) / 2, 2)


def watch_and_exit(
    underlying_symbol: str,
    contract_symbol: str,
    qty: int,
    option_side: str,  # "call" | "put" - determines which direction is target vs stop
    target_underlying: Optional[float] = None,
    stop_underlying: Optional[float] = None,
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Blocking poll loop: watches the underlying's price and submits a
    closing 'sell' order on the option contract the moment the caller's
    stated target or stop level is crossed. Intended to run one thread per
    open options position (see watch_and_exit_async), since Alpaca gives
    no server-side bracket for options.

    Direction depends on option_side: a CALL profits as the underlying
    rises (target = price >= target_underlying, stop = price <=
    stop_underlying); a PUT is the reverse.

    Gives up after MAX_MONITOR_HOURS so a position that never hits either
    level doesn't leave a thread running forever - the caller is
    responsible for deciding what to do with a position that timed out
    unmanaged (e.g. surface it for manual review, don't just abandon it).
    """
    if target_underlying is None and stop_underlying is None:
        return {"status": "skipped", "reason": "no target or stop given"}

    is_call = option_side.lower().startswith("c")
    deadline = time.time() + MAX_MONITOR_HOURS * 3600
    log.info(
        "Watching %s (%s %s x%s) via underlying %s: target=%s stop=%s",
        contract_symbol, option_side.upper(), qty, qty, underlying_symbol,
        target_underlying, stop_underlying,
    )

    while time.time() < deadline:
        underlying_price = latest_quote(underlying_symbol)
        if underlying_price is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        hit = None
        if is_call:
            if target_underlying is not None and underlying_price >= target_underlying:
                hit = "target"
            elif stop_underlying is not None and underlying_price <= stop_underlying:
                hit = "stop"
        else:
            if target_underlying is not None and underlying_price <= target_underlying:
                hit = "target"
            elif stop_underlying is not None and underlying_price >= stop_underlying:
                hit = "stop"

        if hit:
            exit_price = _option_limit_price(contract_symbol)
            try:
                if exit_price is not None:
                    order = place_option_order(
                        contract_symbol=contract_symbol, side="sell", qty=qty,
                        order_type="limit", limit_price=exit_price,
                    )
                else:
                    raise RuntimeError("no option quote available to price the exit")
            except Exception as e:
                log.error("Exit order failed for %s at %s hit (underlying=%.2f): %s",
                          contract_symbol, hit, underlying_price, e)
                result = {"status": "exit_order_failed", "reason": hit,
                          "underlying_price": underlying_price, "error": str(e)}
            else:
                log.info("Exit (%s) submitted for %s, underlying=%.2f, option_limit=%.2f: %s",
                          hit, contract_symbol, underlying_price, exit_price, order)
                result = {"status": "exited", "reason": hit,
                          "underlying_price": underlying_price, "option_price": exit_price, "order": order}
            if on_exit:
                on_exit(result)
            return result

        time.sleep(POLL_INTERVAL_SEC)

    log.warning("Monitor for %s timed out after %.1fh with no exit - position needs manual attention.",
                contract_symbol, MAX_MONITOR_HOURS)
    result = {"status": "timed_out", "reason": "max_monitor_hours_exceeded"}
    if on_exit:
        on_exit(result)
    return result


def watch_and_exit_async(*args, **kwargs) -> threading.Thread:
    """Fire-and-forget version of watch_and_exit() for callers that can't block."""
    t = threading.Thread(target=watch_and_exit, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
