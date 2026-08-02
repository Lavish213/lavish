# lavish_core/trade/options_exit_monitor.py
# Alpaca doesn't support bracket/OCO orders on options positions, so any
# exit plan - hers or ours - only means something if this actively watches
# for it and submits a closing order when a level is hit.
#
# She doesn't always post a sell/exit alert. Her stated target/stop (when
# given) are followed first, since that's the actual call being copied -
# but this always ALSO runs its own independent guardrails underneath
# hers, so a position is never left completely unmanaged just because she
# went quiet:
#   - a hard stop-loss on the option's own premium (she trades short-dated
#     options; a gap can blow through an underlying-price stop before this
#     loop's next poll, so a premium-based backstop matters)
#   - a trailing stop once meaningfully in profit, so a round-trip back to
#     a loss can't happen silently just because no one said "sell"
#   - a forced close near market close on the contract's expiration day,
#     regardless of P&L - theta/pin risk goes non-linear into the close on
#     0-5 DTE contracts, which is what she trades
#   - a max-hold-time cutoff as a last resort if none of the above fire
#
# Standard retail options risk management (this is not a novel invention -
# see e.g. E*TRADE's and Schwab's own writeups on automating options exits):
# stop around 40-50% of premium, take-profit/trail around 50-100%+, and
# always have a time-based exit near expiration.
from __future__ import annotations
import os, time, logging, threading
from datetime import date, datetime, timezone
from typing import Optional, Dict, Any, Callable

from lavish_core.trade.broker_alpaca import latest_quote, get_clock
from lavish_core.trade.options_broker import latest_option_quote, place_option_order
from lavish_core.utils.alerts import post_discord

log = logging.getLogger("options_exit_monitor")

POLL_INTERVAL_SEC = float(os.getenv("OPTIONS_EXIT_POLL_INTERVAL_SEC", "5"))
MAX_MONITOR_HOURS = float(os.getenv("OPTIONS_EXIT_MAX_MONITOR_HOURS", "8"))

# Our own guardrails - always active, independent of anything she posts.
OUR_STOP_LOSS_PCT = float(os.getenv("OPTIONS_OWN_STOP_LOSS_PCT", "0.50"))       # exit if premium down 50%
OUR_TRAIL_TRIGGER_PCT = float(os.getenv("OPTIONS_OWN_TRAIL_TRIGGER_PCT", "0.30"))  # start trailing once up 30%
OUR_TRAIL_DRAWDOWN_PCT = float(os.getenv("OPTIONS_OWN_TRAIL_DRAWDOWN_PCT", "0.20"))  # give back at most 20% off the high
EXPIRY_FORCE_CLOSE_MINUTES = float(os.getenv("OPTIONS_EXPIRY_FORCE_CLOSE_MINUTES", "30"))  # force out N min before close, on expiry day


def _option_limit_price(contract_symbol: str) -> Optional[float]:
    q = latest_option_quote(contract_symbol)
    if not q or not q.get("bid") or not q.get("ask"):
        return None
    return round((q["bid"] + q["ask"]) / 2, 2)


def _minutes_to_close() -> Optional[float]:
    try:
        clk = get_clock()
        if not clk.get("is_open"):
            return None
        next_close = datetime.fromisoformat(clk["next_close"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0.0, (next_close - now).total_seconds() / 60.0)
    except Exception as e:
        log.warning("Could not read market clock: %s", e)
        return None


def watch_and_exit(
    underlying_symbol: str,
    contract_symbol: str,
    qty: int,
    option_side: str,  # "call" | "put" - determines which direction is target vs stop
    entry_price: float,
    expiry: Optional[date] = None,
    target_underlying: Optional[float] = None,
    stop_underlying: Optional[float] = None,
    on_exit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Blocking poll loop: watches both her stated underlying levels (if any)
    AND our own premium-based guardrails, and submits a closing 'sell'
    order on the option contract the moment anything triggers. Intended to
    run one thread per open options position (see watch_and_exit_async).

    Direction for her levels depends on option_side: a CALL profits as the
    underlying rises (target = price >= target_underlying, stop = price <=
    stop_underlying); a PUT is the reverse.
    """
    is_call = option_side.lower().startswith("c")
    deadline = time.time() + MAX_MONITOR_HOURS * 3600
    high_water_pct = 0.0
    trailing_active = False

    log.info(
        "Watching %s (%s x%s, entry=%.2f) via underlying %s: her_target=%s her_stop=%s "
        "| our guardrails: stop=-%.0f%% trail_trigger=+%.0f%% trail_drawdown=%.0f%%",
        contract_symbol, option_side.upper(), qty, entry_price, underlying_symbol,
        target_underlying, stop_underlying,
        OUR_STOP_LOSS_PCT * 100, OUR_TRAIL_TRIGGER_PCT * 100, OUR_TRAIL_DRAWDOWN_PCT * 100,
    )

    def _exit(reason: str, underlying_price: Optional[float] = None) -> Dict[str, Any]:
        exit_price = _option_limit_price(contract_symbol)
        try:
            if exit_price is None:
                raise RuntimeError("no option quote available to price the exit")
            order = place_option_order(
                contract_symbol=contract_symbol, side="sell", qty=qty,
                order_type="limit", limit_price=exit_price,
            )
        except Exception as e:
            log.error("Exit order failed for %s at %s: %s", contract_symbol, reason, e)
            result = {"status": "exit_order_failed", "reason": reason,
                      "underlying_price": underlying_price, "error": str(e)}
            post_discord(f"🚨 Failed to submit exit order for {contract_symbol} (trigger: {reason}): {e}. "
                         f"This position may be unmanaged until manually reviewed.")
        else:
            log.info("Exit (%s) submitted for %s, option_limit=%.2f: %s", reason, contract_symbol, exit_price, order)
            result = {"status": "exited", "reason": reason,
                      "underlying_price": underlying_price, "option_price": exit_price, "order": order}
            if reason in ("max_hold_timeout", "expiry_force_close"):
                post_discord(f"⏱️ {contract_symbol} force-closed ({reason}) at {exit_price:.2f} - "
                             f"neither her levels nor our stop/trail fired before this cutoff.")
        if on_exit:
            on_exit(result)
        return result

    while time.time() < deadline:
        # --- her stated levels (checked first - this is the call being copied) ---
        underlying_price = latest_quote(underlying_symbol)
        if underlying_price is not None:
            if is_call:
                if target_underlying is not None and underlying_price >= target_underlying:
                    return _exit("her_target", underlying_price)
                if stop_underlying is not None and underlying_price <= stop_underlying:
                    return _exit("her_stop", underlying_price)
            else:
                if target_underlying is not None and underlying_price <= target_underlying:
                    return _exit("her_target", underlying_price)
                if stop_underlying is not None and underlying_price >= stop_underlying:
                    return _exit("her_stop", underlying_price)

        # --- our own guardrails (always run, regardless of whether she gave levels) ---
        premium = _option_limit_price(contract_symbol)
        if premium is not None and entry_price > 0:
            pnl_pct = (premium - entry_price) / entry_price
            high_water_pct = max(high_water_pct, pnl_pct)
            if not trailing_active and high_water_pct >= OUR_TRAIL_TRIGGER_PCT:
                trailing_active = True
                log.info("%s: trailing stop armed at high-water +%.1f%%", contract_symbol, high_water_pct * 100)

            if pnl_pct <= -OUR_STOP_LOSS_PCT:
                return _exit("our_stop_loss", underlying_price)
            if trailing_active and (high_water_pct - pnl_pct) >= OUR_TRAIL_DRAWDOWN_PCT:
                return _exit("our_trailing_stop", underlying_price)

        # --- forced close near expiration, regardless of P&L ---
        if expiry is not None and datetime.now(timezone.utc).date() >= expiry:
            mins_left = _minutes_to_close()
            if mins_left is not None and mins_left <= EXPIRY_FORCE_CLOSE_MINUTES:
                return _exit("expiry_force_close", underlying_price)

        time.sleep(POLL_INTERVAL_SEC)

    # Max hold time reached with nothing else triggering - force out rather
    # than abandon the position unmanaged.
    log.warning("%s: max hold time (%.1fh) reached with no other exit - forcing close.",
                contract_symbol, MAX_MONITOR_HOURS)
    return _exit("max_hold_timeout")


def watch_and_exit_async(*args, **kwargs) -> threading.Thread:
    """Fire-and-forget version of watch_and_exit() for callers that can't block."""
    t = threading.Thread(target=watch_and_exit, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
