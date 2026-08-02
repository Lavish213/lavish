# lavish_core/trade/circuit_breaker.py
# P&L-based circuit breaker: halts new entries when the account is down too
# much for the day/week, or after too many losses in a row. Everything else
# in trade_agent.py/trade_handler.py gates on position size and exposure -
# nothing gated on "are we actually losing money right now" until this.
#
# Common industry defaults (daily 2-5%, weekly ~7%, pause after ~5 losses in
# a row) - all overridable via env, all conservative-by-default so a missing
# env var never silently disables protection.
from __future__ import annotations
import os, time, logging
from typing import Tuple, Optional

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.trade.broker_alpaca import get_account
from lavish_core.utils.alerts import post_discord

log = logging.getLogger("circuit_breaker")

DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03"))
WEEKLY_LOSS_LIMIT_PCT = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.07"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

_DAY_SECONDS = 24 * 3600
_WEEK_SECONDS = 7 * _DAY_SECONDS


def _get_baseline(store: HybridStore, key: str, window_seconds: float, current_equity: float) -> float:
    """
    Returns the equity to measure drawdown against, rolling the baseline
    forward (to current equity) whenever the window has elapsed - i.e. a
    fresh day/week starts a fresh baseline, same as any daily-loss-limit
    trading rule.
    """
    limits = store.get_risk_limits()
    ts_key, eq_key = f"{key}_baseline_ts", f"{key}_baseline_equity"
    baseline_ts = limits.get(ts_key)
    baseline_equity = limits.get(eq_key)

    now = time.time()
    if baseline_ts is None or baseline_equity is None or (now - baseline_ts) > window_seconds:
        store.set_risk_limits({ts_key: now, eq_key: current_equity})
        return current_equity
    return baseline_equity


def check_ok(store: HybridStore) -> Tuple[bool, str]:
    """
    Call before submitting ANY new entry order (equity or options). Returns
    (True, "ok") to proceed, or (False, reason) to block the trade.
    Fails open (allows trading) only if the account equity read itself
    fails - a broker outage shouldn't be indistinguishable from "we're not
    trading right now" in the logs, so this logs loudly either way.
    """
    try:
        acct = get_account()
        equity = float(acct.get("equity") or 0.0)
    except Exception as e:
        log.warning("circuit_breaker: could not read account equity (%s) - allowing trade, but this needs attention.", e)
        return True, "ok (equity check unavailable)"

    if equity <= 0:
        return True, "ok (no equity data)"

    day_baseline = _get_baseline(store, "circuit_breaker_day", _DAY_SECONDS, equity)
    week_baseline = _get_baseline(store, "circuit_breaker_week", _WEEK_SECONDS, equity)

    day_dd = (day_baseline - equity) / day_baseline if day_baseline > 0 else 0.0
    week_dd = (week_baseline - equity) / week_baseline if week_baseline > 0 else 0.0

    if day_dd >= DAILY_LOSS_LIMIT_PCT:
        reason = f"daily loss limit hit: down {day_dd:.1%} today (limit {DAILY_LOSS_LIMIT_PCT:.1%})"
        log.error("circuit_breaker: BLOCKING new trades - %s", reason)
        _alert_once(store, kind=1, message=f"🛑 Circuit breaker tripped: {reason}. New trades blocked.")
        return False, reason

    if week_dd >= WEEKLY_LOSS_LIMIT_PCT:
        reason = f"weekly loss limit hit: down {week_dd:.1%} this week (limit {WEEKLY_LOSS_LIMIT_PCT:.1%})"
        log.error("circuit_breaker: BLOCKING new trades - %s", reason)
        _alert_once(store, kind=2, message=f"🛑 Circuit breaker tripped: {reason}. New trades blocked.")
        return False, reason

    consecutive = int(store.get_risk_limits().get("circuit_breaker_consecutive_losses", 0))
    if consecutive >= MAX_CONSECUTIVE_LOSSES:
        reason = f"{consecutive} consecutive losses (limit {MAX_CONSECUTIVE_LOSSES}) - paused pending manual review"
        log.error("circuit_breaker: BLOCKING new trades - %s", reason)
        _alert_once(store, kind=3, message=f"🛑 Circuit breaker tripped: {reason}. New trades blocked until reset_consecutive_losses() is called.")
        return False, reason

    _alert_once(store, kind=0, message="")  # clears the "already alerted" state once trading resumes
    return True, "ok"


def _alert_once(store: HybridStore, kind: int, message: str) -> None:
    """
    Fires a Discord alert only on the transition into a given blocked
    state (or back to healthy), not on every single check_ok() call while
    still blocked - otherwise every incoming alert during a halt would
    spam a duplicate notification.
    """
    last_kind = int(store.get_risk_limits().get("circuit_breaker_last_alert_kind", 0))
    if last_kind == kind:
        return
    store.set_risk_limits({"circuit_breaker_last_alert_kind": kind})
    if kind != 0 and message:
        post_discord(message)


def record_trade_outcome(store: HybridStore, pnl: float) -> None:
    """
    Call after a position is fully closed with a known realized P&L, to
    drive the consecutive-loss counter. A win resets it; a loss (or exact
    breakeven, treated as non-win) increments it.
    """
    limits = store.get_risk_limits()
    consecutive = int(limits.get("circuit_breaker_consecutive_losses", 0))
    consecutive = 0 if pnl > 0 else consecutive + 1
    store.set_risk_limits({"circuit_breaker_consecutive_losses": consecutive})
    if consecutive >= MAX_CONSECUTIVE_LOSSES:
        log.error("circuit_breaker: %s consecutive losses reached - new trades are now blocked until reset_consecutive_losses() is called.", consecutive)


def reset_consecutive_losses(store: HybridStore) -> None:
    """Manual override to resume trading after a human has reviewed a losing streak."""
    store.set_risk_limits({"circuit_breaker_consecutive_losses": 0})
    log.info("circuit_breaker: consecutive-loss counter manually reset.")
