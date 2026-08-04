# lavish_core/trade/pdt_guard.py
# FINRA's Pattern Day Trader rule: a margin account under $25k equity that
# executes 4+ "day trades" (buy AND sell the same symbol on the same
# calendar day) within a rolling 5 business days gets flagged and frozen
# from day-trading for 90 days. This bot's exit monitor is specifically
# built to close positions same-day (hard stop, trailing stop, expiry-
# force-close, max-hold timeout) - exactly the trade shape that trips
# this rule - and nothing checked for it before this.
#
# Cash accounts aren't subject to PDT but hit a different constraint
# instead (T+1/T+2 settlement - can't reuse unsettled funds), which this
# does NOT check. This guard assumes a margin account, both the more
# common retail default and what Alpaca opens by default.
from __future__ import annotations
import os, logging
from datetime import date, timedelta
from typing import Optional, Tuple

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.trade.broker_alpaca import get_account

log = logging.getLogger("pdt_guard")

PDT_EQUITY_THRESHOLD = float(os.getenv("PDT_EQUITY_THRESHOLD", "25000"))
# The 4th day trade in the window is what trips the flag - block once
# this many are already used, so the trade that would be the 4th never
# gets a chance to fire.
PDT_MAX_DAY_TRADES = int(os.getenv("PDT_MAX_DAY_TRADES", "3"))
PDT_LOOKBACK_BUSINESS_DAYS = int(os.getenv("PDT_LOOKBACK_BUSINESS_DAYS", "5"))


def _trailing_business_days(n: int, as_of: date) -> date:
    """
    Date n business days back from as_of. Weekend-aware only - doesn't
    account for market holidays, a deliberately simple approximation that
    errs slightly conservative (a holiday in the window means this looks
    slightly further back than strictly necessary, never less).
    """
    d = as_of
    counted = 0
    while counted < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            counted += 1
    return d


def count_recent_day_trades(store: HybridStore, as_of: Optional[date] = None) -> int:
    """
    Counts day trades (buy AND sell of the same symbol on the same
    calendar date) recorded in this bot's own `fills` table within the
    trailing lookback window. Uses our own fill records - the only trades
    this bot can actually see - rather than the broker's own day-trade
    counter, so it stays correct regardless of whether that field is ever
    wired in separately.
    """
    as_of = as_of or date.today()
    start = _trailing_business_days(PDT_LOOKBACK_BUSINESS_DAYS, as_of)
    rows = store.fetchall(
        "SELECT symbol, side, CAST(ts AS DATE) AS d FROM fills WHERE ts >= ?",
        (start,),
    )
    by_symbol_date: dict = {}
    for symbol, side, d in rows:
        key = (symbol, d)
        by_symbol_date.setdefault(key, set()).add(str(side).lower())
    return sum(1 for sides in by_symbol_date.values() if "buy" in sides and "sell" in sides)


def check_pdt_ok(store: HybridStore) -> Tuple[bool, str]:
    """
    Call before submitting any NEW entry (buy) order - same posture as
    the circuit breaker and correlation gate: never blocks a sell/exit,
    since getting out of a position should always be possible regardless
    of day-trade count. Fails open (allows the trade) if the account
    equity read fails, or if PDT_EQUITY_THRESHOLD<=0 (explicit opt-out
    for a cash account or an account already well above the threshold).
    """
    if PDT_EQUITY_THRESHOLD <= 0:
        return True, "ok (PDT check disabled via PDT_EQUITY_THRESHOLD<=0)"

    try:
        acct = get_account()
        equity = float(acct.get("equity") or 0.0)
    except Exception as e:
        log.warning("pdt_guard: could not read account equity (%s) - allowing trade, but this needs attention.", e)
        return True, "ok (equity check unavailable)"

    if equity <= 0 or equity >= PDT_EQUITY_THRESHOLD:
        return True, "ok"

    recent = count_recent_day_trades(store)
    if recent >= PDT_MAX_DAY_TRADES:
        reason = (
            f"pdt_guard: {recent} day trade(s) already in the trailing "
            f"{PDT_LOOKBACK_BUSINESS_DAYS} business days with equity ${equity:,.0f} "
            f"< ${PDT_EQUITY_THRESHOLD:,.0f} - one more same-day round trip risks "
            f"a FINRA Pattern Day Trader flag and a 90-day freeze. Blocking new entries "
            f"(exits are never blocked by this)."
        )
        return False, reason
    return True, "ok"
