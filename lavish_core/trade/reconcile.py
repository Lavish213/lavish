# lavish_core/trade/reconcile.py
# Every open-position exit guardrail (stop/trail/expiry-close) runs as an
# in-memory background thread in options_exit_monitor.py. That thread only
# exists while this process is alive - if run_ingest.py crashes, gets
# redeployed, or is just restarted, every open options position instantly
# has NO ONE watching it: no stop-loss, no trailing stop, no expiry-day
# close. Alpaca has no server-side dead-man's-switch of its own (checked -
# it doesn't auto-cancel/flatten on client disconnect), so this has to be
# handled here: on startup, and periodically thereafter, find any open
# option position that isn't being watched and start a guardrail monitor
# for it using our own default rules (her original stated target/stop
# isn't persisted anywhere, so a recovered position runs on our
# guardrails only - which is the whole point of having them).
from __future__ import annotations
import os, re, time, threading, logging
from datetime import date, datetime
from typing import Optional, Dict, Any, Set

from lavish_core.trade.broker_alpaca import get_positions
from lavish_core.trade.options_exit_monitor import watch_and_exit_async
from lavish_core.utils.alerts import post_discord

log = logging.getLogger("reconcile")

RECONCILE_INTERVAL_SECONDS = float(os.getenv("RECONCILE_INTERVAL_SECONDS", "120"))

# OCC symbol: TICKER + YYMMDD + C/P + strike*1000 zero-padded to 8 digits
_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

# In-process record of which contract symbols already have an active
# monitor, so reconciliation doesn't start a second one for a position
# this same process opened itself moments ago.
_watched: Set[str] = set()
_watched_lock = threading.Lock()


def mark_watched(contract_symbol: str) -> None:
    with _watched_lock:
        _watched.add(contract_symbol)


def unmark_watched(contract_symbol: str) -> None:
    with _watched_lock:
        _watched.discard(contract_symbol)


def parse_occ_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    ticker, yymmdd, cp, strike_digits = m.groups()
    try:
        expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "ticker": ticker,
        "expiry": expiry,
        "side": "CALL" if cp == "C" else "PUT",
        "strike": int(strike_digits) / 1000.0,
    }


def reconcile_open_positions() -> int:
    """
    Finds open long option positions with no active monitor in this
    process and starts one for each, using our own default guardrails.
    Returns the number recovered. Safe to call repeatedly - a position
    already in _watched is skipped.
    """
    try:
        positions = get_positions()
    except Exception as e:
        log.error("reconcile: could not fetch positions: %s", e)
        return 0

    recovered = 0
    for pos in positions:
        if pos.get("asset_class") != "us_option":
            continue
        if pos.get("side") != "long" or float(pos.get("qty", 0)) <= 0:
            continue  # only long calls/puts are ever opened by this bot

        symbol = pos.get("symbol", "")
        with _watched_lock:
            already_watched = symbol in _watched
        if already_watched:
            continue

        parsed = parse_occ_symbol(symbol)
        if not parsed:
            log.warning("reconcile: found option position %s but couldn't parse its OCC symbol - skipping.", symbol)
            continue

        try:
            entry_price = float(pos.get("avg_entry_price", 0) or 0)
            qty = int(float(pos.get("qty", 0)))
        except (TypeError, ValueError):
            log.warning("reconcile: bad qty/avg_entry_price on position %s - skipping.", symbol)
            continue
        if entry_price <= 0 or qty <= 0:
            continue

        log.warning(
            "reconcile: found unmanaged open position %s (%s x%s, entry=%.2f, exp=%s) - "
            "starting a guardrail monitor with our own defaults (no original target/stop to recover).",
            symbol, parsed["side"], qty, entry_price, parsed["expiry"],
        )
        post_discord(
            f"⚠️ Recovered an unmanaged options position on startup/reconcile: "
            f"{parsed['ticker']} {parsed['side']} ${parsed['strike']:.2f} exp {parsed['expiry']} "
            f"(qty={qty}, entry={entry_price:.2f}). No original stop/target from her was persisted - "
            f"running our own stop-loss/trailing/expiry guardrails on it now."
        )

        mark_watched(symbol)

        def _on_exit(result: Dict[str, Any], _symbol=symbol) -> None:
            unmark_watched(_symbol)
            if result.get("status") not in ("exited",):
                post_discord(f"⚠️ Exit monitor for recovered position {_symbol} ended abnormally: {result}")

        watch_and_exit_async(
            underlying_symbol=parsed["ticker"],
            contract_symbol=symbol,
            qty=qty,
            option_side=parsed["side"],
            entry_price=entry_price,
            expiry=parsed["expiry"],
            target_underlying=None,
            stop_underlying=None,
            on_exit=_on_exit,
        )
        recovered += 1

    return recovered


def run_periodic_reconciliation() -> None:
    """Blocking loop - run in its own background thread from run_ingest.py."""
    log.info("Reconciliation loop started (every %.0fs).", RECONCILE_INTERVAL_SECONDS)
    while True:
        try:
            n = reconcile_open_positions()
            if n:
                log.info("reconcile: recovered %d unmanaged position(s).", n)
        except Exception as e:
            log.error("reconcile: unexpected error: %s", e)
        time.sleep(RECONCILE_INTERVAL_SECONDS)
