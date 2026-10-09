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
# for it - using her original stated target/stop when we can recover it
# (persisted in the entry order's meta - see trade_handler.py), falling
# back to our own defaults only when that lookup comes up empty.
from __future__ import annotations
import json, os, re, time, threading, logging
from datetime import date, datetime
from typing import Optional, Dict, Any, Set, Tuple

from lavish_core.trade.broker_alpaca import get_positions
from lavish_core.trade.options_exit_monitor import watch_and_exit_async
from lavish_core.utils.alerts import post_discord
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB

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


def _find_original_target_stop(contract_symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Looks up the most recent 'buy' order for this contract to recover her
    original stated target/stop (persisted in its meta by trade_handler.py).
    Returns (None, None) on any lookup failure or if the order predates
    this field being persisted - callers must treat that the same as
    "she didn't give one", not as an error.
    """
    try:
        store = HybridStore(duckdb_path=str(DEFAULT_DB))
        rows = store.fetchall(
            "SELECT meta FROM orders WHERE symbol = ? AND side = 'buy' ORDER BY ts DESC LIMIT 1",
            (contract_symbol,),
        )
        if not rows:
            return None, None
        meta_raw = rows[0][0]
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        target = meta.get("target_underlying")
        stop = meta.get("stop_underlying")
        return (float(target) if target is not None else None,
                float(stop) if stop is not None else None)
    except Exception as e:
        log.warning("reconcile: could not recover original target/stop for %s: %s", contract_symbol, e)
        return None, None


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

        target_underlying, stop_underlying = _find_original_target_stop(symbol)
        recovered_her_levels = target_underlying is not None or stop_underlying is not None

        log.warning(
            "reconcile: found unmanaged open position %s (%s x%s, entry=%.2f, exp=%s) - "
            "starting a guardrail monitor (%s).",
            symbol, parsed["side"], qty, entry_price, parsed["expiry"],
            f"recovered her target={target_underlying} stop={stop_underlying}" if recovered_her_levels
            else "no original target/stop found - our defaults only",
        )
        post_discord(
            f"⚠️ Recovered an unmanaged options position on startup/reconcile: "
            f"{parsed['ticker']} {parsed['side']} ${parsed['strike']:.2f} exp {parsed['expiry']} "
            f"(qty={qty}, entry={entry_price:.2f}). " +
            (f"Recovered her stated target={target_underlying} stop={stop_underlying} from the order log - "
             f"watching those plus our own guardrails."
             if recovered_her_levels else
             "No original stop/target from her was found in the order log - "
             "running our own stop-loss/trailing/expiry guardrails on it now.")
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
            target_underlying=target_underlying,
            stop_underlying=stop_underlying,
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
