# lavish_core/trading/trade_handler.py
from __future__ import annotations
import os, json
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Any, Optional

from lavish_core.logger_setup import get_logger
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB
from lavish_core.trade.trade_agent import place_trade, _dry_price  # dry-run fallback pricing only
from lavish_core.trade.broker_alpaca import latest_quote
from lavish_core.trade.options_broker import resolve_and_price_contract, place_option_order
from lavish_core.trade.options_exit_monitor import watch_and_exit_async
from lavish_core.trade.circuit_breaker import check_ok as circuit_breaker_check_ok, record_trade_outcome

log = get_logger("trade", log_dir="logs")

CONF_FLOOR = float(os.getenv("SIGNAL_CONFIDENCE_FLOOR", "0.55"))
TRADE_MODE = os.getenv("TRADE_MODE", "dry").lower()  # dry | paper | live

# Bracket protection for new long entries. Without this, a submitted order
# had no exit plan at all - a losing position just sat there indefinitely.
STOP_LOSS_PCT = float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.015"))
TAKE_PROFIT_PCT = float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.02"))

DEFAULT_OPTION_TRADE_DOLLARS = float(os.getenv("DEFAULT_OPTION_TRADE_DOLLARS", "200"))
OPTION_STRIKE_TOLERANCE = float(os.getenv("OPTION_STRIKE_TOLERANCE", "5"))

# She posts to Patreon and Discord independently, and it's not consistently
# one before the other - sometimes Patreon is first by a few minutes. Watch
# both, whichever fires first executes, and skip the same alert showing up
# again on the other source within this window.
SIGNAL_DEDUP_WINDOW_SECONDS = float(os.getenv("SIGNAL_DEDUP_WINDOW_SECONDS", "300"))

def _coerce_side(action: str) -> Optional[str]:
    a = (action or "").strip().lower()
    if a in ("buy", "long"): return "buy"
    if a in ("sell", "short"): return "sell"
    return None

def _find_duplicate_signal(
    store: HybridStore, symbol: str, side: str, strike: Optional[float] = None, expiry: Optional[str] = None,
) -> Optional[str]:
    """Returns the source of an already-recorded matching signal within the dedup window, or None."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=SIGNAL_DEDUP_WINDOW_SECONDS)
    rows = store.fetchall(
        "SELECT source, payload FROM signals WHERE symbol=? AND side=? AND ts > ? ORDER BY ts DESC",
        (symbol.upper(), side.lower(), cutoff),
    )
    for src, payload_raw in rows:
        if strike is None and expiry is None:
            return src
        try:
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
        except Exception:
            payload = {}
        row_strike, row_expiry = payload.get("strike"), payload.get("expiry")
        if row_strike is not None and expiry == row_expiry and abs(float(row_strike) - float(strike)) < 0.01:
            return src
    return None

def execute_trade_from_post(signal: Dict[str, Any]) -> None:
    """
    Dispatches to the options or equity path based on the signal's shape.
    Dedupes against the *other* alert source first - see
    SIGNAL_DEDUP_WINDOW_SECONDS above.

    Options signal (from extract_signal.parse_alert/parse_text):
      { ticker, side('CALL'|'PUT'), strike, expiry('YYYY-MM-DD'),
        confidence, target_hint(underlying), stop_hint(underlying),
        amount_usd(optional), source, note }

    Equity signal (from patreon_trigger.py etc):
      { source, action('BUY'|'SELL'|etc), symbol, confidence(0..1),
        amount_usd(optional), note(optional) }
    """
    is_option = bool(signal.get("strike") and signal.get("expiry")
                      and str(signal.get("side", "")).upper() in ("CALL", "PUT"))
    symbol = str(signal.get("ticker") or signal.get("symbol") or "").upper().strip()
    dedup_side = str(signal.get("side") or signal.get("action") or "").lower().strip()
    source = signal.get("source", "unknown")

    if symbol and dedup_side:
        store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
        strike = float(signal["strike"]) if is_option else None
        expiry = str(signal.get("expiry")) if is_option else None
        dup_source = _find_duplicate_signal(store, symbol, dedup_side, strike, expiry)
        if dup_source:
            log.info("Skip %s %s: duplicate of %s's alert within %.0fs (source=%s)",
                      dedup_side.upper(), symbol, dup_source, SIGNAL_DEDUP_WINDOW_SECONDS, source)
            return
        store.log_signal(
            symbol=symbol, side=dedup_side, source=source,
            confidence=float(signal.get("confidence", 0) or 0),
            payload={"strike": strike, "expiry": expiry, "note": signal.get("note", "")},
        )

    if is_option:
        _execute_option_trade(signal)
    else:
        _execute_equity_trade(signal)

def _execute_option_trade(signal: Dict[str, Any]) -> None:
    ticker = str(signal.get("ticker") or signal.get("symbol") or "").upper().strip()
    option_side = str(signal.get("side", "")).upper().strip()
    strike = signal.get("strike")
    expiry_raw = signal.get("expiry")
    conf = float(signal.get("confidence", 0) or 0)
    target_hint = signal.get("target_hint")
    stop_hint = signal.get("stop_hint")
    amt = signal.get("amount_usd")
    note = signal.get("note", "")

    if not ticker or not strike or not expiry_raw:
        log.info("Skip options trade: missing ticker/strike/expiry in %s", signal)
        return
    if conf < CONF_FLOOR:
        log.info("Skip options trade: confidence %.2f < floor %.2f (%s %s %s)",
                  conf, CONF_FLOOR, ticker, option_side, strike)
        return

    try:
        expiry = expiry_raw if isinstance(expiry_raw, date) else datetime.fromisoformat(str(expiry_raw)).date()
    except Exception as e:
        log.warning("Skip options trade: unparseable expiry %r: %s", expiry_raw, e)
        return

    contract = resolve_and_price_contract(
        ticker, expiry, option_side, float(strike), strike_tolerance=OPTION_STRIKE_TOLERANCE,
    )
    if not contract:
        log.warning("Skip options trade: no listed contract found near %s %s $%.2f exp %s",
                     ticker, option_side, float(strike), expiry)
        return

    mid_price = contract.get("mid_price")
    if not mid_price or mid_price <= 0:
        log.warning("Skip options trade: no usable quote for %s", contract.get("symbol"))
        return

    dollars = float(amt) if amt is not None else DEFAULT_OPTION_TRADE_DOLLARS
    qty = max(1, int(dollars // (mid_price * 100)))

    log.info("🔔 options signal → %s %s $%.2f exp %s (contract=%s qty=%s mid=%.2f mode=%s target=%s stop=%s)",
              option_side, ticker, float(strike), expiry, contract["symbol"], qty, mid_price,
              TRADE_MODE, target_hint, stop_hint)

    if TRADE_MODE not in ("paper", "live"):
        log.info("dry mode: would BUY %s x%s @ ~%.2f (no order submitted)", contract["symbol"], qty, mid_price)
        return

    store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)

    cb_ok, cb_reason = circuit_breaker_check_ok(store)
    if not cb_ok:
        store.submit_order(
            symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
            limit_price=mid_price, tif="day", venue=TRADE_MODE, status="rejected",
            meta={"reason": f"circuit_breaker: {cb_reason}", "source": signal.get("source", "discord"), "note": note},
        )
        log.error("Skip options trade: circuit_breaker: %s", cb_reason)
        return

    try:
        order = place_option_order(
            contract_symbol=contract["symbol"], side="buy", qty=qty,
            order_type="limit", limit_price=mid_price,
        )
    except Exception as e:
        log.error("Options order failed for %s: %s", contract["symbol"], e)
        store.submit_order(
            symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
            limit_price=mid_price, tif="day", venue=TRADE_MODE, status="rejected",
            meta={"reason": str(e), "source": signal.get("source", "discord"), "note": note},
        )
        return

    log.info("Options order result: %s", order)
    oid = store.submit_order(
        symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
        limit_price=mid_price, tif="day", venue=TRADE_MODE, status=order.get("status", "submitted"),
        client_id=order.get("client_order_id"),
        meta={"broker": "alpaca", "raw": order, "source": signal.get("source", "discord"), "note": note,
              "ticker": ticker, "option_side": option_side, "strike": float(strike), "expiry": expiry.isoformat()},
    )

    if target_hint is None and stop_hint is None:
        log.warning("No target/stop given for %s - position has no automated exit plan.", contract["symbol"])
        return

    def _on_exit(result: Dict[str, Any]) -> None:
        # Runs in the monitor's background thread - use a fresh store
        # connection rather than sharing one across threads.
        exit_store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
        exit_price = result.get("option_price")
        if result.get("status") == "exited" and exit_price is not None:
            pnl = (float(exit_price) - mid_price) * 100 * qty
            exit_store.log_fill(
                order_id=oid, symbol=contract["symbol"], side="sell",
                qty=qty, price=float(exit_price), fee=0.0, venue=TRADE_MODE,
            )
            record_trade_outcome(exit_store, pnl)
            log.info("Closed %s: entry=%.2f exit=%.2f realized_pnl=%.2f (%s)",
                      contract["symbol"], mid_price, exit_price, pnl, result.get("reason"))
        else:
            log.warning("Exit monitor for %s ended without a clean fill: %s", contract["symbol"], result)

    watch_and_exit_async(
        underlying_symbol=ticker,
        contract_symbol=contract["symbol"],
        qty=qty,
        option_side=option_side,
        target_underlying=float(target_hint) if target_hint is not None else None,
        stop_underlying=float(stop_hint) if stop_hint is not None else None,
        on_exit=_on_exit,
    )

def _execute_equity_trade(signal: Dict[str, Any]) -> None:
    """
    Expected signal fields:
      { source, action('BUY'|'SELL'|etc), symbol, confidence(0..1),
        amount_usd(optional), note(optional) }
    """
    sym   = str(signal.get("symbol", "")).upper().strip()
    side  = _coerce_side(str(signal.get("action", "")))
    conf  = float(signal.get("confidence", 0) or 0)
    note  = signal.get("note", "")
    amt   = signal.get("amount_usd")  # may be None

    if not sym or not side:
        log.info("Skip trade: missing symbol/side in %s", signal)
        return
    if conf < CONF_FLOOR:
        log.info("Skip trade: confidence %.2f < floor %.2f (%s %s)", conf, CONF_FLOOR, side, sym)
        return

    # Open HybridStore (DuckDB/Redis) for audit + risk gates
    store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)

    # Size: if amount_usd provided → qty = amount / ref_price; else default $500 block
    # Use a real market quote when we have broker creds; a hash-based dummy
    # price would size real orders against a number unrelated to the market.
    try:
        ref_price = latest_quote(sym) or _dry_price(sym)
    except Exception as e:
        log.warning("latest_quote failed for %s, falling back to dry price: %s", sym, e)
        ref_price = _dry_price(sym)
    dollars = float(amt) if amt is not None else float(os.getenv("DEFAULT_TRADE_DOLLARS", "500"))
    qty = max(1.0, round(dollars / max(0.01, ref_price), 0))

    meta = {"source": signal.get("source", "patreon"), "confidence": conf, "note": note}

    # Only bracket new long entries. A "sell" here is closing/shorting, not
    # opening a position, so there's nothing to attach a bracket exit to.
    take_profit = stop_loss = None
    if side == "buy" and ref_price > 0:
        take_profit = round(ref_price * (1 + TAKE_PROFIT_PCT), 2)
        stop_loss = round(ref_price * (1 - STOP_LOSS_PCT), 2)

    log.info("🔔 signal → %s %s (qty=%.0f, conf=%.2f, mode=%s, ref=%.2f, tp=%s, sl=%s)",
             side.upper(), sym, qty, conf, TRADE_MODE, ref_price, take_profit, stop_loss)

    out = place_trade(
        store=store,
        symbol=sym,
        side=side,
        qty=qty,
        mode=TRADE_MODE,
        order_type="market",
        limit_price=None,
        tif="day",
        client_id=None,
        meta=meta,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    log.info("Trade result: %s", out)

def main():
    log.info("Trade handler ready (mode=%s, floor=%.2f).", TRADE_MODE, )