# lavish_core/trading/trade_handler.py
from __future__ import annotations
import os, json, time
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Any, Optional

from lavish_core.logger_setup import get_logger
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB
from lavish_core.trade.trade_agent import place_trade, _dry_price  # dry-run fallback pricing only
from lavish_core.trade.broker_alpaca import latest_quote, get_order as broker_get_order, cancel_order as broker_cancel_order
from lavish_core.trade.options_broker import resolve_and_price_contract, place_option_order
from lavish_core.trade.options_exit_monitor import watch_and_exit_async
from lavish_core.trade.equity_exit_monitor import watch_bracket_async
from lavish_core.trade.circuit_breaker import check_ok as circuit_breaker_check_ok, record_trade_outcome
from lavish_core.trade.reconcile import mark_watched, unmark_watched
from lavish_core.trade.portfolio_risk import check_correlation_ok
from lavish_core.trade.pdt_guard import check_pdt_ok

log = get_logger("trade", log_dir="logs")

CONF_FLOOR = float(os.getenv("SIGNAL_CONFIDENCE_FLOOR", "0.55"))
TRADE_MODE = os.getenv("TRADE_MODE", "dry").lower()  # dry | paper | live

# Staged capital rollout: scales every computed position size uniformly, so
# going live can start at a fraction of intended size (0.1-0.25) and ramp up
# as real performance confirms the bot, without touching
# DEFAULT_TRADE_DOLLARS/DEFAULT_OPTION_TRADE_DOLLARS or amount_usd signals
# themselves. 1.0 = no scaling (default, unchanged behavior).
POSITION_SIZE_SCALE = float(os.getenv("POSITION_SIZE_SCALE", "1.0"))

# A real contract seen this session quoted bid $23 / ask $2,690 - a spread
# so wide the "mid price" isn't a fair fill on either side. Reject rather
# than submit a limit order into that. Expressed as spread / mid, not
# spread / bid, so a near-zero bid on a dead contract doesn't produce a
# nonsensical (or divide-by-near-zero) ratio.
OPTION_MAX_SPREAD_PCT = float(os.getenv("OPTION_MAX_SPREAD_PCT", "0.50"))

# Same fill-confirmation pattern trade_agent.place_trade uses for equity
# orders, applied to options entries too - previously an options order was
# submitted and the exit monitor started immediately after, with no check
# that the limit order (at mid, which can sit unfilled on a wide spread)
# actually filled first.
OPTIONS_FILL_POLL_ATTEMPTS = int(os.getenv("OPTIONS_FILL_POLL_ATTEMPTS", "5"))
OPTIONS_FILL_POLL_INTERVAL_SEC = float(os.getenv("OPTIONS_FILL_POLL_INTERVAL_SEC", "1.0"))

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

    # Thin/illiquid contracts can quote a "mid" that isn't actually
    # tradeable - a real example seen this session was a contract with
    # bid $23 / ask $2,690 ("mid" $1,356.50, nowhere near a fair fill on
    # either side). A limit order at that mid either doesn't fill at all
    # or, worse, does fill and burns most of the position's value on the
    # spread alone. Reject outright rather than submit into that.
    bid, ask = contract.get("bid"), contract.get("ask")
    if bid is not None and ask is not None and bid > 0:
        spread_pct = (ask - bid) / mid_price
        if spread_pct > OPTION_MAX_SPREAD_PCT:
            log.warning("Skip options trade: %s spread too wide (bid=%.2f ask=%.2f mid=%.2f, %.0f%% > %.0f%% max)",
                        contract["symbol"], bid, ask, mid_price, spread_pct * 100, OPTION_MAX_SPREAD_PCT * 100)
            return

    dollars = (float(amt) if amt is not None else DEFAULT_OPTION_TRADE_DOLLARS) * POSITION_SIZE_SCALE
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

    corr_ok, corr_reason = check_correlation_ok(ticker)
    if not corr_ok:
        store.submit_order(
            symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
            limit_price=mid_price, tif="day", venue=TRADE_MODE, status="rejected",
            meta={"reason": corr_reason, "source": signal.get("source", "discord"), "note": note},
        )
        log.error("Skip options trade: %s", corr_reason)
        return

    pdt_ok, pdt_reason = check_pdt_ok(store)
    if not pdt_ok:
        store.submit_order(
            symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
            limit_price=mid_price, tif="day", venue=TRADE_MODE, status="rejected",
            meta={"reason": pdt_reason, "source": signal.get("source", "discord"), "note": note},
        )
        log.error("Skip options trade: %s", pdt_reason)
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

    # Confirm the entry actually filled before starting exit monitoring -
    # a limit order resting at mid on a wide options spread can sit open
    # for a while (or never fill). Mirrors trade_agent.place_trade's
    # equity fill-poll; without this, the exit monitor previously started
    # tracking a position that might not exist yet.
    broker_oid = order.get("id")
    final_status = order.get("status", "submitted")
    filled_qty = order.get("filled_qty")
    filled_avg_price = order.get("filled_avg_price")
    if broker_oid and TRADE_MODE in ("paper", "live"):
        for _ in range(OPTIONS_FILL_POLL_ATTEMPTS):
            if final_status in ("filled", "canceled", "expired", "rejected"):
                break
            time.sleep(OPTIONS_FILL_POLL_INTERVAL_SEC)
            try:
                polled = broker_get_order(broker_oid)
                final_status = polled.get("status", final_status)
                filled_qty = polled.get("filled_qty", filled_qty)
                filled_avg_price = polled.get("filled_avg_price", filled_avg_price)
            except Exception as e:
                log.warning("Options order status poll failed for %s: %s", broker_oid, e)
                break

    oid = store.submit_order(
        symbol=contract["symbol"], side="buy", qty=qty, order_type="limit",
        limit_price=mid_price, tif="day", venue=TRADE_MODE, status=final_status,
        client_id=order.get("client_order_id"),
        meta={"broker": "alpaca", "raw": order, "source": signal.get("source", "discord"), "note": note,
              "ticker": ticker, "option_side": option_side, "strike": float(strike), "expiry": expiry.isoformat(),
              "expected_price": mid_price, "filled_avg_price": filled_avg_price},
    )

    if final_status not in ("filled", "partially_filled"):
        log.warning("Options order for %s never confirmed filled after %d polls (status=%s) - canceling, no exit monitor started.",
                    contract["symbol"], OPTIONS_FILL_POLL_ATTEMPTS, final_status)
        if broker_oid and TRADE_MODE in ("paper", "live"):
            try:
                broker_cancel_order(broker_oid)
            except Exception as e:
                log.warning("Cancel of unfilled options order %s failed: %s", broker_oid, e)
        return

    # Use the real fill price (slippage vs. the mid_price we sized/quoted
    # against) for both the exit-monitor's entry baseline and P&L, not the
    # pre-fill estimate.
    entry_price = float(filled_avg_price) if filled_avg_price else mid_price
    filled_qty_n = int(float(filled_qty)) if filled_qty else qty
    slippage = round(entry_price - mid_price, 4)
    log.info("Options entry filled: %s qty=%s entry=%.2f (quoted mid=%.2f, slippage=%.4f)",
              contract["symbol"], filled_qty_n, entry_price, mid_price, slippage)
    store.log_fill(
        order_id=oid, symbol=contract["symbol"], side="buy",
        qty=filled_qty_n, price=entry_price, fee=0.0, venue=TRADE_MODE,
    )

    if target_hint is None and stop_hint is None:
        log.info("No target/stop given for %s - relying on our own guardrails (stop/trail/expiry) for the exit.",
                  contract["symbol"])

    def _on_exit(result: Dict[str, Any]) -> None:
        # Runs in the monitor's background thread - use a fresh store
        # connection rather than sharing one across threads.
        unmark_watched(contract["symbol"])
        exit_store = HybridStore(duckdb_path=str(DEFAULT_DB), redis_url=os.environ.get("REDIS_URL") or None)
        exit_price = result.get("option_price")
        if result.get("status") == "exited" and exit_price is not None:
            pnl = (float(exit_price) - entry_price) * 100 * filled_qty_n
            exit_store.log_fill(
                order_id=oid, symbol=contract["symbol"], side="sell",
                qty=filled_qty_n, price=float(exit_price), fee=0.0, venue=TRADE_MODE,
            )
            record_trade_outcome(exit_store, pnl)
            log.info("Closed %s: entry=%.2f exit=%.2f realized_pnl=%.2f (%s)",
                      contract["symbol"], entry_price, exit_price, pnl, result.get("reason"))
        else:
            log.warning("Exit monitor for %s ended without a clean fill: %s", contract["symbol"], result)

    mark_watched(contract["symbol"])
    watch_and_exit_async(
        underlying_symbol=ticker,
        contract_symbol=contract["symbol"],
        qty=filled_qty_n,
        option_side=option_side,
        entry_price=entry_price,
        expiry=expiry,
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
    target_hint = signal.get("target_hint")
    stop_hint = signal.get("stop_hint")

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
    dollars = (float(amt) if amt is not None else float(os.getenv("DEFAULT_TRADE_DOLLARS", "500"))) * POSITION_SIZE_SCALE
    qty = max(1.0, round(dollars / max(0.01, ref_price), 0))

    meta = {"source": signal.get("source", "patreon"), "confidence": conf, "note": note}

    # Only bracket new long entries. A "sell" here is closing/shorting, not
    # opening a position, so there's nothing to attach a bracket exit to.
    # Prefer her stated target/stop when she gave one (same principle as the
    # options path) - only fall back to our fixed pct bracket for whichever
    # side she didn't specify, or if her number fails a basic sanity check
    # (a target below current price or a stop above it is a parsing miss,
    # not a real level - trust the safe default instead of that number).
    take_profit = stop_loss = None
    if side == "buy" and ref_price > 0:
        take_profit = round(ref_price * (1 + TAKE_PROFIT_PCT), 2)
        stop_loss = round(ref_price * (1 - STOP_LOSS_PCT), 2)
        if target_hint is not None:
            t = float(target_hint)
            if t > ref_price:
                take_profit = round(t, 2)
            else:
                log.warning("Ignoring target_hint %.2f for %s (not above ref %.2f) - using default TP",
                            t, sym, ref_price)
        if stop_hint is not None:
            s = float(stop_hint)
            if s < ref_price:
                stop_loss = round(s, 2)
            else:
                log.warning("Ignoring stop_hint %.2f for %s (not below ref %.2f) - using default SL",
                            s, sym, ref_price)

    # Same concentration check as the options path - only gates new long
    # entries, same reasoning as the bracket above (a sell is reducing/
    # closing exposure, not adding to it).
    if side == "buy" and TRADE_MODE in ("paper", "live"):
        corr_ok, corr_reason = check_correlation_ok(sym)
        if not corr_ok:
            store.submit_order(
                symbol=sym, side=side, qty=qty, order_type="market",
                tif="day", venue=TRADE_MODE, status="rejected",
                meta={"reason": corr_reason, **meta},
            )
            log.error("Skip trade: %s", corr_reason)
            return

        pdt_ok, pdt_reason = check_pdt_ok(store)
        if not pdt_ok:
            store.submit_order(
                symbol=sym, side=side, qty=qty, order_type="market",
                tif="day", venue=TRADE_MODE, status="rejected",
                meta={"reason": pdt_reason, **meta},
            )
            log.error("Skip trade: %s", pdt_reason)
            return

    log.info("🔔 signal → %s %s (qty=%.0f, conf=%.2f, mode=%s, ref=%.2f, tp=%s, sl=%s, her_target=%s, her_stop=%s)",
             side.upper(), sym, qty, conf, TRADE_MODE, ref_price, take_profit, stop_loss, target_hint, stop_hint)

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

    # A bracket's take_profit/stop_loss legs fill on Alpaca's side with no
    # signal back to this bot unless something watches for it - without
    # this, the entry's DB order just sits "filled" forever with no linked
    # exit, and track_record.py has no way to know the position closed.
    legs = out.get("legs") or []
    if side == "buy" and out.get("status") == "filled" and (take_profit is not None or stop_loss is not None) and legs:
        leg_ids = [leg.get("id") for leg in legs if leg.get("id")]
        if leg_ids:
            entry_price = float(out.get("filled_avg_price") or ref_price)
            watch_bracket_async(
                order_id=out["order_id"], symbol=sym, qty=qty,
                entry_price=entry_price, leg_ids=leg_ids, venue=TRADE_MODE,
            )

def main():
    log.info("Trade handler ready (mode=%s, floor=%.2f).", TRADE_MODE, )