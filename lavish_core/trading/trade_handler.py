# lavish_core/trading/trade_handler.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional

from lavish_core.logger_setup import get_logger
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB
from lavish_core.trade.trade_agent import place_trade, _dry_price  # reuse pricing logic

log = get_logger("trade", log_dir="logs")

CONF_FLOOR = float(os.getenv("SIGNAL_CONFIDENCE_FLOOR", "0.55"))
TRADE_MODE = os.getenv("TRADE_MODE", "dry").lower()  # dry | paper | live

def _coerce_side(action: str) -> Optional[str]:
    a = (action or "").strip().lower()
    if a in ("buy", "long"): return "buy"
    if a in ("sell", "short"): return "sell"
    return None

def execute_trade_from_post(signal: Dict[str, Any]) -> None:
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
    ref_price = _dry_price(sym)
    dollars = float(amt) if amt is not None else float(os.getenv("DEFAULT_TRADE_DOLLARS", "500"))
    qty = max(1.0, round(dollars / max(0.01, ref_price), 0))

    meta = {"source": signal.get("source", "patreon"), "confidence": conf, "note": note}

    log.info("🔔 signal → %s %s (qty=%.0f, conf=%.2f, mode=%s, ref=%.2f)",
             side.upper(), sym, qty, conf, TRADE_MODE, ref_price)

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
    )
    log.info("Trade result: %s", out)

def main():
    log.info("Trade handler ready (mode=%s, floor=%.2f).", TRADE_MODE, )