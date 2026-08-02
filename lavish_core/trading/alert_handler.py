# lavish_core/trading/alert_handler.py
# Shared by every alert source (Patreon, Discord, anything added later):
# parse whatever text/image came in, build the signal shape
# trade_handler.execute_trade_from_post() expects, and hand it off.
# Previously Patreon and Discord each had their own parsing logic - Patreon's
# was a bare ticker+BUY/SELL regex with no strike/expiry/target/stop
# awareness at all, so a Patreon-sourced options alert silently downgraded
# into a plain equity guess. Both sources now get the same fidelity.
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from lavish_core.logger_setup import get_logger
from lavish_core.vision.extract_signal import parse_alert
from lavish_core.trading.trade_handler import execute_trade_from_post

log = get_logger("alert_handler", log_dir="logs")

DEFAULT_CONFIDENCE = float(os.getenv("ALERT_DEFAULT_CONFIDENCE", "0.7"))

BUY_WORDS = ("BUY", "CALL", "LONG", "BTO", "ENTRY")
SELL_WORDS = ("SELL", "PUT", "SHORT", "STC", "EXIT", "CLOSE")


def _equity_action_from_text(text: str) -> Optional[str]:
    up = (text or "").upper()
    has_buy = any(w in up for w in BUY_WORDS)
    has_sell = any(w in up for w in SELL_WORDS)
    if has_buy and not has_sell:
        return "BUY"
    if has_sell and not has_buy:
        return "SELL"
    return None


def handle_alert_text(
    text: str,
    image_path: Optional[str] = None,
    note: str = "",
    source: str = "alert",
    amount_usd: Optional[float] = None,
) -> None:
    """
    Core logic, independent of which source called it: parse an alert
    (text and/or an attached trade-card image) and hand it to the
    execution pipeline. Used by both the Patreon poller and the Discord
    listener so "whichever source posts first" behaves identically either
    way - trade_handler.execute_trade_from_post() dedupes the second
    source's copy of the same alert.
    """
    parsed = parse_alert(text=text, img_path=Path(image_path) if image_path else None, source=source)

    ticker = parsed.get("ticker")
    if not ticker:
        log.info("[%s] No ticker found in alert, skipping: %r", source, (text or "")[:200])
        return

    confidence = parsed.get("confidence") or DEFAULT_CONFIDENCE

    if parsed.get("side") in ("CALL", "PUT") and parsed.get("strike") and parsed.get("expiry"):
        signal = {
            "source": source,
            "ticker": ticker,
            "side": parsed["side"],
            "strike": parsed["strike"],
            "expiry": parsed["expiry"],
            "confidence": confidence,
            "target_hint": parsed.get("target_hint"),
            "stop_hint": parsed.get("stop_hint"),
            "note": note or parsed.get("notes_excerpt", ""),
            "amount_usd": amount_usd,
        }
        log.info("[%s] Parsed options alert: %s", source, signal)
        execute_trade_from_post(signal)
        return

    action = _equity_action_from_text(text)
    if not action:
        log.info("[%s] Ticker %s found but no clear BUY/SELL direction, skipping: %r",
                  source, ticker, (text or "")[:200])
        return

    signal = {
        "source": source,
        "action": action,
        "symbol": ticker,
        "confidence": confidence,
        "note": note or parsed.get("notes_excerpt", ""),
        "amount_usd": amount_usd,
    }
    log.info("[%s] Parsed equity alert: %s", source, signal)
    execute_trade_from_post(signal)
