# lavish_core/trading/alert_handler.py
# Shared by every alert source (Patreon, Discord, anything added later):
# parse whatever text/image came in, build the signal shape
# trade_handler.execute_trade_from_post() expects, and hand it off.
# Previously Patreon and Discord each had their own parsing logic - Patreon's
# was a bare ticker+BUY/SELL regex with no strike/expiry/target/stop
# awareness at all, so a Patreon-sourced options alert silently downgraded
# into a plain equity guess. Both sources now get the same fidelity.
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional

from lavish_core.logger_setup import get_logger
from lavish_core.vision.extract_signal import parse_alert
from lavish_core.trading.trade_handler import execute_trade_from_post
from lavish_core.utils.alerts import post_discord

log = get_logger("alert_handler", log_dir="logs")

DEFAULT_CONFIDENCE = float(os.getenv("ALERT_DEFAULT_CONFIDENCE", "0.7"))

BUY_WORDS = ("BUY", "CALL", "LONG", "BTO", "ENTRY")
SELL_WORDS = ("SELL", "PUT", "SHORT", "STC", "EXIT", "CLOSE")

# Real screenshots seen this session showed her also trading perpetual
# futures/leveraged tokens on Hyperliquid ("MU/USDC-P", "QQQ tokenized
# perp") - a completely different instrument and risk/leverage profile
# than plain equity. Alpaca can't execute perps/leveraged tokens at all,
# and nothing here could safely translate "10x leveraged MU perp" into
# "buy $500 of plain MU shares" - that's not copying her trade, it's
# making an unrelated bet on the same ticker. Detect and skip loudly
# instead of silently downgrading it into an equity buy.
_PERP_RE = re.compile(
    r"\b[A-Z]{1,6}[-/]USDC?-?P\b"      # MU/USDC-P, SNDK-USDC-P style pair notation
    r"|\bperp(?:etual)?s?\b"           # "perp", "perps", "perpetual futures"
    r"|\btokeniz(?:ed)?\s+perp\b"      # "tokenized perp"
    r"|\b\d{1,3}x\s*lev(?:erage)?\b",  # "10x leverage" / "5x lev" - not bare "10x" (too hype-language-prone)
    re.IGNORECASE,
)


def _is_perp_or_leveraged(text: str) -> bool:
    return bool(_PERP_RE.search(text or ""))


def _equity_action_from_text(text: str) -> Optional[str]:
    # Word-boundary match, not substring - "w in up" would match "CALL"
    # inside "RECALL"/"CALLBACK" and "PUT" inside "DISPUTE"/"COMPUTE"/
    # "REPUTATION", turning ordinary commentary next to a real ticker into
    # a false BUY/SELL classification. Optional trailing "S" so plural
    # casual phrasing ("grabbing calls", "loading puts", "adding to my
    # longs", "exits the position") still matches - a bare \bCALL\b would
    # otherwise miss "CALLS" entirely (the "L"->"S" boundary doesn't
    # exist, both are word characters).
    up = (text or "").upper()
    has_buy = any(re.search(rf"\b{re.escape(w)}S?\b", up) for w in BUY_WORDS)
    has_sell = any(re.search(rf"\b{re.escape(w)}S?\b", up) for w in SELL_WORDS)
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
    if _is_perp_or_leveraged(text):
        log.warning("[%s] Alert looks like a perp/leveraged-token trade (not equity/options) - "
                    "skipping, can't be safely copied as a plain stock buy: %r", source, (text or "")[:200])
        post_discord(
            f"⚠️ Skipped a {source} alert that looks like a perpetual futures/leveraged-token trade, "
            f"not equity or options - this bot can't execute those and won't guess-translate it into "
            f"a plain stock buy (different leverage/risk profile entirely). Review manually if you want "
            f"to act on it: {(text or '')[:200]}"
        )
        return

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
        # She doesn't always give these on equity calls either, but when she
        # does, trade_handler prefers them over the fixed pct bracket.
        "target_hint": parsed.get("target_hint"),
        "stop_hint": parsed.get("stop_hint"),
    }
    log.info("[%s] Parsed equity alert: %s", source, signal)
    execute_trade_from_post(signal)
