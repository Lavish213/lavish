# Robust adapter so train_from_images can always call parse_trade_text(...)
from __future__ import annotations
import re, importlib
from typing import Dict, Any, Callable, Optional

# Try to load the user's parser.py and pick whichever function exists
_CANDIDATES = [
    "parse_trade_text", "parse_trade", "parse_text",
    "parse_signal", "parse_order", "parse"
]

def _load_user_parser() -> Optional[Callable[[str], Any]]:
    try:
        mod = importlib.import_module("lavish_core.vision.parser")
    except Exception:
        return None
    for name in _CANDIDATES:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None

_USER_PARSE_FN = _load_user_parser()

def _fallback_parse(text: str) -> Dict[str, Any]:
    """
    Minimal heuristic so pipeline never crashes if user parser is missing.
    Extracts: label/action (BUY/SELL/CALL/PUT), ticker, qty, strike, expiry.
    """
    t = text.upper()

    # action/label
    label = ""
    if re.search(r"\b(BUY|CALL)\b", t):  label = "BUY" if "BUY" in t else "CALL"
    elif re.search(r"\b(SELL|PUT)\b", t): label = "SELL" if "SELL" in t else "PUT"

    # ticker (1–5 letters)
    m = re.search(r"\b([A-Z]{1,5})\b", t)
    ticker = m.group(1) if m else ""

    # qty
    m = re.search(r"\b(\d+)\s*(?:SH|SHARES|CT|CONTRACTS?)\b", t)
    quantity = int(m.group(1)) if m else None

    # strike like 123.45C / 123P, or 'STRIKE 123'
    strike = None
    opt_type = ""
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*([CP])\b", t)
    if m:
        strike = float(m.group(1))
        opt_type = "CALL" if m.group(2) == "C" else "PUT"
    else:
        m = re.search(r"STRIKE[^0-9]*([\d\.]+)", t)
        if m:
            strike = float(m.group(1))

    # simple expiry YYYY-MM-DD or MM/DD/YY
    expiry = ""
    m = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2})\b", t)
    if m:
        expiry = m.group(1).replace("/", "-")

    # confidence crude heuristic
    conf = 0.5
    conf += 0.2 if label else 0
    conf += 0.1 if ticker else 0
    conf += 0.1 if strike is not None else 0
    conf += 0.1 if expiry else 0
    conf = max(0.0, min(1.0, conf))

    return {
        "label": label or (opt_type if opt_type else ""),
        "action": label,
        "option_type": opt_type,
        "ticker": ticker,
        "quantity": quantity,
        "strike": strike,
        "expiry": expiry,
        "confidence": round(conf, 2),
    }

def parse_trade_text(text: str) -> Dict[str, Any]:
    """
    Single entry-point used by train_from_images. If the user's parser exposes any
    of the known function names, we call it. Otherwise we fall back to heuristics.
    We always return a dict with the expected keys.
    """
    if _USER_PARSE_FN:
        try:
            out = _USER_PARSE_FN(text)
            if isinstance(out, dict):
                # Ensure all expected keys exist
                return {
                    "label": out.get("label") or out.get("action", ""),
                    "action": out.get("action", out.get("label", "")),
                    "option_type": out.get("option_type", ""),
                    "ticker": out.get("ticker", ""),
                    "quantity": out.get("quantity"),
                    "strike": out.get("strike"),
                    "expiry": out.get("expiry", ""),
                    "confidence": out.get("confidence", 0.0),
                }
        except Exception:
            # fall through to heuristic
            pass
    return _fallback_parse(text or "")