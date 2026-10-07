# Real-pipeline integration test, no network/credentials: runs the same
# sample alerts as simulate_alerts.py (deliberately includes the real
# phrasing bugs found and fixed by testing actual screenshots this
# project) through the real parser + routing functions, with assertions
# instead of printed output - so a regression here fails CI/pytest
# instead of requiring a manual simulate_alerts.py read-through.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_alerts import build_sample_alerts
from lavish_core.vision.extract_signal import parse_text
from lavish_core.trading.alert_handler import _is_perp_or_leveraged, _equity_action_from_text


def _classify(text: str) -> str:
    """Mirrors simulate_alerts.dry_run()'s decision tree, returns a short label."""
    if _is_perp_or_leveraged(text):
        return "skip:perp"
    r = parse_text(text)
    ticker = r.get("ticker")
    if not ticker:
        return "skip:no_ticker"
    if r.get("side") in ("CALL", "PUT") and r.get("strike") and r.get("expiry"):
        return f"trade:options:{ticker}:{r['side']}:{r['strike']}"
    action = _equity_action_from_text(text)
    if action:
        return f"trade:equity:{action}:{ticker}"
    return "skip:no_direction"


def test_sample_alerts_classify_as_designed():
    alerts = {a["label"]: a["text"] for a in build_sample_alerts()}

    price_first = next(t for label, t in alerts.items() if "price-first" in label)
    assert _classify(price_first).startswith("trade:options:AAPL:CALL:190")

    put_stop = next(t for label, t in alerts.items() if "stop-loss phrasing" in label)
    assert _classify(put_stop).startswith("trade:options:SPY:PUT:528")

    side_before_price = next(t for label, t in alerts.items() if "Side-before-price" in label)
    assert _classify(side_before_price).startswith("trade:options:SPY:CALL:530")

    casual_equity = next(t for label, t in alerts.items() if "Casual equity" in label)
    assert _classify(casual_equity) == "trade:equity:BUY:NVDA"

    ambiguous = next(t for label, t in alerts.items() if "Ambiguous" in label)
    assert _classify(ambiguous) == "skip:no_direction"

    perp = next(t for label, t in alerts.items() if "Perp/leveraged" in label)
    assert _classify(perp) == "skip:perp"

    off_whitelist = next(t for label, t in alerts.items() if "Unknown ticker" in label)
    assert _classify(off_whitelist) == "skip:no_ticker"

    ddog = next(t for label, t in alerts.items() if "DDOG" in label)
    assert _classify(ddog).startswith("trade:options:DDOG:CALL:130")


def test_build_sample_alerts_returns_eight_cases():
    # Pins the count so a future edit that silently drops a case is caught.
    assert len(build_sample_alerts()) == 8
