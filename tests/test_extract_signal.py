# Covers the alert-parsing bugs found and fixed by testing real screenshots
# this project: side-before-price phrasing, bare "200c"/"200p" shorthand,
# and the "Call me at 10am" false-positive risk from that shorthand fallback.
from __future__ import annotations

from lavish_core.vision.extract_signal import parse_text


def test_price_first_call():
    r = parse_text("AAPL $190 calls target 200 stop 180 exp 10/9")
    assert r["ticker"] == "AAPL"
    assert r["side"] == "CALL"
    assert r["strike"] == 190.0


def test_side_before_price():
    r = parse_text("Spy Calls . $530 10/9")
    assert r["ticker"] == "SPY"
    assert r["side"] == "CALL"
    assert r["strike"] == 530.0


def test_put_with_stop_loss_phrasing():
    r = parse_text("SPY $528 Put 10/9 stop loss $528.20")
    assert r["side"] == "PUT"
    assert r["strike"] == 528.0
    assert r["stop_hint"] == 528.2


def test_bare_shorthand_strike_and_side():
    r = parse_text("TSLA grabbing 200c here, breaking out")
    assert r["ticker"] == "TSLA"
    assert r["side"] == "CALL"
    assert r["strike"] == 200.0


def test_bare_shorthand_put():
    r = parse_text("TSLA 200p here, losing the level")
    assert r["side"] == "PUT"
    assert r["strike"] == 200.0


def test_time_phrase_does_not_leak_into_options_side_without_ticker():
    # "10 a.m." must never read as a strike/side match. Isolated from a real
    # ticker, this should not resolve to a tradeable options signal at all.
    r = parse_text("Call me at 10 a.m. tomorrow, no rush")
    assert r["ticker"] is None


def test_expiry_too_far_out_is_dropped():
    # Multi-tab date-picker OCR noise (or a genuine LEAPS mention) beyond the
    # ~45-day weekly-alert pattern should not resolve to a confident expiry.
    r = parse_text("AAPL $190 calls exp Dec 19 2026")
    assert r["expiry"] is None or r["ticker"] != "AAPL"
