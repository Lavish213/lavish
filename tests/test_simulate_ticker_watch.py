# Covers simulate_ticker_watch.py's offline, network-free logic: synthetic
# alert generation is reproducible given a seed, and classify_and_route()
# agrees with the real parser/routing/confidence-floor behavior already
# covered in test_pipeline_integration.py.
from __future__ import annotations

import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_ticker_watch import (
    TOP_TICKERS, FALLBACK_REFERENCE_PRICES, generate_synthetic_alerts, classify_and_route,
)

# All offline-mode tests pass price_lookups={ticker: None} explicitly to
# force the static-fallback path - deterministic, no network dependency,
# and exactly what this sandbox actually runs under.
_NO_NETWORK = {t: None for t in TOP_TICKERS}


def test_generation_is_reproducible_given_same_seed():
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    a = generate_synthetic_alerts(TOP_TICKERS, start, end, 5, random.Random(7), price_lookups=_NO_NETWORK)
    b = generate_synthetic_alerts(TOP_TICKERS, start, end, 5, random.Random(7), price_lookups=_NO_NETWORK)
    assert [(x.ticker, x.date, x.text) for x in a] == [(x.ticker, x.date, x.text) for x in b]


def test_generation_covers_every_ticker():
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    alerts = generate_synthetic_alerts(TOP_TICKERS, start, end, 3, random.Random(1), price_lookups=_NO_NETWORK)
    seen = {a.ticker for a in alerts}
    assert seen == set(TOP_TICKERS)


def test_options_strikes_are_anchored_near_reference_price():
    # Regression test for the bug found via a real GitHub Actions run:
    # strikes used to be a flat rng.uniform(50, 600) draw unrelated to the
    # ticker's real price, producing nonsense like a $599 call on AAPL
    # near $250 - which collapsed every Black-Scholes estimate to a
    # meaningless -100%. Strikes must now land within the documented
    # +/-10% (rounded to the nearest $5) band of the reference price.
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    alerts = generate_synthetic_alerts(TOP_TICKERS, start, end, 15, random.Random(3), price_lookups=_NO_NETWORK)
    options_alerts = [a for a in alerts if a.designed_kind == "options"]
    assert options_alerts, "expected at least one options alert in this sample"
    for a in options_alerts:
        ref = FALLBACK_REFERENCE_PRICES[a.ticker]
        strikes = [int(s) for s in re.findall(r"\$?(\d+)[cp]?\b", a.text) if s.isdigit()]
        assert strikes, f"no strike-looking number found in: {a.text!r}"
        strike = strikes[0]
        assert 0.85 * ref <= strike <= 1.15 * ref, (
            f"{a.ticker} strike {strike} not within 15% of reference ${ref}: {a.text!r}"
        )


def test_options_alerts_carry_a_short_intended_expiry():
    # Regression test for a bug found via a real GitHub Actions run: an
    # alert dated 2025-10-20 saying "exp 10/24" printed as expiring
    # 2026-10-24 downstream, because the real parser (correctly, for a
    # real live alert) anchors bare "10/24" text to the CURRENT real
    # year - which silently turns a synthetic past-dated alert into a
    # year-plus theoretical hold instead of the intended ~1-7 day weekly
    # option. intended_expiry must stay close to the alert's own date
    # regardless of what the real parser re-derives from the text.
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    alerts = generate_synthetic_alerts(TOP_TICKERS, start, end, 15, random.Random(3), price_lookups=_NO_NETWORK)
    options_alerts = [a for a in alerts if a.designed_kind == "options"]
    assert options_alerts, "expected at least one options alert in this sample"
    for a in options_alerts:
        assert a.intended_expiry is not None
        gap = (a.intended_expiry - a.date).days
        assert 0 < gap <= 7, f"{a.ticker} @ {a.date}: intended_expiry {a.intended_expiry} is {gap}d out"


def test_classify_and_route_recognizes_clear_options_alert():
    from simulate_ticker_watch import SyntheticAlert
    a = SyntheticAlert(ticker="AAPL", date=date.today(), text="AAPL $190 calls target 200 stop 180 exp 10/9",
                        designed_kind="options")
    r = classify_and_route(a)
    assert r["outcome"] == "trade:options"
    assert r["ticker"] == "AAPL"
    assert r["side"] == "CALL"


def test_classify_and_route_skips_ambiguous_text():
    from simulate_ticker_watch import SyntheticAlert
    a = SyntheticAlert(ticker="SPY", date=date.today(), text="SPY looking spicy today, watch this level",
                        designed_kind="ambiguous")
    r = classify_and_route(a)
    assert r["outcome"] == "skip:ticker_found_no_clear_direction"
