# Covers simulate_ticker_watch.py's offline, network-free logic: synthetic
# alert generation is reproducible given a seed, and classify_and_route()
# agrees with the real parser/routing/confidence-floor behavior already
# covered in test_pipeline_integration.py.
from __future__ import annotations

import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulate_ticker_watch import TOP_TICKERS, generate_synthetic_alerts, classify_and_route


def test_generation_is_reproducible_given_same_seed():
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    a = generate_synthetic_alerts(TOP_TICKERS, start, end, 5, random.Random(7))
    b = generate_synthetic_alerts(TOP_TICKERS, start, end, 5, random.Random(7))
    assert [(x.ticker, x.date, x.text) for x in a] == [(x.ticker, x.date, x.text) for x in b]


def test_generation_covers_every_ticker():
    end = date(2026, 10, 9)
    start = end - timedelta(days=365)
    alerts = generate_synthetic_alerts(TOP_TICKERS, start, end, 3, random.Random(1))
    seen = {a.ticker for a in alerts}
    assert seen == set(TOP_TICKERS)


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
