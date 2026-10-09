#!/usr/bin/env python3
# simulate_ticker_watch.py
#
# Answers "if the bot watched alerts across SPY, NVDA and the other most-
# traded mega-cap tickers over a year, how would it behave and what are
# the stats" - NOT a replay of her real alerts (no dated history of those
# exists anywhere this project has found - see AUDIT_LOG.md/the project
# history). This generates a large, clearly-synthetic alert stream across
# a fixed basket of real, heavily-traded tickers, spread over a simulated
# year, and runs every one through the REAL parser and REAL routing logic
# (same functions the live bot uses, same confidence floor) to get honest
# mechanism stats: how many would actually trade, by what path, and - for
# the ones that would - what real market data says would have happened.
#
# Three layers of honesty here, stated plainly rather than blurred:
#   1. The ALERT TEXT and TIMING are synthetic/invented - not her real
#      calls, not real timestamps. This tests the pipeline's MECHANISM
#      (parsing, routing, confidence floor, guardrail triggering), not
#      her real track record.
#   2. Equity P&L uses REAL historical daily prices (ml/backtester.py's
#      walk-forward simulator, already-tested) for the tickers below -
#      genuine market data, synthetic entry timing.
#   3. Options P&L has no free historical options-chain data anywhere
#      (ml/backtester.py's own docstring already establishes this) - uses
#      Black-Scholes via lavish_core/reporting/option_estimator.py, a
#      THEORETICAL estimate using realized volatility as an implied-vol
#      stand-in, not a real fill. Labeled as such in the output.
#
# This sandbox has no network access to Yahoo Finance (confirmed blocked
# by the egress proxy - policy denial, not transient). Classification
# stats (parsing/routing/skip-rate) work fully offline and are real right
# now. Price-dependent P&L numbers will show "no_data"/None here and need
# to be run on a machine with real network access for real numbers.
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Quiets yfinance's per-call "cookie/crumb fetch failed" noise (harmless,
# cosmetic retries) so a run with many signals stays readable - real
# failures still surface via ml/backtester.py's own WARNING-level log.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lavish_core.vision.extract_signal import parse_text
from lavish_core.trading.alert_handler import _is_perp_or_leveraged, _equity_action_from_text
from lavish_core.trading.trade_handler import CONF_FLOOR
from ml.backtester import BacktestSignal, run_backtest, print_report, fetch_price_history
from lavish_core.reporting.option_estimator import estimate_option_pnl, print_estimate

# The real top-10 most-traded/most-watched mega-cap + index tickers - all
# already on the live WHITELIST_TICKERS (env.example), not invented names.
TOP_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL"]

# Used only when real price history can't be fetched (e.g. this sandbox,
# confirmed network-blocked from Yahoo Finance) - keeps synthetic strikes
# in a plausible ballpark instead of crashing or falling back to a flat
# $50-600 random range (the bug this table fixes: that range produced
# strikes wildly unrelated to a ticker's real price - e.g. a $599 call on
# AAPL trading near $250 - which collapsed every Black-Scholes estimate
# to a meaningless -100%). Rough centers as of this project, not exact;
# real price history (fetched once per ticker below) always wins when
# network access allows it.
FALLBACK_REFERENCE_PRICES = {
    "SPY": 670, "QQQ": 610, "AAPL": 250, "MSFT": 480, "NVDA": 180,
    "TSLA": 420, "AMD": 220, "META": 650, "AMZN": 240, "GOOGL": 280,
}

EQUITY_BUY_TEMPLATES = [
    "BUY {t} now, breaking out",
    "{t} looking strong here, grabbing shares",
    "adding {t} on this pullback",
    "{t} long here, momentum building",
]
EQUITY_SELL_TEMPLATES = [
    "selling {t} here, taking profit",
    "SELL {t} now, locking it in",
    "closing {t}, done for today",
    "{t} trimming the position here",
]
OPTIONS_TEMPLATES = [
    "{t} ${strike} {side_word} target {target} stop {stop} exp {exp}",
    "{t} {side_word} . ${strike} {exp}",
    "grabbing {t} {strike}{side_letter} here",
]
AMBIGUOUS_TEMPLATES = [
    "{t} looking spicy today, watch this level",
    "keeping an eye on {t} into the close",
    "{t} - interesting setup forming, not in yet",
]


def _next_weekday_friday(d: date) -> date:
    days_ahead = (4 - d.weekday()) % 7
    return d + timedelta(days=days_ahead or 7)


@dataclass
class SyntheticAlert:
    ticker: str
    date: date
    text: str
    designed_kind: str  # "equity_buy" | "equity_sell" | "options" | "ambiguous"


def _build_price_lookups(tickers: list[str], start: date, end: date) -> dict:
    """One fetch per ticker, reused for every alert on that ticker - not
    one fetch per alert. Returns {ticker: price_history_df_or_None}."""
    out = {}
    for t in tickers:
        out[t] = fetch_price_history(t, start, end)
    return out


def _reference_price(ticker: str, d: date, price_lookups: dict) -> float:
    """Real Close nearest-at-or-before d if we have history for this
    ticker; otherwise the static fallback. Never raises, never returns a
    strike-breaking None - that's the whole point of this helper."""
    hist = price_lookups.get(ticker)
    if hist is not None:
        rows = hist.loc[hist.index <= d]
        if not rows.empty:
            return float(rows.iloc[-1]["Close"])
    return float(FALLBACK_REFERENCE_PRICES.get(ticker, 200))


def generate_synthetic_alerts(tickers: list[str], start: date, end: date,
                               alerts_per_ticker: int, rng: random.Random,
                               price_lookups: Optional[dict] = None) -> list[SyntheticAlert]:
    if price_lookups is None:
        price_lookups = _build_price_lookups(tickers, start, end)
    out: list[SyntheticAlert] = []
    span_days = (end - start).days
    for t in tickers:
        for _ in range(alerts_per_ticker):
            d = start + timedelta(days=rng.randint(0, span_days))
            kind = rng.choices(
                ["equity_buy", "equity_sell", "options", "ambiguous"],
                weights=[0.30, 0.15, 0.35, 0.20],
            )[0]
            if kind == "equity_buy":
                text = rng.choice(EQUITY_BUY_TEMPLATES).format(t=t)
            elif kind == "equity_sell":
                text = rng.choice(EQUITY_SELL_TEMPLATES).format(t=t)
            elif kind == "options":
                # Strike anchored to the real (or fallback) reference
                # price at this date, within +/-10% - a near-the-money
                # weekly call/put, the shape her real alerts actually
                # take, not a flat $50-600 draw unrelated to the ticker.
                ref_price = _reference_price(t, d, price_lookups)
                strike = round(ref_price * rng.uniform(0.90, 1.10) / 5) * 5
                strike = max(5, strike)
                side_word = rng.choice(["calls", "puts"])
                side_letter = "c" if side_word == "calls" else "p"
                exp = _next_weekday_friday(d)
                target = round(strike * rng.uniform(1.1, 1.3), 0)
                stop = round(strike * rng.uniform(0.75, 0.9), 0)
                text = rng.choice(OPTIONS_TEMPLATES).format(
                    t=t, strike=int(strike), side_word=side_word, side_letter=side_letter,
                    target=int(target), stop=int(stop), exp=f"{exp.month}/{exp.day}",
                )
            else:
                text = rng.choice(AMBIGUOUS_TEMPLATES).format(t=t)
            out.append(SyntheticAlert(ticker=t, date=d, text=text, designed_kind=kind))
    out.sort(key=lambda a: a.date)
    return out


def classify_and_route(alert: SyntheticAlert) -> dict:
    """Mirrors the real pipeline's decision path (alert_handler + trade_handler's
    confidence floor) - same functions the live bot calls, not a reimplementation."""
    if _is_perp_or_leveraged(alert.text):
        return {"outcome": "skip:perp"}
    r = parse_text(alert.text)
    ticker = r.get("ticker")
    conf = r.get("confidence", 0.0)
    if not ticker:
        return {"outcome": "skip:no_ticker_or_not_whitelisted"}
    if r.get("side") in ("CALL", "PUT") and r.get("strike") and r.get("expiry"):
        if conf < CONF_FLOOR:
            return {"outcome": "skip:below_confidence_floor"}
        return {"outcome": "trade:options", "ticker": ticker, "side": r["side"],
                "strike": r["strike"], "expiry": r["expiry"], "target": r.get("target_hint"),
                "stop": r.get("stop_hint"), "confidence": conf}
    action = _equity_action_from_text(alert.text)
    if action:
        if conf < CONF_FLOOR:
            return {"outcome": "skip:below_confidence_floor"}
        return {"outcome": "trade:equity", "ticker": ticker, "action": action, "confidence": conf}
    return {"outcome": "skip:ticker_found_no_clear_direction"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=365, help="simulated window size, trailing from today")
    p.add_argument("--alerts-per-ticker", type=int, default=25)
    p.add_argument("--seed", type=int, default=42, help="reproducible synthetic alert generation")
    p.add_argument("--max-hold-days", type=int, default=10, help="equity walk-forward cap, same default as ml/backtester.py")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    end = date.today()
    start = end - timedelta(days=args.days)

    price_lookups = _build_price_lookups(TOP_TICKERS, start, end)
    real_price_tickers = [t for t in TOP_TICKERS if price_lookups.get(t) is not None]
    if real_price_tickers:
        print(f"Strike anchoring: real price history available for {len(real_price_tickers)}/{len(TOP_TICKERS)} "
              f"tickers ({', '.join(real_price_tickers)}).")
    missing = [t for t in TOP_TICKERS if t not in real_price_tickers]
    if missing:
        print(f"Strike anchoring: NO price history for {', '.join(missing)} - using static fallback reference "
              f"prices for strike generation (no network access, or fetch failed).")
    print()

    alerts = generate_synthetic_alerts(TOP_TICKERS, start, end, args.alerts_per_ticker, rng,
                                        price_lookups=price_lookups)
    print(f"=== Generated {len(alerts)} synthetic alerts across {len(TOP_TICKERS)} tickers "
          f"({start} to {end}) ===\n")

    results = [classify_and_route(a) for a in alerts]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    print("--- Classification (real parser + real routing + real confidence floor) ---")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {k}: {counts[k]} ({100*counts[k]/len(alerts):.1f}%)")

    equity_trades = [r for r in results if r["outcome"] == "trade:equity"]
    options_trades = [r for r in results if r["outcome"] == "trade:options"]
    print(f"\nTotal would-trade: {len(equity_trades) + len(options_trades)} / {len(alerts)} "
          f"({100*(len(equity_trades)+len(options_trades))/len(alerts):.1f}%)")

    # --- Bug-exposure diagnostic: a naive long-only ledger of this run's
    # own BUY/CALL signals, flagging any SELL with nothing open to close -
    # exactly the shape of the confirmed, still-unfixed accidental-short
    # bug in trade_handler.py (qty computed from dollars, never checked
    # against held position).
    open_qty: dict[str, int] = {}
    accidental_short_risk = 0
    for a, r in zip(alerts, results):
        if r["outcome"] == "trade:equity":
            if r["action"] == "BUY":
                open_qty[r["ticker"]] = open_qty.get(r["ticker"], 0) + 1
            elif r["action"] == "SELL":
                if open_qty.get(r["ticker"], 0) <= 0:
                    accidental_short_risk += 1
                else:
                    open_qty[r["ticker"]] -= 1
    print(f"\n--- Bug-exposure diagnostic (current unfixed trade_handler.py) ---")
    print(f"SELL signals with no matching open position in this run: {accidental_short_risk} "
          f"/ {len(equity_trades)} equity trades")
    print("(Each one would risk an accidental short under the current code - confirmed, unfixed bug.)")

    # --- Real equity P&L via the existing, already-tested backtester ---
    equity_signals = []
    for a, r in zip(alerts, results):
        if r["outcome"] == "trade:equity" and r["action"] == "BUY":
            equity_signals.append(BacktestSignal(symbol=r["ticker"], entry_date=a.date, source="synthetic"))
    print(f"\n--- Equity backtest ({len(equity_signals)} BUY signals, real historical daily prices) ---")
    if equity_signals:
        report = run_backtest(equity_signals, max_hold_days=args.max_hold_days)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print_report(report)
        if report["simulated_trades"] == 0:
            print("\n(All trades skipped for no_data - this sandbox has no network access to Yahoo Finance, "
                  "confirmed blocked. Run this on a machine with real network access for real numbers.)")
    else:
        print("No BUY-side equity signals generated this run.")

    # --- Theoretical options P&L (Black-Scholes, clearly not a real fill) ---
    print(f"\n--- Options estimates ({len(options_trades)} signals, THEORETICAL Black-Scholes, "
          f"not real fills - see option_estimator.py) ---")
    shown = 0
    for a, r in zip(alerts, results):
        if r["outcome"] != "trade:options" or shown >= 5:
            continue
        try:
            expiry = date.fromisoformat(r["expiry"])
            dte = max(1, (expiry - a.date).days)
            est = estimate_option_pnl(
                ticker=r["ticker"], dte_days=dte, entry_date=a.date,
                strike=r["strike"], option_type=r["side"].lower(),
            )
            if est:
                print_estimate(est)
                shown += 1
        except Exception as e:
            print(f"  (skipped {r['ticker']}: {e})")
    if options_trades and shown == 0:
        print("(No estimates produced - this sandbox has no network access for the underlying's price history.)")
    elif len(options_trades) > shown:
        print(f"  ... and {len(options_trades) - shown} more options signals not shown (sample capped at 5).")


if __name__ == "__main__":
    main()
