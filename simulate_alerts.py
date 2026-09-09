#!/usr/bin/env python3
# simulate_alerts.py
# "Simulate and test hella" - a batch of realistic sample alerts run
# through the REAL parsing + execution pipeline, not mocks. Built because
# this sandbox's network proxy blocks paper-api.alpaca.markets entirely
# (confirmed by direct test) - real-API testing can't happen from inside
# that chat session regardless of credentials, so this is meant to be run
# on your own machine (or wherever ends up hosting the bot), with your
# own real Alpaca paper keys in a local .env that never goes through chat.
#
# Two modes:
#   python3 simulate_alerts.py             - dry run (default): parses
#     every sample alert and prints what WOULD happen. No network calls,
#     no orders, safe to run with no .env at all.
#   python3 simulate_alerts.py --live      - actually runs each alert
#     through handle_alert_text() for real. If TRADE_MODE=paper and real
#     Alpaca paper keys are in .env, this submits REAL paper orders you
#     can watch land in your Alpaca dashboard. Pauses between each (see
#     --pause) so they're easy to follow one at a time instead of a wall
#     of simultaneous orders.
#
# The sample alerts deliberately include the real phrasing bugs found and
# fixed by testing actual screenshots this session (side-before-price
# strikes, casual "grabbing calls" phrasing) plus the cases that should
# be correctly SKIPPED (perp/leveraged-token alerts, ambiguous direction,
# an off-whitelist ticker) - so a clean run here is a real end-to-end
# confirmation the pipeline still behaves as designed, not just that it
# doesn't crash.
from __future__ import annotations
import argparse, os, sys, time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _next_friday(from_date: date | None = None) -> date:
    d = from_date or date.today()
    days_ahead = (4 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # if today IS Friday, use next week's for real runway instead of 0DTE
    return d + timedelta(days=days_ahead)


def build_sample_alerts() -> list[dict]:
    fri = _next_friday()
    fri_str = f"{fri.month}/{fri.day}"
    return [
        {
            "label": "Clean options call, price-first (standard phrasing)",
            "text": f"AAPL $190 calls target 200 stop 180 exp {fri_str}",
        },
        {
            "label": "Clean options put with explicit stop-loss phrasing",
            "text": f"SPY $528 Put {fri_str} stop loss $528.20",
        },
        {
            "label": 'Side-before-price phrasing ("Calls . $X" - real bug fixed this session)',
            "text": f"Spy Calls . $530 {fri_str}",
        },
        {
            "label": "Casual equity phrasing, no strike (should route to the equity path)",
            "text": "BUY NVDA now, breaking out",
        },
        {
            "label": "Ambiguous - no clear buy/sell direction (should be SKIPPED)",
            "text": "SPY looking spicy today, watch this level",
        },
        {
            "label": "Perp/leveraged-token alert (should be SKIPPED with a warning, not mis-copied as equity)",
            "text": "MU/USDC-P long here, 10x leverage, big move incoming",
        },
        {
            "label": "Unknown ticker not on the whitelist (should be SKIPPED)",
            "text": "XYZ 50 calls target 55 stop 45",
        },
        {
            "label": "Recently-added whitelist ticker (DDOG - confirmed real this session)",
            "text": f"DDOG 130 calls target 140 stop 120 exp {fri_str}",
        },
    ]


def dry_run(alerts: list[dict]) -> None:
    from lavish_core.vision.extract_signal import parse_text
    from lavish_core.trading.alert_handler import _is_perp_or_leveraged, _equity_action_from_text

    print(f"=== DRY RUN: parsing {len(alerts)} sample alerts (no network, no orders) ===\n")
    for i, a in enumerate(alerts, 1):
        print(f"[{i}/{len(alerts)}] {a['label']}")
        print(f"    text: {a['text']!r}")
        if _is_perp_or_leveraged(a["text"]):
            print("    -> SKIPPED: detected as a perp/leveraged-token alert")
            print()
            continue
        r = parse_text(a["text"])
        ticker = r.get("ticker")
        if not ticker:
            print("    -> SKIPPED: no whitelisted ticker found")
            print()
            continue
        if r.get("side") in ("CALL", "PUT") and r.get("strike") and r.get("expiry"):
            print(f"    -> WOULD TRADE (options): {ticker} {r['side']} ${r['strike']} exp {r['expiry']} "
                  f"target={r.get('target_hint')} stop={r.get('stop_hint')}")
        else:
            action = _equity_action_from_text(a["text"])
            if action:
                print(f"    -> WOULD TRADE (equity): {action} {ticker}")
            else:
                print(f"    -> SKIPPED: ticker {ticker} found but no clear buy/sell direction")
        print()
    print("Dry run complete - nothing was sent anywhere. Re-run with --live (and a real .env with "
          "TRADE_MODE=paper + real Alpaca paper keys) to actually submit paper orders.")


def live_run(alerts: list[dict], pause: float) -> None:
    mode = os.getenv("TRADE_MODE", "dry")
    if mode not in ("paper", "live"):
        print(f"TRADE_MODE={mode!r} in .env - nothing will actually submit. Set TRADE_MODE=paper for this to do anything.")
        return
    if mode == "live":
        confirm = input("TRADE_MODE=live - this will submit REAL orders with REAL money. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    from lavish_core.trading.alert_handler import handle_alert_text

    print(f"=== LIVE RUN ({mode}): submitting {len(alerts)} simulated alerts to the real pipeline ===")
    print(f"Watch your Alpaca dashboard - {pause}s pause between each so they're easy to follow.\n")
    for i, a in enumerate(alerts, 1):
        print(f"[{i}/{len(alerts)}] {a['label']}")
        print(f"    text: {a['text']!r}")
        handle_alert_text(text=a["text"], note=f"simulate_alerts:{i}", source="simulate-alerts")
        print()
        if i < len(alerts):
            time.sleep(pause)
    print("Done. Check your Alpaca paper dashboard and logs/lavish.log for what actually happened.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="Actually submit real (paper) orders instead of just parsing. Requires TRADE_MODE=paper "
                        "and real Alpaca keys in .env.")
    p.add_argument("--pause", type=float, default=8.0,
                   help="Seconds between each --live alert, so you can watch it land in Alpaca before the next fires.")
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")

    alerts = build_sample_alerts()
    if args.live:
        live_run(alerts, args.pause)
    else:
        dry_run(alerts)


if __name__ == "__main__":
    main()
