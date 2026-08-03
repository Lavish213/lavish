# lavish_core/reporting/wsb_basket.py
# "Watch the top 10 WallStreetBets stocks, compare last year to now" - the
# original idea from early in this project, now built for real: takes a
# basket of tickers (default: 2025's most-cited retail/WSB favorites, per
# web research - see DEFAULT_BASKET below) and compares each one's price on
# a given start date against its most recent available price, using real
# daily history (yfinance, same source as ml/backtester.py - reused from
# there rather than duplicated here).
#
# Network note: this couldn't be run against live data from inside the
# coding sandbox this was built in (outbound Yahoo traffic is blocked by
# the sandbox's proxy - same limitation hit building the backtester). The
# comparison logic itself is fully tested against synthetic price data
# (see the commit message for what was verified) - this is ready to run
# for real once deployed somewhere with actual internet access.
#
# Usage:
#   python -m lavish_core.reporting.wsb_basket --start 2025-01-01
#   python -m lavish_core.reporting.wsb_basket --tickers NVDA,TSLA,PLTR --start 2025-06-01
from __future__ import annotations
import argparse, json
from datetime import date, datetime
from typing import Any, Dict, Optional

from ml.backtester import fetch_price_history

# Seeded from actual research (web search, Aug 2026) into 2025's most-cited
# WallStreetBets/retail-trader favorites - not a permanent list, just a
# reasonable default. The AI/mega-cap dip-buys (NVDA/TSLA/PLTR/AMZN/HOOD)
# were reported as broad winners for retail traders in 2025; the
# "D.O.R.K." names (DNUT/OPEN/RKT/KSS) were a July 2025 meme-stock spike -
# included deliberately alongside the mega-caps so the comparison actually
# shows the difference between "retail favorite that held up" and "meme
# spike that gave it back", not just a basket of winners.
DEFAULT_BASKET: Dict[str, str] = {
    "NVDA": "AI/mega-cap retail favorite",
    "TSLA": "AI/mega-cap retail favorite",
    "PLTR": "Long-term retail favorite",
    "AMZN": "Dip-buy favorite",
    "HOOD": "Retail favorite",
    "SOFI": "2025 2nd-half rally",
    "DNUT": "D.O.R.K. meme spike (Jul 2025)",
    "OPEN": "D.O.R.K. meme spike (Jul 2025)",
    "RKT": "D.O.R.K. meme spike (Jul 2025)",
    "KSS": "D.O.R.K. meme spike (Jul 2025)",
}


def compare_basket(
    tickers: Dict[str, str],
    start: date,
    end: Optional[date] = None,
    dollars_each: float = 1000.0,
) -> Dict[str, Any]:
    """
    For each ticker: finds the first available trading day on/after `start`
    for the "then" price, and the most recent available day (up to `end`,
    default today) for the "now" price. A ticker with no data in that
    window (bad symbol, delisted, network failure) is reported as
    no_data rather than silently dropped - same "admit don't know" posture
    as everywhere else parsing real market data in this codebase.
    """
    end = end or date.today()
    rows = []

    for ticker, note in tickers.items():
        hist = fetch_price_history(ticker, start, end)
        if hist is None or hist.empty:
            rows.append({"ticker": ticker, "note": note, "status": "no_data"})
            continue

        avail_from_start = hist.loc[hist.index >= start]
        if avail_from_start.empty:
            rows.append({"ticker": ticker, "note": note, "status": "no_data_after_start"})
            continue

        start_date_actual = avail_from_start.index[0]
        start_price = float(avail_from_start.iloc[0]["Close"])
        current_date_actual = hist.index[-1]
        current_price = float(hist.iloc[-1]["Close"])

        if start_price <= 0:
            rows.append({"ticker": ticker, "note": note, "status": "bad_start_price"})
            continue

        pct_change = (current_price - start_price) / start_price
        rows.append({
            "ticker": ticker, "note": note, "status": "ok",
            "start_date": str(start_date_actual), "start_price": round(start_price, 2),
            "current_date": str(current_date_actual), "current_price": round(current_price, 2),
            "pct_change": round(pct_change * 100, 2),
            "dollar_pnl": round(dollars_each * pct_change, 2),
        })

    ok_rows = [r for r in rows if r["status"] == "ok"]
    avg_pct = round(sum(r["pct_change"] for r in ok_rows) / len(ok_rows), 2) if ok_rows else None
    total_pnl = round(sum(r["dollar_pnl"] for r in ok_rows), 2) if ok_rows else None

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "dollars_each": dollars_each,
        "basket_size": len(tickers),
        "resolved": len(ok_rows),
        "basket_avg_pct_change": avg_pct,
        "total_pnl_equal_weighted": total_pnl,
        "rows": rows,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Basket comparison: {report['start']} -> {report['end']} "
          f"(${report['dollars_each']:.0f} each) ===")
    for r in report["rows"]:
        if r["status"] != "ok":
            print(f"  {r['ticker']:6s} {r['note']:32s} SKIPPED ({r['status']})")
            continue
        pct_str = f"{r['pct_change']:+.2f}%"
        pnl_str = f"-${abs(r['dollar_pnl']):.2f}" if r["dollar_pnl"] < 0 else f"+${r['dollar_pnl']:.2f}"
        print(f"  {r['ticker']:6s} {r['note']:32s} "
              f"{r['start_date']} @ {r['start_price']:>9.2f}  ->  "
              f"{r['current_date']} @ {r['current_price']:>9.2f}   "
              f"{pct_str}  ({pnl_str})")
    print(f"\nResolved {report['resolved']}/{report['basket_size']} tickers.")
    if report["basket_avg_pct_change"] is not None:
        total = report["total_pnl_equal_weighted"]
        total_str = f"-${abs(total):.2f}" if total < 0 else f"+${total:.2f}"
        print(f"Basket average: {report['basket_avg_pct_change']:+.2f}%")
        print(f"Equal-weighted P&L (${report['dollars_each']:.0f} into each resolved ticker): {total_str}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare a basket of stocks from a start date to now")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD - e.g. --start 2025-01-01")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default today")
    p.add_argument("--tickers", type=str, default=None,
                    help="comma-separated tickers to override the default WSB-favorites basket")
    p.add_argument("--dollars-each", type=float, default=1000.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date() if args.end else None
    tickers = ({t.strip().upper(): "custom" for t in args.tickers.split(",") if t.strip()}
               if args.tickers else DEFAULT_BASKET)

    report = compare_basket(tickers, start=start, end=end, dollars_each=args.dollars_each)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
