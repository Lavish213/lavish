# lavish_core/reporting/track_record.py
# Orders/fills/signals were being logged to DuckDB all along (HybridStore)
# but nothing ever read them back as a track record - the only way to know
# whether the bot was actually making money was to eyeball log files.
#
# Usage:
#   python -m lavish_core.reporting.track_record [--days 30] [--db path]
#
# Both paths are now fully recorded: options round trips (entry + the exit
# monitor's exit fill, always logged under the same order_id), and equity
# bracket round trips (entry + whichever take_profit/stop_loss leg fills,
# watched by equity_exit_monitor.py and logged under the same order_id).
# An entry that still shows up under "open entries" below means exactly
# that - no exit fill has been observed by this bot yet, not "untracked."
# The one case genuinely outside what this bot can see: a position closed
# by something other than its own bracket (manual intervention outside the
# bot, a day order expiring unfilled with the position closed some other
# way) - equity_exit_monitor.py reports that as "no_fill" in its own logs
# rather than fabricating an exit fill that never happened.
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB

_OCC_RE = re.compile(r"^([A-Z]+)\d{6}[CP]\d{8}$")


def _underlying(symbol: str) -> str:
    m = _OCC_RE.match(symbol.upper())
    return m.group(1) if m else symbol.upper()


def _is_option(symbol: str) -> bool:
    return bool(_OCC_RE.match(symbol.upper()))


def _load_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def build_report(store: HybridStore, days: int = 30) -> Dict[str, Any]:
    fills = store.fetchall(
        """
        SELECT order_id, symbol, side, qty, price, ts
        FROM fills
        WHERE ts > now() - INTERVAL '{}' DAY
        ORDER BY ts ASC
        """.format(int(days))
    )
    orders = store.fetchall(
        "SELECT id, meta, status FROM orders WHERE ts > now() - INTERVAL '{}' DAY".format(int(days))
    )
    order_meta: Dict[str, Dict[str, Any]] = {oid: _load_json(meta) for oid, meta, _status in orders}

    by_order: Dict[str, List[tuple]] = defaultdict(list)
    for order_id, symbol, side, qty, price, ts in fills:
        by_order[order_id].append((symbol, side, float(qty), float(price), ts))

    round_trips: List[Dict[str, Any]] = []
    open_entries: List[Dict[str, Any]] = []

    for order_id, rows in by_order.items():
        buys = [r for r in rows if r[1] == "buy"]
        sells = [r for r in rows if r[1] == "sell"]
        symbol = rows[0][0]
        meta = order_meta.get(order_id, {})
        source = meta.get("source", "unknown")
        expected_price = meta.get("expected_price")

        if not buys:
            continue  # sell-only fill with no matching buy in window - not enough info

        buy_qty = sum(r[2] for r in buys)
        buy_notional = sum(r[2] * r[3] for r in buys)
        buy_avg = buy_notional / buy_qty if buy_qty else 0.0
        slippage = round(buy_avg - float(expected_price), 4) if expected_price else None

        if not sells:
            open_entries.append({
                "order_id": order_id, "symbol": symbol, "source": source,
                "entry_price": round(buy_avg, 4), "qty": buy_qty, "slippage": slippage,
            })
            continue

        sell_qty = sum(r[2] for r in sells)
        sell_notional = sum(r[2] * r[3] for r in sells)
        sell_avg = sell_notional / sell_qty if sell_qty else 0.0
        multiplier = 100 if _is_option(symbol) else 1
        matched_qty = min(buy_qty, sell_qty)
        pnl = (sell_avg - buy_avg) * multiplier * matched_qty

        round_trips.append({
            "order_id": order_id, "symbol": symbol, "underlying": _underlying(symbol),
            "source": source, "entry_price": round(buy_avg, 4), "exit_price": round(sell_avg, 4),
            "qty": matched_qty, "pnl": round(pnl, 2), "slippage": slippage,
        })

    total_pnl = sum(t["pnl"] for t in round_trips)
    wins = [t for t in round_trips if t["pnl"] > 0]
    losses = [t for t in round_trips if t["pnl"] <= 0]
    win_rate = (len(wins) / len(round_trips)) if round_trips else None

    by_symbol: Dict[str, float] = defaultdict(float)
    by_source: Dict[str, float] = defaultdict(float)
    for t in round_trips:
        by_symbol[t["underlying"]] += t["pnl"]
        by_source[t["source"]] += t["pnl"]

    slippages = [t["slippage"] for t in round_trips if t["slippage"] is not None]
    avg_slippage = round(sum(slippages) / len(slippages), 4) if slippages else None

    return {
        "window_days": days,
        "closed_round_trips": len(round_trips),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(round_trips), 2) if round_trips else None,
        "avg_entry_slippage": avg_slippage,
        "pnl_by_underlying": dict(sorted(by_symbol.items(), key=lambda kv: -kv[1])),
        "pnl_by_source": dict(by_source),
        "open_entries_no_recorded_exit": len(open_entries),
        "open_entries": open_entries,
        "round_trips": round_trips,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Lavish_bot track record (last {report['window_days']} days) ===")
    print(f"Closed round trips: {report['closed_round_trips']} "
          f"(wins={report['wins']} losses={report['losses']} "
          f"win_rate={report['win_rate']})")
    print(f"Total P&L: ${report['total_pnl']:.2f}" if report['total_pnl'] is not None else "Total P&L: n/a")
    if report['avg_pnl_per_trade'] is not None:
        print(f"Avg P&L/trade: ${report['avg_pnl_per_trade']:.2f}")
    if report['avg_entry_slippage'] is not None:
        print(f"Avg entry slippage (actual - expected): ${report['avg_entry_slippage']:.4f}")
    print(f"\nP&L by underlying:")
    for sym, pnl in report["pnl_by_underlying"].items():
        print(f"  {sym:6s} ${pnl:.2f}")
    print(f"\nP&L by source:")
    for src, pnl in report["pnl_by_source"].items():
        print(f"  {src:12s} ${pnl:.2f}")
    print(f"\nOpen entries with no recorded exit fill: {report['open_entries_no_recorded_exit']}")
    print("(still genuinely open, or closed by something other than the bot's own bracket/exit "
          "monitor - see module docstring)")


def main() -> None:
    p = argparse.ArgumentParser(description="Lavish_bot track record / audit report")
    p.add_argument("--days", type=int, default=30, help="lookback window in days (default 30)")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB), help="path to the DuckDB file")
    p.add_argument("--json", action="store_true", help="print raw JSON instead of a formatted report")
    args = p.parse_args()

    store = HybridStore(duckdb_path=args.db)
    report = build_report(store, days=args.days)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
