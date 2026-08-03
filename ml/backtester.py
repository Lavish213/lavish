# ml/backtester.py
# This file existed but was completely empty - referenced nowhere, wired to
# nothing. There was no way to answer "would this bot's rules actually have
# made money on past alerts" without live-trading it and waiting.
#
# Equity only. Options backtesting needs historical options-chain data
# (bid/ask per strike/expiry/date), which has no free, reliable source -
# yfinance's options data is snapshot-only (current chain, not historical).
# Simulating options P&L from just the underlying's historical price would
# mean inventing an implied-vol/greeks model good enough to trust real
# money on, which is a materially different and much larger undertaking
# than this pass covers. Said plainly rather than faked with a rough
# approximation dressed up as a real backtest.
#
# Two ways to get signals in:
#   --from-db        replay real captured equity signals from the `signals`
#                     table (what actually got alerted, i.e. does the bot's
#                     rule set on real data work)
#   --csv path.csv    manually specified test cases (symbol,side,date[,target,stop])
#                     for testing hypotheticals independent of alert history
#
# Simulation: walk forward day-by-day from entry using yfinance daily OHLC.
# A day where BOTH the stop and target could have been hit (High >= target
# AND Low <= stop on the same day) resolves to the stop - conservative by
# design, since intraday sequencing isn't knowable from daily bars alone.
# No target/stop given -> falls back to this bot's own default guardrail
# percentages (DEFAULT_STOP_LOSS_PCT/DEFAULT_TAKE_PROFIT_PCT), same as what
# a live equity trade with no stated levels would actually run under.
#
# Usage:
#   python -m ml.backtester --from-db --days 90
#   python -m ml.backtester --csv signals.csv --max-hold-days 10
from __future__ import annotations
import argparse, csv, json, logging, os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger("backtester")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_STOP_LOSS_PCT = float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.015"))
DEFAULT_TAKE_PROFIT_PCT = float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.02"))
DEFAULT_MAX_HOLD_DAYS = int(os.getenv("BACKTEST_MAX_HOLD_DAYS", "10"))


@dataclass
class BacktestSignal:
    symbol: str
    entry_date: date
    target: Optional[float] = None
    stop: Optional[float] = None
    source: str = "unknown"


@dataclass
class TradeResult:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str]
    exit_price: Optional[float]
    exit_reason: str  # "target" | "stop" | "max_hold" | "no_data"
    qty: float
    pnl: Optional[float]
    source: str


def fetch_price_history(symbol: str, start: date, end: date):
    """
    Daily OHLC via yfinance - unauthenticated, already a dependency (also
    used as broker_alpaca's data-outage fallback). Returns None (never
    raises) on any failure so a bad symbol or a network hiccup skips that
    one signal instead of aborting the whole backtest run.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if df is None or df.empty:
            return None
        df.index = df.index.date
        return df
    except Exception as e:
        log.warning("fetch_price_history failed for %s: %s", symbol, e)
        return None


def simulate_trade(
    price_history,
    entry_date: date,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    entry_price: Optional[float] = None,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> Dict[str, Any]:
    """
    Long-only walk-forward simulation over a daily OHLC DataFrame (index =
    date, columns Open/High/Low/Close - the yfinance .history() shape).
    Entry fills at entry_date's Open (or an explicit override). From the
    entry day onward, checks each day's High/Low against target/stop; if
    neither ever triggers within max_hold_days trading rows, exits at the
    last available day's Close.
    """
    if entry_date not in price_history.index:
        return {"exit_date": None, "exit_price": None, "exit_reason": "no_data", "entry_price": None}

    rows = price_history.loc[entry_date:]
    if rows.empty:
        return {"exit_date": None, "exit_price": None, "exit_reason": "no_data", "entry_price": None}

    entry_row = rows.iloc[0]
    fill_price = entry_price if entry_price is not None else float(entry_row["Open"])
    target_px = target if target is not None else round(fill_price * (1 + DEFAULT_TAKE_PROFIT_PCT), 2)
    stop_px = stop if stop is not None else round(fill_price * (1 - DEFAULT_STOP_LOSS_PCT), 2)

    window = rows.iloc[: max_hold_days + 1]
    for d, row in window.iloc[1:].iterrows():  # first row is entry day itself - skip re-checking it
        hit_stop = float(row["Low"]) <= stop_px
        hit_target = float(row["High"]) >= target_px
        if hit_stop:  # conservative: stop wins on an ambiguous same-day double-hit
            return {"exit_date": str(d), "exit_price": stop_px, "exit_reason": "stop", "entry_price": fill_price}
        if hit_target:
            return {"exit_date": str(d), "exit_price": target_px, "exit_reason": "target", "entry_price": fill_price}

    last = window.iloc[-1]
    return {"exit_date": str(window.index[-1]), "exit_price": float(last["Close"]),
            "exit_reason": "max_hold", "entry_price": fill_price}


def run_backtest(signals: List[BacktestSignal], max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
                  qty: float = 1.0) -> Dict[str, Any]:
    results: List[TradeResult] = []
    skipped = 0

    for sig in signals:
        history = fetch_price_history(sig.symbol, sig.entry_date, sig.entry_date + timedelta(days=max_hold_days + 5))
        if history is None:
            skipped += 1
            log.info("Skip %s @ %s: no price history available.", sig.symbol, sig.entry_date)
            continue

        sim = simulate_trade(history, sig.entry_date, target=sig.target, stop=sig.stop, max_hold_days=max_hold_days)
        if sim["exit_reason"] == "no_data":
            skipped += 1
            log.info("Skip %s @ %s: entry date not in price history (holiday/weekend/delisted?).",
                      sig.symbol, sig.entry_date)
            continue

        pnl = (sim["exit_price"] - sim["entry_price"]) * qty if sim["exit_price"] is not None else None
        results.append(TradeResult(
            symbol=sig.symbol, entry_date=sig.entry_date.isoformat(), entry_price=sim["entry_price"],
            exit_date=sim["exit_date"], exit_price=sim["exit_price"], exit_reason=sim["exit_reason"],
            qty=qty, pnl=round(pnl, 2) if pnl is not None else None, source=sig.source,
        ))

    total_pnl = sum(r.pnl for r in results if r.pnl is not None)
    wins = [r for r in results if r.pnl is not None and r.pnl > 0]
    win_rate = (len(wins) / len(results)) if results else None

    return {
        "simulated_trades": len(results),
        "skipped_no_data": skipped,
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(results), 2) if results else None,
        "trades": [asdict(r) for r in results],
    }


def load_signals_from_csv(path: str) -> List[BacktestSignal]:
    out: List[BacktestSignal] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            side = (row.get("side") or "buy").strip().lower()
            if side not in ("buy", "call", "long"):
                continue  # backtesting a fresh entry only makes sense for opens, not exits/shorts
            out.append(BacktestSignal(
                symbol=row["symbol"].strip().upper(),
                entry_date=datetime.fromisoformat(row["date"].strip()).date(),
                target=float(row["target"]) if row.get("target") else None,
                stop=float(row["stop"]) if row.get("stop") else None,
                source="csv",
            ))
    return out


def load_signals_from_db(days: int = 90) -> List[BacktestSignal]:
    from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB
    store = HybridStore(duckdb_path=str(DEFAULT_DB))
    rows = store.fetchall(
        "SELECT symbol, side, source, ts, payload FROM signals "
        "WHERE side='buy' AND ts > now() - INTERVAL '{}' DAY".format(int(days))
    )
    out: List[BacktestSignal] = []
    for symbol, side, source, ts, payload_raw in rows:
        try:
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
        except Exception:
            payload = {}
        if payload.get("strike"):
            continue  # options-shaped signal - not backtestable here, see module docstring
        entry_date = ts.date() if hasattr(ts, "date") else datetime.fromisoformat(str(ts)).date()
        out.append(BacktestSignal(symbol=symbol, entry_date=entry_date, source=source))
    return out


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Backtest results ===")
    print(f"Simulated trades: {report['simulated_trades']} (skipped, no data: {report['skipped_no_data']})")
    print(f"Win rate: {report['win_rate']}")
    print(f"Total P&L: ${report['total_pnl']:.2f}" if report['total_pnl'] is not None else "Total P&L: n/a")
    if report['avg_pnl_per_trade'] is not None:
        print(f"Avg P&L/trade: ${report['avg_pnl_per_trade']:.2f}")
    for t in report["trades"]:
        print(f"  {t['symbol']:6s} {t['entry_date']} @ {t['entry_price']:.2f} -> "
              f"{t['exit_date']} @ {t['exit_price']:.2f} ({t['exit_reason']}) pnl={t['pnl']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Lavish_bot equity backtester")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true", help="replay real captured equity signals from DuckDB")
    src.add_argument("--csv", type=str, help="path to a CSV of symbol,side,date[,target,stop]")
    p.add_argument("--days", type=int, default=90, help="lookback window for --from-db (default 90)")
    p.add_argument("--max-hold-days", type=int, default=DEFAULT_MAX_HOLD_DAYS)
    p.add_argument("--qty", type=float, default=1.0, help="shares per simulated trade")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    signals = load_signals_from_db(days=args.days) if args.from_db else load_signals_from_csv(args.csv)
    if not signals:
        print("No signals to backtest.")
        return

    report = run_backtest(signals, max_hold_days=args.max_hold_days, qty=args.qty)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
