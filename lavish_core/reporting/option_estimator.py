# lavish_core/reporting/option_estimator.py
# "If I show you a stock, how much would I have made buying an option on
# it" - real historical options premium data doesn't exist anywhere for
# free (the same wall hit building ml/backtester.py, which is why that
# one only does equity). This doesn't get around that wall - it estimates
# instead, using Black-Scholes with the underlying's own real historical
# volatility as a stand-in for implied volatility.
#
# Said plainly, not glossed over: this is a THEORETICAL estimate, not a
# reconstruction of what the contract actually would have cost. Real
# market premiums are usually priced above simple historical/realized
# volatility - implied vol carries a risk premium, and spikes around
# earnings/news in ways trailing price history doesn't capture. Treat
# the output as "roughly what a fairly-priced contract might have done,"
# not "what your broker would have actually filled."
#
# Usage:
#   python -m lavish_core.reporting.option_estimator --ticker SBUX --dte 21 --type call
#   python -m lavish_core.reporting.option_estimator --ticker SBUX --entry-date 2026-07-14 --exit-date 2026-08-04 --dte 21 --strike 80 --type call
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from ml.backtester import fetch_price_history

DEFAULT_RISK_FREE_RATE = 0.05  # rough current short-term rate; T-bill yield is the standard proxy
DEFAULT_VOL_WINDOW_DAYS = 30   # trailing window for realized volatility


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf - no scipy dependency needed for one function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Standard Black-Scholes European option price. T in years, sigma
    annualized. T<=0 or sigma<=0 collapses to intrinsic value (an
    expired/zero-vol option has no time value left to price).
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def historical_volatility(price_history, as_of: date, window_days: int = DEFAULT_VOL_WINDOW_DAYS) -> Optional[float]:
    """
    Annualized realized volatility from daily log returns over the
    `window_days` trading days immediately before (and including)
    `as_of`. Standard proxy for implied volatility when no real options
    quote exists - see module docstring for why this is an
    approximation, not the real thing.
    """
    hist = price_history.loc[price_history.index <= as_of]
    if len(hist) < 2:
        return None
    closes = hist["Close"].tail(window_days + 1).tolist()
    if len(closes) < 2:
        return None
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252)  # annualize (252 trading days/year)


@dataclass
class OptionEstimate:
    ticker: str
    option_type: str
    strike: float
    entry_date: str
    entry_underlying_price: float
    expiry_date: str
    exit_date: str
    exit_underlying_price: float
    volatility_used_pct: float
    entry_premium_est: float
    exit_premium_est: float
    pnl_per_contract_est: float
    pnl_pct_est: float


def estimate_option_pnl(
    ticker: str,
    dte_days: int,
    entry_date: Optional[date] = None,
    exit_date: Optional[date] = None,
    strike: Optional[float] = None,
    option_type: str = "call",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    vol_window_days: int = DEFAULT_VOL_WINDOW_DAYS,
) -> Optional[OptionEstimate]:
    """
    entry_date defaults to today. exit_date defaults to entry_date + dte_days
    (i.e. held to expiry) capped at the latest available price data. strike
    defaults to at-the-money (the entry underlying price itself).
    """
    entry_date = entry_date or date.today()
    expiry_date = entry_date + timedelta(days=dte_days)
    exit_date = exit_date or expiry_date

    # Fetch a window wide enough to cover the vol lookback before entry
    # through the exit date.
    fetch_start = entry_date - timedelta(days=vol_window_days * 2)  # *2: calendar days to cover enough trading days
    fetch_end = max(expiry_date, exit_date) + timedelta(days=3)
    hist = fetch_price_history(ticker, fetch_start, fetch_end)
    if hist is None or hist.empty:
        return None

    entry_rows = hist.loc[hist.index >= entry_date]
    if entry_rows.empty:
        return None
    entry_actual_date = entry_rows.index[0]
    entry_price = float(entry_rows.iloc[0]["Close"])

    exit_rows = hist.loc[hist.index >= exit_date]
    if not exit_rows.empty:
        exit_actual_date = exit_rows.index[0]
        exit_price = float(exit_rows.iloc[0]["Close"])
    else:
        # exit_date beyond available data - use the last available row
        exit_actual_date = hist.index[-1]
        exit_price = float(hist.iloc[-1]["Close"])

    sigma = historical_volatility(hist, as_of=entry_actual_date, window_days=vol_window_days)
    if sigma is None:
        return None

    k = strike if strike is not None else round(entry_price)

    t_entry_years = dte_days / 365.0
    t_exit_years = max(0.0, (expiry_date - exit_actual_date).days / 365.0)

    entry_premium = black_scholes_price(entry_price, k, t_entry_years, risk_free_rate, sigma, option_type)
    exit_premium = black_scholes_price(exit_price, k, t_exit_years, risk_free_rate, sigma, option_type)

    pnl_per_contract = (exit_premium - entry_premium) * 100
    pnl_pct = ((exit_premium - entry_premium) / entry_premium * 100) if entry_premium > 0 else None

    return OptionEstimate(
        ticker=ticker.upper(),
        option_type=option_type,
        strike=round(k, 2),
        entry_date=str(entry_actual_date),
        entry_underlying_price=round(entry_price, 2),
        expiry_date=expiry_date.isoformat(),
        exit_date=str(exit_actual_date),
        exit_underlying_price=round(exit_price, 2),
        volatility_used_pct=round(sigma * 100, 1),
        entry_premium_est=round(entry_premium, 2),
        exit_premium_est=round(exit_premium, 2),
        pnl_per_contract_est=round(pnl_per_contract, 2),
        pnl_pct_est=round(pnl_pct, 1) if pnl_pct is not None else None,
    )


def print_estimate(est: OptionEstimate) -> None:
    print(f"\n=== Theoretical option estimate: {est.ticker} ${est.strike} {est.option_type.upper()} "
          f"exp {est.expiry_date} ===")
    print("(Black-Scholes using historical/realized volatility as an implied-vol stand-in - "
          "NOT a real quoted market premium. Real premiums usually run higher, especially "
          "around earnings/news. See module docstring.)")
    print(f"Volatility used: {est.volatility_used_pct}% (annualized, {DEFAULT_VOL_WINDOW_DAYS}-day trailing)")
    print(f"Entry: {est.entry_date} - underlying ${est.entry_underlying_price}, "
          f"est. premium ${est.entry_premium_est} (${est.entry_premium_est * 100:.2f}/contract)")
    print(f"Exit:  {est.exit_date} - underlying ${est.exit_underlying_price}, "
          f"est. premium ${est.exit_premium_est} (${est.exit_premium_est * 100:.2f}/contract)")
    print(f"\nEstimated P&L per contract: ${est.pnl_per_contract_est:+.2f}"
          + (f" ({est.pnl_pct_est:+.1f}%)" if est.pnl_pct_est is not None else ""))


def main() -> None:
    p = argparse.ArgumentParser(description="Theoretical options P&L estimator (Black-Scholes, not real market data)")
    p.add_argument("--ticker", required=True)
    p.add_argument("--dte", type=int, required=True, help="days to expiry from entry date")
    p.add_argument("--entry-date", type=str, default=None, help="YYYY-MM-DD, default today")
    p.add_argument("--exit-date", type=str, default=None, help="YYYY-MM-DD, default = held to expiry")
    p.add_argument("--strike", type=float, default=None, help="default: at-the-money at entry")
    p.add_argument("--type", dest="option_type", choices=["call", "put"], default="call")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    entry_date = datetime.fromisoformat(args.entry_date).date() if args.entry_date else None
    exit_date = datetime.fromisoformat(args.exit_date).date() if args.exit_date else None

    est = estimate_option_pnl(
        ticker=args.ticker, dte_days=args.dte, entry_date=entry_date, exit_date=exit_date,
        strike=args.strike, option_type=args.option_type,
    )
    if est is None:
        print(f"No price data available for {args.ticker} in the requested window.")
        return

    if args.json:
        print(json.dumps(asdict(est), indent=2, default=str))
    else:
        print_estimate(est)


if __name__ == "__main__":
    main()
