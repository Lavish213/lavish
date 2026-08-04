# lavish_core/trade/portfolio_risk.py
# Portfolio-level concentration check - nothing in the rest of the repo
# looked at *other* open positions before sizing/allowing a new one. Three
# alerts on AAPL, MSFT and NVDA in the same session read as three
# independent trades but are really one concentrated bet on mega-cap tech;
# per-symbol risk limits (trade_agent.RiskLimits) don't catch that because
# they only look at exposure to the *same* symbol.
#
# PLANNED UPGRADE (not built - noted here so it isn't lost, not started
# because there's nothing real to build it against yet):
# check_correlation_ok() below is a blunt bucket-count rule (max N
# positions in a hand-maintained correlation group). A real portfolio-risk
# library (riskfolio-lib is the concrete one identified) could replace
# this with actual variance/CVaR-based sizing instead of a flat headcount
# limit. Deliberately NOT wired up yet:
#   - It needs a real covariance matrix, which needs real historical
#     returns for whatever's actually been traded - there's no live
#     trading history yet to compute one from meaningfully.
#   - Pulling in riskfolio-lib (and its own numpy/scipy/cvxpy-class
#     dependency tree) for a feature with nothing real to optimize over
#     yet would bloat requirements-deploy.txt for no working benefit -
#     same reasoning that drove trimming that file down in the first
#     place.
# Trigger to actually build this: once a few weeks of real paper-trading
# history exist (via track_record.py) and the current bucket rule shows
# a concrete case where it was too blunt (blocked something that wasn't
# really concentrated risk, or missed a real concentration it doesn't
# have a bucket for) - build it against that real evidence, not
# speculatively now.
from __future__ import annotations
import os, json, logging, re
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("portfolio_risk")

# Static bucket map covering the tickers this bot actually whitelists/trades
# (see WHITELIST_TICKERS in env.example). Anything not listed here is
# treated as its own single-ticker group, i.e. no assumed correlation -
# a conservative default is "unknown = independent", not "unknown = grouped
# together", since the latter would block unrelated tickers from ever
# co-existing.
_DEFAULT_GROUPS: Dict[str, str] = {
    "AAPL": "mega_tech", "MSFT": "mega_tech", "GOOGL": "mega_tech", "META": "mega_tech",
    "NVDA": "semis", "AMD": "semis", "MU": "semis", "AVGO": "semis",
    "SPY": "broad_index", "QQQ": "broad_index",
    "TSLA": "ev_growth",
    "MSTR": "crypto_proxy", "HOOD": "fintech_growth",
    "CRM": "software", "CRWV": "software",
    "UNH": "healthcare", "NVO": "healthcare", "ATAI": "healthcare",
}

MAX_CORRELATED_POSITIONS = int(os.getenv("MAX_CORRELATED_POSITIONS", "3"))


def _load_groups() -> Dict[str, str]:
    raw = os.getenv("CORRELATION_GROUPS_JSON", "")
    if not raw:
        return _DEFAULT_GROUPS
    try:
        override = json.loads(raw)
        merged = dict(_DEFAULT_GROUPS)
        merged.update({k.upper(): v for k, v in override.items()})
        return merged
    except Exception as e:
        log.warning("CORRELATION_GROUPS_JSON invalid, using defaults: %s", e)
        return _DEFAULT_GROUPS


def group_of(ticker: str) -> str:
    groups = _load_groups()
    return groups.get(ticker.upper(), ticker.upper())


def _underlying_from_symbol(symbol: str) -> str:
    """
    Equity positions are already a plain ticker. Option positions are OCC
    symbols (TICKER + YYMMDD + C/P + strike) - strip at the first digit to
    recover the underlying so a MSFT call and a plain MSFT share position
    still count as the same concentration bucket.
    """
    m = re.match(r"^([A-Z]+)\d", symbol.upper())
    return m.group(1) if m else symbol.upper()


def check_correlation_ok(new_ticker: str, get_positions_fn=None) -> Tuple[bool, Optional[str]]:
    """
    Returns (ok, reason). Counts *distinct underlyings* already held in the
    same correlation group as new_ticker (not contract/share count - one
    large NVDA position and one small NVDA position are still one bet).
    A brand-new position in new_ticker's own underlying is never blocked by
    this (adding to/re-entering the same symbol isn't "more concentration",
    it's the same position) - only *other* symbols in the same group count.
    """
    if get_positions_fn is None:
        from lavish_core.trade.broker_alpaca import get_positions as get_positions_fn

    target_group = group_of(new_ticker)
    new_ticker_u = new_ticker.upper()

    try:
        positions = get_positions_fn() or []
    except Exception as e:
        # Can't verify concentration if the broker call itself fails - fail
        # open here (same posture as latest_quote's soft-fail) rather than
        # blocking every trade because of an unrelated API hiccup. The
        # circuit breaker and per-symbol limits still apply independently.
        log.warning("check_correlation_ok: get_positions failed, skipping check: %s", e)
        return True, None

    held_underlyings = set()
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        qty = float(p.get("qty", 0) or 0)
        if not sym or qty == 0:
            continue
        held_underlyings.add(_underlying_from_symbol(sym))

    correlated = {u for u in held_underlyings if u != new_ticker_u and group_of(u) == target_group}

    if len(correlated) >= MAX_CORRELATED_POSITIONS:
        return False, (
            f"correlation_limit: already holding {len(correlated)} position(s) in group "
            f"'{target_group}' ({sorted(correlated)}), limit is {MAX_CORRELATED_POSITIONS}"
        )
    return True, None
