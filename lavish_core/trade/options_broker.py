# lavish_core/trade/options_broker.py
# Options order execution against Alpaca. Nothing in the rest of the repo
# could place an options order before this - only plain equity orders existed.
from __future__ import annotations
import os, json, logging
from datetime import date, datetime
from typing import Optional, Dict, Any, List
import requests

from lavish_core.trade.broker_alpaca import HEADERS, DATA_URL, _check_keys

log = logging.getLogger("options_broker")

BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ORDERS_URL = f"{BASE_URL}/v2/orders"
CONTRACTS_URL = f"{BASE_URL}/v2/options/contracts"


def build_occ_symbol(ticker: str, expiry: date, side: str, strike: float) -> str:
    """
    Standard OCC symbol: TICKER + YYMMDD + C/P + strike*1000 zero-padded to 8 digits.
    e.g. AAPL, 2024-05-31, "call", 190.0 -> AAPL240531C00190000
    """
    ticker = ticker.upper().strip()
    cp = "C" if side.lower().startswith("c") else "P"
    strike_int = round(strike * 1000)
    return f"{ticker}{expiry.strftime('%y%m%d')}{cp}{strike_int:08d}"


def find_contract(
    ticker: str,
    expiry: date,
    side: str,
    strike: float,
    strike_tolerance: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Look up the actual tradable contract on Alpaca closest to the requested
    strike/expiry. A caller-guessed strike from an OCR'd/parsed alert won't
    always match a listed strike exactly (weeklies vs monthlies, $2.50 vs $5
    increments), so this resolves to what's really tradable instead of
    hoping build_occ_symbol() guessed a real contract.
    """
    _check_keys()
    params = {
        "underlying_symbols": ticker.upper(),
        "expiration_date": expiry.isoformat(),
        "type": "call" if side.lower().startswith("c") else "put",
        "status": "active",
        "limit": 100,
    }
    r = requests.get(CONTRACTS_URL, headers=HEADERS, params=params, timeout=20)
    if r.status_code != 200:
        log.warning("options contract lookup failed %s: %s", r.status_code, r.text[:300])
        return None

    contracts: List[Dict[str, Any]] = r.json().get("option_contracts", [])
    if not contracts:
        return None

    best = min(contracts, key=lambda c: abs(float(c.get("strike_price", 0)) - strike))
    deviation = abs(float(best.get("strike_price", 0)) - strike)
    if strike_tolerance and deviation > strike_tolerance:
        log.warning(
            "closest listed strike %.2f is %.2f away from requested %.2f (tolerance %.2f) for %s",
            float(best["strike_price"]), deviation, strike, strike_tolerance, ticker,
        )
        return None
    return best


def latest_option_quote(contract_symbol: str) -> Optional[Dict[str, float]]:
    """Best bid/ask for a contract, so callers can sanity-check spread before submitting."""
    _check_keys()
    try:
        r = requests.get(
            f"{DATA_URL}/v1beta1/options/quotes/latest",
            headers=HEADERS, params={"symbols": contract_symbol}, timeout=10,
        )
        if r.status_code != 200:
            return None
        q = r.json().get("quotes", {}).get(contract_symbol)
        if not q:
            return None
        return {"bid": float(q.get("bp", 0) or 0), "ask": float(q.get("ap", 0) or 0)}
    except Exception as e:
        log.warning("latest_option_quote failed for %s: %s", contract_symbol, e)
        return None


def place_option_order(
    contract_symbol: str,
    side: str,
    qty: int,
    order_type: str = "limit",
    limit_price: Optional[float] = None,
    time_in_force: str = "day",
    client_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit a single-leg options order. Alpaca doesn't support market orders
    reliably on options given typically wide spreads - default to limit,
    with the caller expected to pass a sane limit_price (e.g. from
    latest_option_quote's midpoint). No bracket/OCO support for options on
    Alpaca - exit management has to be a separate poll-and-close loop
    (see options_exit_monitor.py).
    """
    _check_keys()
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if order_type == "limit" and limit_price is None:
        raise ValueError("limit_price is required for limit orders")

    payload: Dict[str, Any] = {
        "symbol": contract_symbol,
        "qty": str(int(qty)),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if order_type == "limit":
        payload["limit_price"] = str(limit_price)
    if client_order_id:
        payload["client_order_id"] = client_order_id

    log.info("[Alpaca options] place_order %s", json.dumps(payload))
    r = requests.post(ORDERS_URL, headers=HEADERS, data=json.dumps(payload), timeout=20)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Alpaca options order error {r.status_code}: {r.text}")
    return r.json()


def resolve_and_price_contract(
    ticker: str, expiry: date, side: str, strike: float, strike_tolerance: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Convenience: find the real contract, then attach a current quote/midpoint to it."""
    contract = find_contract(ticker, expiry, side, strike, strike_tolerance=strike_tolerance)
    if not contract:
        return None
    quote = latest_option_quote(contract["symbol"])
    if quote and quote["bid"] and quote["ask"]:
        contract["mid_price"] = round((quote["bid"] + quote["ask"]) / 2, 2)
        contract["bid"] = quote["bid"]
        contract["ask"] = quote["ask"]
    return contract
