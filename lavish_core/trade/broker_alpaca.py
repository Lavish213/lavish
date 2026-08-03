# lavish_core/trade/broker_alpaca.py
import os, json, logging, time, uuid
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("broker_alpaca")

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL_RAW = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_URL_RAW = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

# Normalize BASE_URL (allow user to put with or without /v2)
BASE_URL = BASE_URL_RAW.rstrip("/")
DATA_URL = DATA_URL_RAW.rstrip("/")
ORDERS_URL = f"{BASE_URL}/v2/orders"
ACCOUNT_URL = f"{BASE_URL}/v2/account"
POS_URL = f"{BASE_URL}/v2/positions"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json"
}

# A previous version of this codebase had proper retry/backoff (in a file
# since removed as a dead duplicate) that never got ported to this one -
# every call here was a single-attempt raw request with no resilience to
# a transient network blip or an Alpaca rate limit. Only GET is retried
# here: POST (place_order) is NOT safe to blindly retry without knowing
# whether the first attempt already went through server-side - that's
# handled instead via a stable client_order_id (see place_order) so a
# duplicate submission is deduplicated by Alpaca itself, not by retrying
# blind.
_retry = Retry(
    total=3, connect=3, read=3,
    backoff_factor=0.5,  # 0.5s, 1s, 2s
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
    respect_retry_after_header=True,
)
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_retry))

def _check_keys():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")

def get_account() -> Dict[str, Any]:
    _check_keys()
    r = SESSION.get(ACCOUNT_URL, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca account error {r.status_code}: {r.text}")
    return r.json()

def get_clock() -> Dict[str, Any]:
    """
    Alpaca's real market clock (is_open, next_open, next_close) - used
    instead of a hardcoded "4pm ET" so early closes/holidays are handled
    correctly for the expiry-day forced-exit rule.
    """
    _check_keys()
    r = SESSION.get(f"{BASE_URL}/v2/clock", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca clock error {r.status_code}: {r.text}")
    return r.json()

def get_positions() -> list:
    _check_keys()
    r = SESSION.get(POS_URL, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca positions error {r.status_code}: {r.text}")
    return r.json()

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    _check_keys()
    r = SESSION.get(f"{POS_URL}/{symbol.upper()}", headers=HEADERS, timeout=20)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca position error {r.status_code}: {r.text}")
    return r.json()

def get_order(order_id: str) -> Dict[str, Any]:
    _check_keys()
    r = SESSION.get(f"{ORDERS_URL}/{order_id}", headers=HEADERS, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca get_order error {r.status_code}: {r.text}")
    return r.json()

def latest_quote(symbol: str) -> Optional[float]:
    """
    Real last-trade price from Alpaca's market data API, with a quote-midpoint
    fallback. Returns None (never raises) so callers can soft-fail to other
    pricing when the market is closed or the symbol has no recent data.
    """
    _check_keys()
    symbol = symbol.upper()
    try:
        r = SESSION.get(
            f"{DATA_URL}/v2/stocks/{symbol}/trades/latest",
            headers=HEADERS, timeout=10,
        )
        if r.status_code == 200:
            p = r.json().get("trade", {}).get("p")
            if p:
                return float(p)
    except Exception as e:
        log.warning(f"latest_quote trade lookup failed for {symbol}: {e}")

    try:
        r = SESSION.get(
            f"{DATA_URL}/v2/stocks/{symbol}/quotes/latest",
            headers=HEADERS, timeout=10,
        )
        if r.status_code == 200:
            q = r.json().get("quote", {})
            ap, bp = q.get("ap"), q.get("bp")
            if ap and bp:
                return float((ap + bp) / 2)
            if ap or bp:
                return float(ap or bp)
    except Exception as e:
        log.warning(f"latest_quote quote lookup failed for {symbol}: {e}")

    return None

def place_order(symbol: str, side: str, qty: str,
                type_: str = "market", time_in_force: str = "day",
                limit_price: Optional[float] = None,
                take_profit: Optional[float] = None,
                stop_loss: Optional[float] = None,
                client_order_id: Optional[str] = None) -> Dict[str, Any]:
    """
    side: 'buy'|'sell'
    qty: string or numeric (Alpaca expects string for notional sometimes)
    type_: market|limit|stop|stop_limit
    """
    _check_keys()
    # A client_order_id is always set, even if the caller didn't pass one.
    # Order submission is a POST and isn't blindly retried (see SESSION's
    # retry policy above, GET-only) precisely because a retry after a
    # timeout can't tell whether the first attempt already went through.
    # A stable idempotency key closes that gap the safe way: if this ever
    # does get submitted twice (a retry layer above this, a network hiccup
    # that looked like a failure but wasn't), Alpaca rejects the second
    # submission as a duplicate client_order_id instead of opening a
    # second position. Minute-bucketed so legitimate distinct orders for
    # the same symbol/side/qty a few minutes apart still get through.
    if not client_order_id:
        minute_bucket = int(time.time() // 60)
        raw = f"{symbol}|{side}|{qty}|{type_}|{minute_bucket}"
        client_order_id = "lavish_" + uuid.uuid5(uuid.NAMESPACE_OID, raw).hex[:20]

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "qty": str(qty),
        "type": type_,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }
    if type_ in ("limit", "stop_limit") and limit_price is not None:
        payload["limit_price"] = str(limit_price)
    if take_profit is not None or stop_loss is not None:
        payload["order_class"] = "bracket"
        if take_profit is not None:
            payload["take_profit"] = {"limit_price": str(take_profit)}
        if stop_loss is not None:
            payload["stop_loss"] = {"stop_price": str(stop_loss)}

    log.info(f"[Alpaca] place_order {json.dumps(payload)}")
    r = SESSION.post(ORDERS_URL, headers=HEADERS, data=json.dumps(payload), timeout=20)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Alpaca order error {r.status_code}: {r.text}")
    return r.json()