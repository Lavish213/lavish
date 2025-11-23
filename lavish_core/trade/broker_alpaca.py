# lavish_core/trade/broker_alpaca.py
import os, json, logging, time
from typing import Optional, Dict, Any
import requests

log = logging.getLogger("broker_alpaca")

API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL_RAW = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Normalize BASE_URL (allow user to put with or without /v2)
BASE_URL = BASE_URL_RAW.rstrip("/")
ORDERS_URL = f"{BASE_URL}/v2/orders"
ACCOUNT_URL = f"{BASE_URL}/v2/account"
POS_URL = f"{BASE_URL}/v2/positions"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json"
}

def _check_keys():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")

def get_account() -> Dict[str, Any]:
    _check_keys()
    r = requests.get(ACCOUNT_URL, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca account error {r.status_code}: {r.text}")
    return r.json()

def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    _check_keys()
    r = requests.get(f"{POS_URL}/{symbol.upper()}", headers=HEADERS, timeout=20)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca position error {r.status_code}: {r.text}")
    return r.json()

def place_order(symbol: str, side: str, qty: str, 
                type_: str = "market", time_in_force: str = "day",
                take_profit: Optional[float] = None,
                stop_loss: Optional[float] = None,
                client_order_id: Optional[str] = None) -> Dict[str, Any]:
    """
    side: 'buy'|'sell'
    qty: string or numeric (Alpaca expects string for notional sometimes)
    type_: market|limit|stop|stop_limit
    """
    _check_keys()
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "qty": str(qty),
        "type": type_,
        "time_in_force": time_in_force
    }
    if take_profit is not None or stop_loss is not None:
        payload["order_class"] = "bracket"
        if take_profit is not None:
            payload["take_profit"] = {"limit_price": str(take_profit)}
        if stop_loss is not None:
            payload["stop_loss"] = {"stop_price": str(stop_loss)}
    if client_order_id:
        payload["client_order_id"] = client_order_id

    log.info(f"[Alpaca] place_order {json.dumps(payload)}")
    r = requests.post(ORDERS_URL, headers=HEADERS, data=json.dumps(payload), timeout=20)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Alpaca order error {r.status_code}: {r.text}")
    return r.json()