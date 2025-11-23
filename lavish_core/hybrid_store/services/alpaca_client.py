import requests
from typing import Optional, Dict, Any
from ..config import get_settings
from ..utils.logger import get_logger

log = get_logger("alpaca")
settings = get_settings()

class AlpacaClient:
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret = secret or settings.ALPACA_SECRET_KEY
        self.base_url = (base_url or settings.ALPACA_BASE_URL).rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

    def _req(self, method: str, path: str, json: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        r = requests.request(method, url, headers=self._headers, json=json, params=params, timeout=20)
        if r.status_code >= 400:
            log.error("Alpaca error %s %s: %s", method, path, r.text)
        r.raise_for_status()
        return r.json()

    # Paper/live trading endpoints
    def submit_order(self, symbol: str, side: str, qty: Optional[float] = None, notional: Optional[float] = None,
                     type: str = "market", time_in_force: str = "day",
                     limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                     client_order_id: Optional[str] = None):
        body = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": type.lower(),
            "time_in_force": time_in_force.lower(),
        }
        if qty is not None:
            body["qty"] = qty
        if notional is not None:
            body["notional"] = notional
        if limit_price is not None:
            body["limit_price"] = limit_price
        if stop_price is not None:
            body["stop_price"] = stop_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._req("POST", "/v2/orders", json=body)

    def get_positions(self):
        return self._req("GET", "/v2/positions")

    def get_account(self):
        return self._req("GET", "/v2/account")