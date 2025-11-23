# lavish_core/trading/executor.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

TRADE_MODE = os.getenv("TRADE_MODE", "mock").lower()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DEFAULT_QTY = int(os.getenv("DEFAULT_QTY", "1"))

@dataclass
class TradeSignal:
    action: str     # BUY / SELL
    symbol: str     # e.g. AAPL
    qty: int        # shares
    reason: str     # free text – “from OCR”, “from Patreon”, etc.

class TradeExecutor:
    def __init__(self):
        self.mode = TRADE_MODE
        if self.mode in ("paper", "live"):
            self.client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=(self.mode == "paper"),
                url_override=ALPACA_BASE_URL
            )
        else:
            self.client = None  # mock

    def place(self, sig: TradeSignal) -> dict:
        """Places or simulates a market order. Returns a dict summary."""
        summary = {
            "mode": self.mode, "action": sig.action, "symbol": sig.symbol,
            "qty": sig.qty, "reason": sig.reason
        }

        if self.mode == "mock":
            summary["status"] = "SIMULATED"
            print(f"🟡 [MOCK] {sig.action} {sig.qty} {sig.symbol} — {sig.reason}")
            return summary

        # Paper / Live
        side = OrderSide.BUY if sig.action.upper() == "BUY" else OrderSide.SELL
        order_req = MarketOrderRequest(
            symbol=sig.symbol.upper(),
            qty=sig.qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        try:
            order = self.client.submit_order(order_req)
            summary["status"] = "PLACED"
            summary["alpaca_order_id"] = order.id
            print(f"✅ [{self.mode.upper()}] {sig.action} {sig.qty} {sig.symbol} — {sig.reason} (order {order.id})")
        except Exception as e:
            summary["status"] = "ERROR"
            summary["error"] = str(e)
            print(f"❌ [{self.mode.upper()}] Failed to place order: {e}")

        return summary