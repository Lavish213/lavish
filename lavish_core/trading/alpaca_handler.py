# lavish_core/trade/alpaca_handler.py
from __future__ import annotations

import os
import csv
import math
import json
import atexit
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# --------------------------
# ENV + CONSTANTS
# --------------------------
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
DEFAULT_TIF = os.getenv("ALPACA_TIF", "day").lower()  # "day" or "gtc"

HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_BACKOFF_SEC = float(os.getenv("HTTP_BACKOFF_SEC", "1.5"))
RISK_CAP_PCT = float(os.getenv("RISK_CAP_PCT", "0.02"))  # max % of equity deployed per order
DRY_RUN = os.getenv("LAVISH_DRY_RUN", "0") == "1"         # set 1 to simulate without sending orders

LOG_DIR = Path(os.getenv("LAVISH_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / "trades.csv"

# --------------------------
# HTTP SESSION (RETRIES)
# --------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type": "application/json",
        }
    )
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        connect=HTTP_MAX_RETRIES,
        read=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_SEC,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = _session()

# --------------------------
# MODELS
# --------------------------
@dataclass
class TradeIntent:
    symbol: str
    side: str  # "buy" | "sell"
    order_type: str = "market"  # "market"|"limit"|"stop"|"stop_limit"
    qty: Optional[int] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    risk_dollars: Optional[float] = None     # if set, we size by risk
    stop_pct: float = 0.02                   # 2% default
    take_pct: float = 0.04                   # 4% default
    tif: str = DEFAULT_TIF                   # day|gtc
    client_order_id: Optional[str] = None
    allow_bracket: bool = True               # enable take/stop bracket

# --------------------------
# UTILITIES
# --------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _log_trade(row: Dict[str, Any]) -> None:
    write_header = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ts","intent","symbol","qty","side","type","limit_price","stop_price",
                "take_profit","stop_loss","order_id","status","reason","dry_run"
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(row)

def _get(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    r = SESSION.get(url, params=params, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {}

def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = SESSION.post(url, data=json.dumps(payload), timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {}

def _delete(url: str) -> Dict[str, Any]:
    r = SESSION.delete(url, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"DELETE {url} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {}

# --------------------------
# CORE QUERIES
# --------------------------
def get_account() -> Dict[str, Any]:
    return _get(f"{ALPACA_BASE_URL}/v2/account")

def get_clock() -> Dict[str, Any]:
    return _get(f"{ALPACA_BASE_URL}/v2/clock")

def market_is_open() -> bool:
    try:
        clk = get_clock()
        return bool(clk.get("is_open", False))
    except Exception:
        return False

def get_positions() -> list[Dict[str, Any]]:
    return _get(f"{ALPACA_BASE_URL}/v2/positions")

def cancel_all_orders() -> None:
    try:
        _delete(f"{ALPACA_BASE_URL}/v2/orders")
        print("🧹 Cancelled all open orders.")
    except Exception as e:
        print(f"⚠️ Cancel orders failed: {e}")

def close_position(symbol: str) -> None:
    try:
        _delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}")
        print(f"🔻 Closed position: {symbol}")
    except Exception as e:
        print(f"⚠️ Close position failed for {symbol}: {e}")

def latest_trade_price(symbol: str) -> float:
    """
    Pull the latest trade price; fallback to last quote if necessary.
    """
    try:
        j = _get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest")
        return float(j["trade"]["p"])
    except Exception:
        q = _get(f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/quotes/latest")
        return float(q["quote"]["ap"] or q["quote"]["bp"])

# --------------------------
# ORDERING
# --------------------------
def submit_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str = "market",
    time_in_force: str = DEFAULT_TIF,
    limit_price: float | None = None,
    stop_price: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    client_order_id: str | None = None,
) -> Dict[str, Any]:
    """
    Sends an order to Alpaca (or logs it if DRY_RUN=1).
    Supports: market|limit|stop|stop_limit and optional bracket.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    order_type = order_type.lower()
    if order_type not in ("market", "limit", "stop", "stop_limit"):
        raise ValueError("order_type must be market|limit|stop|stop_limit")

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id
    if order_type in ("limit", "stop_limit") and limit_price:
        payload["limit_price"] = str(limit_price)
    if order_type in ("stop", "stop_limit") and stop_price:
        payload["stop_price"] = str(stop_price)
    if take_profit is not None or stop_loss is not None:
        payload["order_class"] = "bracket"
        if take_profit is not None:
            payload["take_profit"] = {"limit_price": str(take_profit)}
        if stop_loss is not None:
            payload["stop_loss"] = {"stop_price": str(stop_loss)}

    if DRY_RUN:
        # simulate a response
        faux = {"id": "SIMULATED", "status": "accepted", "symbol": symbol.upper(), "payload": payload}
        _log_trade(
            {
                "ts": _utc_now_iso(),
                "intent": "submit_order",
                "symbol": symbol.upper(),
                "qty": qty,
                "side": side,
                "type": order_type,
                "limit_price": limit_price or "",
                "stop_price": stop_price or "",
                "take_profit": take_profit or "",
                "stop_loss": stop_loss or "",
                "order_id": faux["id"],
                "status": faux["status"],
                "reason": "",
                "dry_run": "1",
            }
        )
        print("🧪 DRY-RUN order:", json.dumps(faux, indent=2))
        return faux

    j = _post(f"{ALPACA_BASE_URL}/v2/orders", payload)
    _log_trade(
        {
            "ts": _utc_now_iso(),
            "intent": "submit_order",
            "symbol": symbol.upper(),
            "qty": qty,
            "side": side,
            "type": order_type,
            "limit_price": limit_price or "",
            "stop_price": stop_price or "",
            "take_profit": take_profit or "",
            "stop_loss": stop_loss or "",
            "order_id": j.get("id", ""),
            "status": j.get("status", ""),
            "reason": "",
            "dry_run": "0",
        }
    )
    return j

def risk_sized_market_order(
    symbol: str,
    side: str,
    risk_dollars: float | None = None,
    stop_pct: float = 0.02,
    take_pct: float = 0.04,
    cap_to_equity_pct: float = RISK_CAP_PCT,
    tif: str = DEFAULT_TIF,
) -> Dict[str, Any]:
    """
    Size by risk and cap:
      qty = floor(min( risk_dollars / (price * stop_pct), equity*cap_to_equity_pct / price ))
    Places market bracket with take/stop.
    """
    acct = get_account()
    equity = float(acct.get("equity", "0") or 0)
    price = latest_trade_price(symbol)

    if not risk_dollars:
        # default: max( $50, 0.5% equity )
        risk_dollars = max(50.0, equity * 0.005)

    cap_dollars = equity * cap_to_equity_pct
    risk_qty = risk_dollars / max(price * stop_pct, 1e-9)
    cap_qty = cap_dollars / max(price, 1e-9)
    qty = max(1, int(math.floor(min(risk_qty, cap_qty))))

    take_price = round(price * (1 + take_pct * (1 if side.lower() == "buy" else -1)), 2)
    stop_price = round(price * (1 - stop_pct * (1 if side.lower() == "buy" else -1)), 2)

    print(
        f"🧮 {symbol} px={price:.2f} | equity=${equity:.2f} | "
        f"qty={qty} | take={take_price} | stop={stop_price} | tif={tif}"
    )

    return submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        order_type="market",
        time_in_force=tif,
        take_profit=take_price,
        stop_loss=stop_price,
    )

# --------------------------
# PUBLIC ENTRY FOR EXTERNAL TRIGGERS
# --------------------------
def execute_trade_from_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize & execute a trade from an external post (Patreon, Discord, etc).

    Expected flexible fields in `post`:
      - symbol / ticker
      - side: buy|sell|long|short
      - qty (optional if using risk)
      - order_type: market|limit|stop|stop_limit
      - limit_price / stop_price (for non-market types)
      - risk_dollars (optional; if present we do risk sizing)
      - stop_pct / take_pct (optional)
      - tif: day|gtc
      - client_order_id (optional)
    """
    sym = (post.get("symbol") or post.get("ticker") or "").upper().strip()
    if not sym:
        raise ValueError("Missing symbol/ticker")

    side_raw = (post.get("side") or "").lower().strip()
    side = "buy" if side_raw in ("buy", "long") else "sell" if side_raw in ("sell", "short") else None
    if side is None:
        raise ValueError("Invalid side, expected buy|sell|long|short")

    order_type = (post.get("order_type") or "market").lower()
    tif = (post.get("tif") or DEFAULT_TIF).lower()

    qty = post.get("qty")
    qty = int(qty) if qty not in (None, "") else None

    limit_price = float(post["limit_price"]) if "limit_price" in post and post["limit_price"] not in (None, "") else None
    stop_price  = float(post["stop_price"])  if "stop_price"  in post and post["stop_price"]  not in (None, "") else None

    risk_dollars = float(post["risk_dollars"]) if "risk_dollars" in post and post["risk_dollars"] not in (None, "") else None
    stop_pct = float(post.get("stop_pct", 0.02) or 0.02)
    take_pct = float(post.get("take_pct", 0.04) or 0.04)
    client_order_id = post.get("client_order_id")

    # If risk is given -> risk sizing market bracket
    if risk_dollars and order_type == "market":
        return risk_sized_market_order(
            symbol=sym,
            side=side,
            risk_dollars=risk_dollars,
            stop_pct=stop_pct,
            take_pct=take_pct,
            tif=tif,
        )

    # Else if qty given -> submit directly
    if qty is None:
        # Fall back: 1 share if nothing provided (keeps action deterministic)
        qty = 1

    return submit_order(
        symbol=sym,
        qty=qty,
        side=side,
        order_type=order_type,
        time_in_force=tif,
        limit_price=limit_price,
        stop_price=stop_price,
        client_order_id=client_order_id,
    )

# --------------------------
# SAFETY / SHUTDOWN
# --------------------------
def _graceful_exit(*_):
    try:
        _log_trade(
            {
                "ts": _utc_now_iso(),
                "intent": "shutdown",
                "symbol": "",
                "qty": "",
                "side": "",
                "type": "",
                "limit_price": "",
                "stop_price": "",
                "take_profit": "",
                "stop_loss": "",
                "order_id": "",
                "status": "EXIT",
                "reason": "user_interrupt",
                "dry_run": "1" if DRY_RUN else "0",
            }
        )
    except Exception:
        pass
    print("👋 Exiting cleanly.")
    raise SystemExit(0)

signal.signal(signal.SIGINT, _graceful_exit)
atexit.register(lambda: None)

# --------------------------
# SELF-TEST (optional)
# --------------------------
def main() -> None:
    print("🔍 Checking Alpaca connectivity ...")
    try:
        acct = get_account()
        print(f"✅ Account {acct.get('id')} | cash ${acct.get('cash')} | equity ${acct.get('equity')}")
    except Exception as e:
        print(f"❌ Alpaca error: {e}")
        return

    print("⏰ Clock:", get_clock())
    print("📊 Open positions:", [p.get("symbol") for p in get_positions()])

    if os.getenv("RUN_DEMO_ORDER", "0") == "1":
        print("🧪 Demo risk-sized bracket on AAPL (set LAVISH_DRY_RUN=1 for sim).")
        execute_trade_from_post({"symbol": "AAPL", "side": "buy", "order_type": "market", "risk_dollars": 100})

if __name__ == "__main__":
    main()
    
   # === Lavish 24/7 Paper Trade Override ===
# Drop this at the very bottom of alpaca_handler.py

def force_paper_trade_if_closed(order_func):
    """
    Forces paper trades to execute 24/7 even when markets are closed.
    In live mode, it will still try-catch gracefully instead of failing.
    """
    def wrapper(*args, **kwargs):
        import datetime
        from lavish_core.config import PAPER_MODE

        now = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            if PAPER_MODE:
                print(f"⚙️ [{now}] PAPER_MODE active — forcing trade even if market closed.")
                return order_func(*args, **kwargs)
            else:
                # Try normal trade first; if market is closed, print friendly notice.
                try:
                    return order_func(*args, **kwargs)
                except Exception as e:
                    if "market is closed" in str(e).lower():
                        print(f"🕓 [{now}] Market closed — skipping live trade safely.")
                        return None
                    raise
        except Exception as e:
            print(f"⚠️ Trade override error: {e}")
            return None
    return wrapper


# === Apply override to main trade function ===
try:
    execute_trade_from_post = force_paper_trade_if_closed(execute_trade_from_post)
    print("✅ Lavish 24/7 Paper Trade Mode Enabled (No Market Status Needed)")
except NameError:
    print("⚠️ Could not find execute_trade_from_post – patch manually if needed.")