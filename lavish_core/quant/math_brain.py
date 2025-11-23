# lavish_core/quant/math_brain.py
# Robust, dependency-light portfolio math & trade logic.
# - Instrument-agnostic sizing using risk %, ATR stops, vol throttle
# - Trailing stop + time stop + partial profit-take
# - Kelly fraction cap for sanity in high-edge regimes
# - Optional price fetchers (Polygon/AlphaVantage/Finnhub) if keys exist
# - Pure functions first; side-effect free; easy to unit-test

from __future__ import annotations
import os, math, time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import statistics

# ------------------ Config from ENV ------------------
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "25000"))        # fallback if we can’t fetch equity
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005"))        # 0.5% default
MAX_NOTIONAL = float(os.getenv("MAX_NOTIONAL", "10000"))            # per trade cap
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "200"))              # tiny guard
VOL_THROTTLE = float(os.getenv("VOL_THROTTLE", "2.5"))              # z-vol cap -> damp sizing
KELLY_CAP = float(os.getenv("KELLY_CAP", "0.1"))                    # 10% max Kelly allocation cap
DEFAULT_ATR_LEN = int(os.getenv("DEFAULT_ATR_LEN", "14"))
TIME_STOP_BARS = int(os.getenv("TIME_STOP_BARS", "240"))            # ~one trading day on 1-min
PT1_R_MULT = float(os.getenv("PT1_R_MULT", "1.0"))                  # take 1/3 at 1R
PT2_R_MULT = float(os.getenv("PT2_R_MULT", "2.0"))                  # take 1/3 at 2R; rest trail
TRAIL_MULT = float(os.getenv("TRAIL_MULT", "3.0"))                  # ATR trailing multiple

# ------------------ Data structures ------------------
@dataclass
class Bar:
    high: float
    low: float
    close: float

@dataclass
class PositionPlan:
    qty: float
    stop: float
    pt1: float
    pt2: float
    trail_mult: float
    r_amount: float       # $ risk per share
    max_hold_bars: int
    notes: str

# ------------------ Core math utilities ------------------
def ema(values: List[float], span: int) -> List[float]:
    if not values: return []
    k = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out

def zscore(values: List[float]) -> List[float]:
    if len(values) < 2: return [0.0 for _ in values]
    m = statistics.mean(values)
    sd = statistics.pstdev(values)
    return [0.0 if sd == 0 else (v - m) / sd for v in values]

def atr(bars: List[Bar], length: int = DEFAULT_ATR_LEN) -> List[float]:
    """ Wilder ATR """
    if len(bars) < 2: return [0.0] * len(bars)
    trs: List[float] = []
    for i in range(len(bars)):
        if i == 0:
            trs.append(bars[i].high - bars[i].low)
        else:
            h, l, pc = bars[i].high, bars[i].low, bars[i-1].close
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    # Wilder style smoothing
    out = []
    alpha = 1.0 / length
    for i, tr in enumerate(trs):
        if i == 0:
            out.append(tr)
        else:
            out.append(out[-1] + alpha * (tr - out[-1]))
    return out

# ------------------ Sizing & plan ------------------
def kelly_fraction(win_rate: float, rr: float) -> float:
    """
    Kelly f* = p - (1-p)/R ; cap & clamp to [0,1]
    """
    p = max(0.0, min(1.0, win_rate))
    R = max(1e-6, rr)
    f = p - (1 - p)/R
    return max(0.0, min(1.0, f))

def position_plan(
    *,
    side: str,                          # "BUY" or "SELL" (short)
    price: float,
    bars: List[Bar],                    # most recent last
    equity: Optional[float] = None,
    risk_per_trade: Optional[float] = None,
    atr_len: int = DEFAULT_ATR_LEN,
    vol_z_recent: Optional[float] = None,
    est_win_rate: float = 0.52,
    est_rr: float = 1.5
) -> PositionPlan:
    """
    Build a full plan: qty, initial stop, profit targets, trailing parameters.
    - Stop distance = 1 * ATR (can adjust using VOL_THROTTLE damping)
    - Profit targets at 1R and 2R; then trail at TRAIL_MULT * ATR
    - Kelly-capped allocation influences notional cap
    """
    equity = equity if equity is not None else ACCOUNT_EQUITY
    risk_per_trade = risk_per_trade if risk_per_trade is not None else RISK_PER_TRADE

    # ATR
    A = atr(bars, length=atr_len)
    last_atr = A[-1] if A else max(0.5, price * 0.01)  # fallback ~1% of price
    # Volatility throttle: if vol z is big, damp effective risk
    vol_factor = 1.0
    if vol_z_recent is not None and vol_z_recent > 0:
        # linear damp up to VOL_THROTTLE
        vol_factor = max(0.35, 1.0 - min(vol_z_recent, VOL_THROTTLE)/ (VOL_THROTTLE + 1e-9))

    dollar_risk = equity * risk_per_trade * vol_factor

    # Kelly cap (soft): scale allowed risk by kelly fraction (but don’t raise above base)
    kf = min(KELLY_CAP, kelly_fraction(est_win_rate, est_rr))
    dollar_risk = min(dollar_risk, equity * kf)

    # $ risk per share = 1 * ATR
    r_per_share = max(0.01, last_atr)

    qty = math.floor(max(1.0, dollar_risk / r_per_share))
    notional = qty * price
    if notional > MAX_NOTIONAL:
        qty = max(1.0, math.floor(MAX_NOTIONAL / max(0.01, price)))
    if qty * price < MIN_NOTIONAL:
        qty = max(0.0, math.floor(MIN_NOTIONAL / max(0.01, price)))

    # Stops & targets
    if side.upper() == "BUY":
        stop = price - r_per_share
        pt1  = price + PT1_R_MULT * r_per_share
        pt2  = price + PT2_R_MULT * r_per_share
    else:
        stop = price + r_per_share
        pt1  = price - PT1_R_MULT * r_per_share
        pt2  = price - PT2_R_MULT * r_per_share

    notes = f"ATR={last_atr:.3f} vol_factor={vol_factor:.2f} kelly={kf:.3f} risk=${dollar_risk:.2f}"
    return PositionPlan(
        qty=float(qty),
        stop=float(stop),
        pt1=float(pt1),
        pt2=float(pt2),
        trail_mult=float(TRAIL_MULT),
        r_amount=float(r_per_share),
        max_hold_bars=int(TIME_STOP_BARS),
        notes=notes
    )

# ------------------ Trailing & lifecycle helpers ------------------
def trail_stop(side: str, last_price: float, bars: List[Bar], plan: PositionPlan) -> float:
    """ ATR trailing stop """
    A = atr(bars, length=DEFAULT_ATR_LEN)
    a = A[-1] if A else plan.r_amount
    if side.upper() == "BUY":
        return max(plan.stop, last_price - plan.trail_mult * a)
    else:
        return min(plan.stop, last_price + plan.trail_mult * a)

def should_take_profit(side: str, last_price: float, plan: PositionPlan) -> Tuple[bool, str]:
    if side.upper() == "BUY":
        if last_price >= plan.pt2: return True, "TP2"
        if last_price >= plan.pt1: return True, "TP1"
        return False, ""
    else:
        if last_price <= plan.pt2: return True, "TP2"
        if last_price <= plan.pt1: return True, "TP1"
        return False, ""

def time_stop_reached(bars_held: int, plan: PositionPlan) -> bool:
    return bars_held >= plan.max_hold_bars

# ------------------ Optional lightweight fetchers (fallbacks) ------------------
# NOTE: These are optional helpers if you want math_brain to be able to self-serve prices.
# If env keys not present, these will simply return None.

import json, urllib.request

def _http_get_json(url: str, headers: Optional[Dict[str,str]]=None, timeout: int=8) -> Optional[Dict]:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def fetch_bars_alpha(symbol: str, interval: str="1min", limit: int=200) -> Optional[List[Bar]]:
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key: return None
    base = "https://www.alphavantage.co/query"
    if interval.lower() in ("1min","5min","15min","30min","60min"):
        fn = "TIME_SERIES_INTRADAY"
        url = f"{base}?function={fn}&symbol={symbol}&interval={interval}&outputsize=compact&apikey={key}"
        js = _http_get_json(url)
        if not js: return None
        k = next((x for x in js.keys() if "Time Series" in x), None)
        if not k: return None
        series = js[k]
        rows = []
        for _, v in sorted(series.items()):
            try:
                rows.append(Bar(high=float(v["2. high"]), low=float(v["3. low"]), close=float(v["4. close"])))
            except Exception: pass
        return rows[-limit:]
    return None

def fetch_bars_polygon(symbol: str, timespan: str="minute", limit: int=200) -> Optional[List[Bar]]:
    key = os.getenv("POLYGON_API_KEY")
    if not key: return None
    # last X aggregates
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/now/now?limit={limit}&adjusted=true&sort=asc&apiKey={key}"
    js = _http_get_json(url)
    if not js or js.get("status")!="OK": return None
    rows = []
    for it in js.get("results", []):
        try:
            rows.append(Bar(high=float(it["h"]), low=float(it["l"]), close=float(it["c"])))
        except Exception: pass
    return rows

def fetch_bars_finnhub(symbol: str, resolution: str="1", limit: int=200) -> Optional[List[Bar]]:
    key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY_2")
    if not key: return None
    # simplistic: last N minutes using current time
    import time as _t
    now = int(_t.time())
    frm = now - 60*limit
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={resolution}&from={frm}&to={now}&token={key}"
    js = _http_get_json(url)
    if not js or js.get("s")!="ok": return None
    rows = []
    for h,l,c in zip(js["h"], js["l"], js["c"]):
        try:
            rows.append(Bar(high=float(h), low=float(l), close=float(c)))
        except Exception: pass
    return rows

# Convenience: try polygon -> alpha -> finnhub
def fetch_bars_any(symbol: str, limit: int=200) -> Optional[List[Bar]]:
    return (
        fetch_bars_polygon(symbol, "minute", limit) or
        fetch_bars_alpha(symbol, "1min", limit) or
        fetch_bars_finnhub(symbol, "1", limit))