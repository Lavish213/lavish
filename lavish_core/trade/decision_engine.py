# lavish_core/trade/decision_engine.py
import os, math, time
from typing import Dict, Any, Optional

AUTO_CONFIRM_MODE = os.getenv("AUTO_CONFIRM_MODE", "smart")  # 'off'|'smart'|'always'
AUTO_CONFIRM_PCT = float(os.getenv("AUTO_CONFIRM_PCT", "0.65"))  # min blended confidence in 'smart'
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))  # cap of equity per position
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))

def _blend_confidence(patreon: Optional[Dict[str, Any]], math_sig: Optional[Dict[str, Any]]) -> float:
    """
    patreon: {'symbol','action','confidence' in [0,1], 'target_qty'?, 'posted_at', 'raw_text'}
    math_sig: {'trend_z','reversal_risk','cycle_strength','vol_z','score' in [0,1]}
    """
    p = patreon.get("confidence", 0.0) if patreon else 0.0
    m = 0.0
    if math_sig:
        # map our features into a 0..1 confirmation score
        # prefer: strong trend, low reversal risk, decent cycle or quiet vol
        trend_ok = min(1.0, max(0.0, 0.5 + 0.3 * math_sig.get("trend_z", 0.0)))   # z>0 -> >0.5
        rev_ok   = 1.0 - min(1.0, math_sig.get("reversal_risk", 0.0))             # lower risk is better
        cyc_ok   = math_sig.get("cycle_strength", 0.0)
        vol_ok   = 1.0 - min(1.0, max(0.0, (math_sig.get("vol_z", 0.0) + 2.0)/4.0))  # high vol_z -> lower
        m = max(0.0, min(1.0, 0.35*trend_ok + 0.25*rev_ok + 0.20*cyc_ok + 0.20*vol_ok))
    # Blend patreon weight higher (the “stock lady” is primary trigger)
    return 0.65*p + 0.35*m

def _shares_from_risk(price: float, equity: float) -> int:
    risk_dollars = equity * RISK_PER_TRADE_PCT
    if risk_dollars <= 0 or price <= 0:
        return 0
    # naive: 1 ATR not available here, approximate using 1.5% of price
    unit_risk = 0.015 * price
    if unit_risk <= 0:
        return 0
    qty = int(max(0, math.floor(risk_dollars / unit_risk)))
    # also cap by MAX_POSITION_PCT of equity
    max_notional = equity * MAX_POSITION_PCT
    return int(min(qty, max(1, math.floor(max_notional / price))))

def decide_trade(
    equity: float,
    patreon: Optional[Dict[str, Any]],
    math_sig: Optional[Dict[str, Any]],
    current_position: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Returns action dict:
    {
      'should_execute': bool,
      'reason': 'auto-confirm'|'needs-review'|...,
      'symbol': str,
      'side': 'buy'|'sell'|'hold',
      'qty': int,
      'tp': float|None,
      'sl': float|None,
      'blended_confidence': float,
      'source': 'patreon'|'math'|'both'
    }
    """
    if not patreon and not math_sig:
        return {"should_execute": False, "reason": "no-signal"}

    symbol = (patreon or {}).get("symbol") or (math_sig or {}).get("symbol")
    action = (patreon or {}).get("action") or "hold"  # 'buy'|'sell'|'hold'
    price = (math_sig or {}).get("price") or (patreon or {}).get("price") or 0.0

    blended = _blend_confidence(patreon, math_sig)

    # auto-confirm policy
    auto = False
    if AUTO_CONFIRM_MODE == "always":
        auto = True
    elif AUTO_CONFIRM_MODE == "smart":
        auto = blended >= AUTO_CONFIRM_PCT
    else:
        auto = False

    # position sizing
    qty = (patreon or {}).get("target_qty")
    if not qty:
        qty = _shares_from_risk(price, equity)

    # direction adjustments from math if it strongly contradicts
    tz = (math_sig or {}).get("trend_z", 0.0)
    if action == "buy" and tz < -1.2 and (patreon or {}).get("confidence", 0) < 0.8:
        # downgrade to review if math says downtrend
        auto = False
    if action == "sell" and tz > 1.2 and (patreon or {}).get("confidence", 0) < 0.8:
        auto = False

    # simple bracket based on price
    tp = round(price * 1.02, 2) if price else None  # +2%
    sl = round(price * 0.985, 2) if price else None # -1.5%

    # if currently long and action=buy, maybe just hold; if currently short & buy, close portion
    side = action
    if current_position:
        side = action  # you can enrich later with netting logic

    return {
        "should_execute": bool(auto and side in ("buy", "sell") and qty and qty > 0),
        "reason": "auto-confirm" if auto else "needs-review",
        "symbol": symbol,
        "side": side,
        "qty": int(qty or 0),
        "tp": tp,
        "sl": sl,
        "blended_confidence": round(blended, 3),
        "source": "both" if (patreon and math_sig) else ("patreon" if patreon else "math")
    }