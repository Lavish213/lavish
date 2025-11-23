"""
Signal Grading — transparent 0..100 score with breakdown.
Inputs: bars (from candle_reader.read_candles)
Outputs: score, label, diagnostics (JSON-safe)
"""

from __future__ import annotations
import math, numpy as np
from typing import Dict, List

# ---- Helpers ----
def _body_ratio(b):
    rng = max(1e-9, b["high"] - b["low"])
    body = abs(b["close"] - b["open"])
    return float(body / rng)

def _is_bull(b): return b["close"] > b["open"]
def _is_bear(b): return b["open"] > b["close"]

def _pattern_strength(bars: List[Dict]) -> float:
    """Very conservative pattern confidence 0..1."""
    if len(bars) < 3: return 0.2
    b1,b2,b3 = bars[-3], bars[-2], bars[-1]
    # Engulfing (bull)
    bull_engulf = (_is_bear(b2) and _is_bull(b3) and b3["close"] > b2["open"] and b3["open"] < b2["close"])
    bear_engulf = (_is_bull(b2) and _is_bear(b3) and b3["open"] > b2["close"] and b3["close"] < b2["open"])
    # Hammer / Shooting star (shadow to body ratio)
    def _lower_shadow(b):
        return max(0.0, (min(b["open"], b["close"]) - b["low"]))
    def _upper_shadow(b):
        return max(0.0, (b["high"] - max(b["open"], b["close"])))
    hammer = _lower_shadow(b3) > 2.5*abs(b3["close"]-b3["open"]) and _is_bull(b3)
    star   = _upper_shadow(b3) > 2.5*abs(b3["close"]-b3["open"]) and _is_bear(b3)

    score = 0.0
    if bull_engulf: score += 0.35
    if bear_engulf: score += 0.35
    if hammer:      score += 0.25
    if star:        score += 0.25
    # Big body helps
    score += min(0.25, 0.25*_body_ratio(b3)*3)
    return min(1.0, score)

def _trend_context(bars: List[Dict]) -> float:
    if len(bars) < 6: return 0.5
    closes = np.array([b["close"] for b in bars], dtype=float)
    x = np.arange(len(closes[-6:]))
    m = np.polyfit(x, closes[-6:], 1)[0]
    # map slope → 0..1 (symmetrically)
    return float(0.5 + np.tanh(m*5.0)/2.0)

def _volume_confirm(_: List[Dict]) -> float:
    # Placeholder (no volume from image yet): neutral 0.5
    return 0.5

def _atr_volatility(bars: List[Dict]) -> float:
    if len(bars) < 4: return 0.5
    highs = np.array([b["high"] for b in bars], dtype=float)
    lows  = np.array([b["low"]  for b in bars], dtype=float)
    closes= np.array([b["close"] for b in bars], dtype=float)
    trs=[highs[0]-lows[0]]
    for i in range(1,len(bars)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = np.mean(trs[-3:])
    rng = np.mean(highs - lows)
    if rng <= 1e-9: return 0.5
    z = (atr/rng)  # relative expansion
    return float(max(0.0, min(1.0, 0.4 + 0.6*np.tanh(3*(z-0.8)))))

def _support_resistance(bars: List[Dict]) -> float:
    if len(bars) < 8: return 0.5
    closes = np.array([b["close"] for b in bars], dtype=float)
    last = closes[-1]
    win = closes[-8:]
    sup = np.min(win)
    res = np.max(win)
    dist_sup = abs(last - sup) / max(1e-9, res - sup)
    dist_res = abs(res - last) / max(1e-9, res - sup)
    # closer to support (for bull) or to resistance (for bear) → tune later; neutral average
    return float(1.0 - min(dist_sup, dist_res))

def _momentum_divergence(_: List[Dict]) -> float:
    # Hard to get momentum from image alone; keep neutral
    return 0.5

def grade_signal(bars: List[Dict]) -> Dict:
    """
    Returns: {score: 0..100, label: 'HIGH/MED/LOW', breakdown:{...}}
    """
    if not bars:
        return {"score": 0.0, "label": "LOW", "breakdown": {"empty": 1.0}}
    w = {
        "pattern": 0.15,
        "trend":   0.15,
        "volume":  0.10,
        "vol":     0.10,
        "mtf":     0.10,   # placeholder → treat = trend for now
        "sr":      0.10,
        "mom":     0.10,
        "hist":    0.10,   # placeholder
        "rr":      0.05,
        "regime":  0.05
    }
    # Compute partials
    pattern = _pattern_strength(bars)
    trend   = _trend_context(bars)
    volume  = _volume_confirm(bars)
    volty   = _atr_volatility(bars)
    sr      = _support_resistance(bars)
    mom     = _momentum_divergence(bars)
    # neutral placeholders for factors we can wire later
    mtf = trend
    hist = 0.5
    rr   = 0.5
    regime = 0.5

    total = (
        w["pattern"]*pattern + w["trend"]*trend + w["volume"]*volume + w["vol"]*volty +
        w["mtf"]*mtf + w["sr"]*sr + w["mom"]*mom + w["hist"]*hist + w["rr"]*rr + w["regime"]*regime
    )
    score = float(round(total*100, 2))
    label = "HIGH" if score >= 85 else "MED" if score >= 60 else "LOW"
    return {
        "score": score,
        "label": label,
        "breakdown": {
            "pattern": round(pattern,3),
            "trend": round(trend,3),
            "volume": round(volume,3),
            "volatility": round(volty,3),
            "mtf": round(mtf,3),
            "support_resistance": round(sr,3),
            "momentum": round(mom,3),
            "history": hist,
            "risk_reward": rr,
            "regime": regime
        }
    }
