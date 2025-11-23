from __future__ import annotations

def bounded(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def usd_to_qty(usd: float, price: float) -> int:
    if price <= 0: return 0
    return max(1, int(usd / price))