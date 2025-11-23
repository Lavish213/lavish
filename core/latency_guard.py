# core/latency_guard.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class LatencyStats:
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=400))
    errors: Deque[int] = field(default_factory=lambda: deque(maxlen=400))
    last_ts: float = 0.0

    def record(self, latency_s: float, ok: bool):
        self.samples.append(max(0.0, float(latency_s)))
        self.errors.append(0 if ok else 1)
        self.last_ts = time.time()

    def pct(self, q: float) -> Optional[float]:
        if not self.samples:
            return None
        arr = sorted(self.samples)
        i = int(q * (len(arr) - 1))
        return arr[i]

    def error_rate(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)


class LatencyGuard:
    """
    Tracks rolling latency/error stats per provider.
    Provides weights to bias fastest providers.
    """

    def __init__(self, error_trip: float = 0.33, p90_trip: float = 6.0):
        self.error_trip = error_trip   # open provider circuit if too many errors
        self.p90_trip = p90_trip       # open circuit if p90 latency too high
        self.stats: Dict[str, LatencyStats] = {}

    def record(self, provider: str, latency_s: float, ok: bool):
        provider = provider.lower()
        st = self.stats.setdefault(provider, LatencyStats())
        st.record(latency_s, ok)

    def get(self, provider: str) -> LatencyStats:
        return self.stats.setdefault(provider.lower(), LatencyStats())

    def snapshot(self) -> Dict[str, dict]:
        out = {}
        for p, st in self.stats.items():
            out[p] = {
                "p50": st.pct(0.50),
                "p90": st.pct(0.90),
                "p99": st.pct(0.99),
                "error_rate": st.error_rate(),
                "last_ts": st.last_ts,
            }
        return out

    def weight(self, provider: str) -> float:
        st = self.get(provider)
        p90 = st.pct(0.90) or 1.0
        er = st.error_rate()
        # Lower latency and lower error => higher weight
        w = 1.0 / (1.0 + p90) * (1.0 - min(er, 0.9))
        return max(0.05, min(1.5, w))

    def should_open_circuit(self, provider: str) -> bool:
        st = self.get(provider)
        if (st.pct(0.90) or 0.0) > self.p90_trip:
            return True
        if st.error_rate() > self.error_trip:
            return True
        return False