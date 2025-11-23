# core/key_rotator.py
from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class KeyState:
    key: str
    idx: int
    # budgets
    rpm: int               # requests per minute cap
    rpd: Optional[int]     # requests per day cap
    # rolling usage
    used_minute: List[float] = field(default_factory=list)
    used_day: int = 0
    day_start: float = field(default_factory=lambda: time.time())
    # health
    fail_streak: int = 0
    open_until: float = 0.0
    last_used: float = 0.0
    last_status: Optional[int] = None

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def minute_budget_left(self) -> int:
        self._prune_minute()
        return max(0, self.rpm - len(self.used_minute))

    def day_budget_left(self) -> Optional[int]:
        if self.rpd is None:
            return None
        self._reset_day_if_needed()
        return max(0, self.rpd - self.used_day)

    def mark_use(self):
        now = time.time()
        self.used_minute.append(now)
        self._reset_day_if_needed()
        self.used_day += 1
        self.last_used = now

    def mark_fail(self, cooldown_base: float, cooldown_cap: float):
        self.fail_streak += 1
        backoff = min(cooldown_cap, cooldown_base * (2 ** min(self.fail_streak, 6)))
        backoff += random.uniform(0, 1.0)
        self.open_until = time.time() + backoff

    def mark_ok(self):
        self.fail_streak = 0
        self.open_until = 0.0

    def _prune_minute(self):
        now = time.time()
        cutoff = now - 60.0
        if self.used_minute:
            self.used_minute = [t for t in self.used_minute if t >= cutoff]

    def _reset_day_if_needed(self):
        now = time.time()
        if now - self.day_start >= 86400:
            self.day_start = now
            self.used_day = 0


class KeyRotator:
    """
    Provider-aware, health-aware key rotator.
    Picks the healthiest key under budget.
    """

    def __init__(self, cooldown_base: float = 1.5, cooldown_cap: float = 60.0):
        self.cooldown_base = cooldown_base
        self.cooldown_cap = cooldown_cap
        self.providers: Dict[str, List[KeyState]] = {}

    def register_provider(
        self,
        provider: str,
        keys: List[str],
        rpm: int,
        rpd: Optional[int] = None
    ):
        provider = provider.lower()
        states = []
        for i, k in enumerate([x for x in keys if x]):
            states.append(KeyState(key=k, idx=i, rpm=rpm, rpd=rpd))
        self.providers[provider] = states

    def has_provider(self, provider: str) -> bool:
        return provider.lower() in self.providers and len(self.providers[provider.lower()]) > 0

    def pick_key(self, provider: str) -> Tuple[Optional[str], Optional[int]]:
        provider = provider.lower()
        arr = self.providers.get(provider, [])
        if not arr:
            return None, None

        # Filter eligible keys
        eligible: List[KeyState] = []
        for ks in arr:
            if ks.is_open():
                continue
            if ks.minute_budget_left() <= 0:
                continue
            day_left = ks.day_budget_left()
            if day_left is not None and day_left <= 0:
                continue
            eligible.append(ks)

        if not eligible:
            # no keys available right now
            return None, None

        # Score: prefer least used recently, lowest fail streak
        eligible.sort(key=lambda k: (k.fail_streak, k.last_used))
        ks = eligible[0]
        ks.mark_use()
        return ks.key, ks.idx

    def record_result(
        self,
        provider: str,
        key: Optional[str],
        ok: bool,
        status: Optional[int] = None,
        retry_after: Optional[float] = None
    ):
        provider = provider.lower()
        arr = self.providers.get(provider, [])
        if not arr or not key:
            return
        ks = next((x for x in arr if x.key == key), None)
        if not ks:
            return

        ks.last_status = status
        if ok:
            ks.mark_ok()
        else:
            if retry_after:
                ks.open_until = max(ks.open_until, time.time() + retry_after)
            else:
                ks.mark_fail(self.cooldown_base, self.cooldown_cap)

    def soonest_available_in(self, provider: str) -> float:
        provider = provider.lower()
        arr = self.providers.get(provider, [])
        if not arr:
            return 1.0
        now = time.time()
        times = []
        for ks in arr:
            if ks.is_open():
                times.append(max(0.0, ks.open_until - now))
            else:
                # if budget full, wait for minute window
                if ks.minute_budget_left() <= 0:
                    ks._prune_minute()
                    if ks.used_minute:
                        times.append(max(0.0, 60.0 - (now - min(ks.used_minute))))
        return min(times) if times else 0.0

    def snapshot(self) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for p, arr in self.providers.items():
            out[p] = [
                {
                    "idx": ks.idx,
                    "open_until": ks.open_until,
                    "fail_streak": ks.fail_streak,
                    "minute_used": len(ks.used_minute),
                    "day_used": ks.used_day,
                    "last_status": ks.last_status,
                }
                for ks in arr
            ]
        return out