# core/exceptions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict


class LavishError(Exception):
    """Base exception for Lavish core."""


@dataclass
class ConfigError(LavishError):
    message: str
    missing_key: Optional[str] = None

    def __str__(self) -> str:
        if self.missing_key:
            return f"{self.message} (missing={self.missing_key})"
        return self.message


@dataclass
class RateLimited(LavishError):
    provider: str
    key_hint: str = ""
    retry_after: Optional[float] = None
    status: int = 429
    payload_snip: str = ""

    def __str__(self) -> str:
        ra = f", retry_after={self.retry_after}s" if self.retry_after else ""
        kh = f", key={self.key_hint}" if self.key_hint else ""
        return f"RateLimited(provider={self.provider}{kh}{ra})"


@dataclass
class ProviderDown(LavishError):
    provider: str
    status: Optional[int] = None
    message: str = ""

    def __str__(self) -> str:
        st = f" status={self.status}" if self.status is not None else ""
        msg = f" msg={self.message}" if self.message else ""
        return f"ProviderDown(provider={self.provider}{st}{msg})"


@dataclass
class BadResponse(LavishError):
    provider: str
    status: int
    url: str
    payload_snip: str = ""
    extra: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        snip = f" payload='{self.payload_snip[:160]}'" if self.payload_snip else ""
        return f"BadResponse(provider={self.provider}, status={self.status}, url={self.url}{snip})"