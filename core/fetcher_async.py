# core/fetcher_async.py
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, Optional, Tuple

from .exceptions import RateLimited, ProviderDown, BadResponse
from .key_rotator import KeyRotator
from .latency_guard import LatencyGuard


# Optional async clients
try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    httpx = None
    _HAS_HTTPX = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None
    _HAS_AIOHTTP = False

import requests


class Fetcher:
    """
    Golden network path:
      - async requests with bounded concurrency per provider
      - key rotation
      - obey Retry-After
      - exponential backoff + jitter
      - latency/error tracking
      - provider-level circuit
    """

    def __init__(
        self,
        rotator: KeyRotator,
        latency: LatencyGuard,
        logger,
        provider_limits: Optional[Dict[str, int]] = None,
        timeout: float = 12.0,
        max_retries: int = 4,
    ):
        self.rotator = rotator
        self.latency = latency
        self.log = logger
        self.timeout = timeout
        self.max_retries = max_retries

        provider_limits = provider_limits or {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {
            p.lower(): asyncio.Semaphore(n) for p, n in provider_limits.items()
        }

        self._httpx_client: Optional["httpx.AsyncClient"] = None

    def _sem(self, provider: str) -> asyncio.Semaphore:
        provider = provider.lower()
        if provider not in self.semaphores:
            self.semaphores[provider] = asyncio.Semaphore(5)
        return self.semaphores[provider]

    async def _get_client(self):
        if _HAS_HTTPX:
            if self._httpx_client is None:
                self._httpx_client = httpx.AsyncClient(timeout=self.timeout, headers={"Accept": "*/*"})
            return "httpx", self._httpx_client
        if _HAS_AIOHTTP:
            return "aiohttp", None
        return "requests", None

    def _parse_retry_after(self, headers: Dict[str, str]) -> Optional[float]:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    async def request(
        self,
        provider: str,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth_key_param: Optional[str] = None,   # e.g. "token" or "apikey"
        auth_header_name: Optional[str] = None, # e.g. "X-Api-Key"
        data: Any = None,
        json_body: Any = None,
    ) -> Tuple[int, Dict[str, str], str]:
        """
        Returns (status, headers, text).
        """
        provider = provider.lower()
        h = {"User-Agent": "LavishBot/Fetcher", "Accept": "*/*"}
        if headers:
            h.update(headers)

        sem = self._sem(provider)

        async with sem:
            if self.latency.should_open_circuit(provider):
                self.log.warning("Provider circuit open: %s", provider)
                raise ProviderDown(provider=provider, message="circuit open")

            for attempt in range(self.max_retries + 1):
                key, _ = self.rotator.pick_key(provider)
                if auth_key_param and key:
                    params = dict(params or {})
                    params[auth_key_param] = key
                if auth_header_name and key:
                    h[auth_header_name] = key

                if not key and (auth_key_param or auth_header_name):
                    self.log.warning("No key available for provider=%s", provider)

                t0 = time.perf_counter()
                ok = False
                status = None
                text = ""
                resp_headers: Dict[str, str] = {}

                try:
                    client_type, client = await self._get_client()

                    if client_type == "httpx":
                        r = await client.request(method, url, params=params, headers=h, data=data, json=json_body)
                        status = r.status_code
                        resp_headers = dict(r.headers)
                        text = r.text

                    elif client_type == "aiohttp":
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as sess:
                            async with sess.request(method, url, params=params, headers=h, data=data, json=json_body) as r:
                                status = r.status
                                resp_headers = dict(r.headers)
                                text = await r.text()

                    else:
                        # requests fallback in thread
                        def _do():
                            rr = requests.request(method, url, params=params, headers=h, data=data, json=json_body, timeout=self.timeout)
                            return rr
                        r = await asyncio.to_thread(_do)
                        status = r.status_code
                        resp_headers = dict(r.headers)
                        text = r.text

                    latency_s = time.perf_counter() - t0
                    ok = 200 <= (status or 0) < 300
                    self.latency.record(provider, latency_s, ok)

                    if status == 429:
                        ra = self._parse_retry_after(resp_headers)
                        self.rotator.record_result(provider, key, ok=False, status=429, retry_after=ra)
                        raise RateLimited(provider=provider, key_hint=(key or "")[:6], retry_after=ra, payload_snip=text[:180])

                    if not ok:
                        self.rotator.record_result(provider, key, ok=False, status=status)
                        if status and status >= 500:
                            raise ProviderDown(provider=provider, status=status, message=text[:180])
                        raise BadResponse(provider=provider, status=status or 0, url=url, payload_snip=text[:180])

                    # success
                    self.rotator.record_result(provider, key, ok=True, status=status)
                    return status or 0, resp_headers, text

                except RateLimited as e:
                    # backoff obeying retry-after if present
                    wait_s = e.retry_after if e.retry_after else min(60, 2 ** attempt)
                    wait_s += random.uniform(0, 1.0)
                    self.log.warning("429 %s attempt=%d wait=%.2fs", provider, attempt, wait_s)
                    await asyncio.sleep(wait_s)

                except ProviderDown as e:
                    wait_s = min(30, 2 ** attempt) + random.uniform(0, 1.0)
                    self.log.warning("ProviderDown %s attempt=%d wait=%.2fs", provider, attempt, wait_s)
                    await asyncio.sleep(wait_s)

                except BadResponse as e:
                    # small retry for non-429 client errors
                    wait_s = min(8, 1.5 ** attempt) + random.uniform(0, 0.5)
                    self.log.warning("BadResponse %s attempt=%d wait=%.2fs status=%s",
                                     provider, attempt, wait_s, e.status)
                    await asyncio.sleep(wait_s)

                except Exception as e:
                    self.rotator.record_result(provider, key, ok=False, status=status)
                    wait_s = min(10, 1.8 ** attempt) + random.uniform(0, 0.7)
                    self.log.warning("Unknown fetch error provider=%s attempt=%d wait=%.2fs err=%s",
                                     provider, attempt, wait_s, e)
                    await asyncio.sleep(wait_s)

            raise ProviderDown(provider=provider, message="max retries exceeded")

    async def get_json(self, provider: str, url: str, **kw) -> Any:
        status, headers, text = await self.request(provider, "GET", url, **kw)
        try:
            import json
            return json.loads(text)
        except Exception:
            raise BadResponse(provider=provider, status=status, url=url, payload_snip=text[:180])

    async def get_text(self, provider: str, url: str, **kw) -> str:
        _, _, text = await self.request(provider, "GET", url, **kw)
        return text

    async def close(self):
        if self._httpx_client is not None:
            try:
                await self._httpx_client.aclose()
            except Exception:
                pass
            self._httpx_client = None