# lavish_core/patreon/poller.py
import os
import re
import asyncio
import aiohttp
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv

from lavish_core.trading.trade_handler import execute_trade_from_post
from lavish_core.patreon.patreon_refresh import refresh_patreon_token

load_dotenv()
PATREON_TOKEN = os.getenv("PATREON_ACCESS_TOKEN", "")
CAMPAIGN_ID = os.getenv("PATREON_CAMPAIGN_ID", "")
POLL_SEC = int(os.getenv("PATREON_POLL_SEC", "60"))
BASE_URL = "https://www.patreon.com/api/oauth2/v2"

logger = logging.getLogger("lavish.patreon")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | Patreon | %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

SYM_RE = re.compile(r"\b([A-Z]{1,5})\b")
def _parse_signal(text: str) -> Optional[Tuple[str, str]]:
    t = (text or "").lower()
    action = "BUY" if "buy" in t or "long" in t else "SELL" if ("sell" in t or "short" in t) else ""
    m = SYM_RE.search(text or "")
    symbol = m.group(1).upper() if m else ""
    if action and symbol:
        return action, symbol
    return None

async def get_latest_post_text(session: aiohttp.ClientSession, token: str) -> Optional[Tuple[str, datetime]]:
    if not (token and CAMPAIGN_ID):
        logger.warning("Missing Patreon token or campaign ID.")
        return None

    url = (
        f"{BASE_URL}/campaigns/{CAMPAIGN_ID}/posts"
        "?fields[post]=title,content,published_at"
        "&sort=-published_at&page[count]=1"
    )
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "LavishBot/2.0 (+https://lavish.invalid)"}

    async with session.get(url, headers=headers, timeout=12) as res:
        if res.status == 401:
            return ("__TOKEN_EXPIRED__", datetime.now(timezone.utc))
        if res.status != 200:
            txt = await res.text()
            logger.warning("API %s: %s", res.status, txt[:200])
            return None
        js = await res.json()
        posts = js.get("data", [])
        if not posts:
            return None
        attrs = posts[0].get("attributes", {})
        title = (attrs.get("title") or "").strip()
        content = (attrs.get("content") or "").strip()
        created_at = attrs.get("published_at") or attrs.get("created_at")
        created_dt = None
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_dt = datetime.now(timezone.utc)
        return (f"{title}\n{content}".strip(), created_dt or datetime.now(timezone.utc))

async def run_patreon_loop():
    token = PATREON_TOKEN
    last_timestamp: Optional[datetime] = None
    failure_streak = 0
    logger.info("📣 Patreon polling started (async)…")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                got = await get_latest_post_text(session, token)
                if got:
                    text, created_at = got
                    if text == "__TOKEN_EXPIRED__":
                        logger.warning("401 received — refreshing token…")
                        new_tok = refresh_patreon_token()
                        if new_tok:
                            token = new_tok
                        await asyncio.sleep(2)
                        continue

                    if not last_timestamp or (created_at and created_at > last_timestamp):
                        last_timestamp = created_at
                        logger.info("🆕 New post @ %s", created_at.isoformat())
                        parsed = _parse_signal(text)
                        if parsed:
                            action, symbol = parsed
                            execute_trade_from_post({
                                "source": "patreon",
                                "action": action,
                                "symbol": symbol,
                                "confidence": float(os.getenv("DEFAULT_PATRON_CONF", "0.8")),
                                "amount_usd": os.getenv("PATRON_DOLLARS"),
                                "note": f"patreon:{created_at.isoformat()}",
                            })
                        else:
                            logger.info("ℹ️ No trade signal in latest post.")
                        failure_streak = 0
                    else:
                        # No newer post yet
                        failure_streak = 0
                else:
                    failure_streak += 1
                    if failure_streak >= 3:
                        wait = min(POLL_SEC * 3, 300)
                        logger.warning("⚠️ Repeated fetch failures — backing off %ss", wait)
                        await asyncio.sleep(wait)
                        failure_streak = 0
                await asyncio.sleep(POLL_SEC)
            except asyncio.TimeoutError:
                logger.warning("Timeout; continuing.")
            except Exception as e:
                logger.exception("Unexpected poller error: %s", e)
                await asyncio.sleep(POLL_SEC)

def start_patreon_monitor():
    try:
        asyncio.run(run_patreon_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Patreon polling stopped by user.")

if __name__ == "__main__":
    start_patreon_monitor()