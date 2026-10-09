# lavish_core/patreon/trigger.py
from __future__ import annotations
import os, time, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from lavish_core.logger_setup import get_logger
from lavish_core.trading.alert_handler import handle_alert_text
from lavish_core.patreon.patreon_refresh import refresh_patreon_token
from lavish_core.db.hybrid_store import HybridStore, DEFAULT_DB

ROOT = Path(__file__).resolve().parents[1]
VISION_RAW = ROOT / "vision" / "raw"
LOG_DIR = ROOT / "patreon" / "logs"
VISION_RAW.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger("patreon.trigger", log_dir=str(LOG_DIR))

API = "https://www.patreon.com/api/oauth2/v2"
ACCESS   = os.getenv("PATREON_ACCESS_TOKEN", "")
CAMPAIGN = os.getenv("PATREON_CAMPAIGN_ID", "")
POLL_SECONDS = int(os.getenv("PATREON_POLL_SECONDS", "20"))
VISION_AUTO  = os.getenv("VISION_AUTO", "true").strip().lower() in ("1","true","yes","on")

# A post older than this when first seen is stale enough that acting on it
# isn't "copying a fresh alert" anymore (e.g. the bot was down and this is
# catching up on a backlog) - skip trading it, but still mark it seen so
# it isn't retried forever. Generous default since Patreon's own feed can
# lag; tightened per-deployment if latency is confirmed better than this.
MAX_ALERT_AGE_SECONDS = int(os.getenv("PATREON_MAX_ALERT_AGE_SECONDS", "900"))

_INGEST_SOURCE = "patreon"

def _headers(tok: Optional[str]=None) -> Dict[str, str]:
    return {"Authorization": f"Bearer {tok or ACCESS}"}

def _ensure_campaign_id() -> str:
    global CAMPAIGN
    if CAMPAIGN:
        return CAMPAIGN
    url = f"{API}/identity?include=memberships.campaign"
    r = requests.get(url, headers=_headers(), timeout=20)
    if r.status_code == 401:
        new_tok = refresh_patreon_token()
        if not new_tok:
            raise RuntimeError("Unable to refresh Patreon token")
        r = requests.get(url, headers=_headers(new_tok), timeout=20)
    r.raise_for_status()
    data = r.json()
    for inc in data.get("included", []):
        if inc.get("type") == "campaign":
            CAMPAIGN = inc.get("id")
            os.environ["PATREON_CAMPAIGN_ID"] = CAMPAIGN
            log.info(f"Patreon campaign id resolved: {CAMPAIGN}")
            return CAMPAIGN
    raise RuntimeError("No Patreon campaign found on identity")

def _download_images_from_post(post: Dict[str, Any]) -> List[Path]:
    saved: List[Path] = []
    attrs = post.get("attributes", {})
    html = (attrs.get("content") or "")
    for m in re.finditer(r'src="([^"]+)"', html):
        url = m.group(1)
        if not url.lower().startswith("http"):
            continue
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and resp.content:
                fname = f"pat_{post.get('id','unknown')}_{len(saved)+1}.jpg"
                path = VISION_RAW / fname
                path.write_bytes(resp.content)
                saved.append(path)
        except Exception as e:
            log.error("img_download_error: %s : %s", url, e)
    return saved

def _post_age_seconds(post: Dict[str, Any]) -> Optional[float]:
    created_at = (post.get("attributes") or {}).get("created_at")
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def _handle_post(post: Dict[str, Any]) -> None:
    attrs = post.get("attributes", {})
    title = attrs.get("title", "") or ""
    content = (attrs.get("content") or "")
    body = f"{title}\n{content}"
    pid = post.get("id")

    # Download any attached trade-card screenshot *before* parsing, so it
    # feeds the same text+image parser Discord alerts go through - a
    # strike/expiry/stop stated only in the image (not the caption) used
    # to be silently dropped since nothing OCR'd it.
    image_path = None
    if VISION_AUTO:
        imgs = _download_images_from_post(post)
        if imgs:
            image_path = str(imgs[0])
            log.info("Saved %d Patreon image(s) → %s", len(imgs), VISION_RAW)

    patron_dollars = os.getenv("PATRON_DOLLARS")
    handle_alert_text(
        text=body,
        image_path=image_path,
        note=f"patreon:{pid}",
        source="patreon",
        amount_usd=float(patron_dollars) if patron_dollars else None,
    )

def poll_loop():
    global ACCESS
    cid = _ensure_campaign_id()
    log.info(f"📬 Patreon listening (campaign={cid}) — poll={POLL_SECONDS}s vision_auto={VISION_AUTO}")

    # Durable cursor (HybridStore) instead of an in-memory set - a restart
    # used to reset "seen" to empty, so the next poll's most recent posts
    # all looked brand new and got traded as if they just happened.
    store = HybridStore(duckdb_path=str(DEFAULT_DB))
    seen: set[str] = store.get_all_ingest_seen(_INGEST_SOURCE)
    first_run_seed = not seen
    if first_run_seed:
        log.warning(
            "Patreon ingest cursor is empty (first run, or a fresh DB) - the first batch of posts "
            "fetched will be marked seen WITHOUT trading them, so this doesn't replay her entire "
            "recent history as if every post just happened."
        )

    base = f"{API}/campaigns/{cid}/posts?fields[post]=title,content,created_at,post_type&page[count]=10&sort=-created"

    # A periodic "still alive" line so someone tailing logs can tell "quietly
    # healthy, no new posts" apart from "silently stopped polling" without
    # having to infer it from the absence of any log line at all.
    heartbeat_every = max(1, int(900 // max(1, POLL_SECONDS)))  # ~every 15 min
    loop_count = 0

    while True:
        loop_count += 1
        if loop_count % heartbeat_every == 0:
            log.info("Patreon poller heartbeat: alive, %d post(s) seen total.", len(seen))
        try:
            r = requests.get(base, headers=_headers(), timeout=25)
            if r.status_code == 401:
                log.warning("401 from Patreon — refreshing token…")
                new_tok = refresh_patreon_token()
                if new_tok:
                    ACCESS = new_tok
                    r = requests.get(base, headers=_headers(), timeout=25)
            if r.status_code != 200:
                log.warning("Patreon poll error %s: %s", r.status_code, r.text[:200])
            else:
                data = r.json().get("data", [])
                for post in data:
                    pid = post.get("id")
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    store.mark_ingest_seen(_INGEST_SOURCE, pid)

                    if first_run_seed:
                        continue  # cursor-seeding batch - mark seen, don't trade

                    age = _post_age_seconds(post)
                    if age is not None and age > MAX_ALERT_AGE_SECONDS:
                        log.warning("Skipping stale Patreon post %s (%.0fs old, > %ds max) - not trading it.",
                                     pid, age, MAX_ALERT_AGE_SECONDS)
                        continue

                    _handle_post(post)
                first_run_seed = False  # only the very first successful poll is the seed batch
        except Exception as e:
            log.error("patreon_poll_error: %s", e)
        time.sleep(POLL_SECONDS)

def main():
    poll_loop()

if __name__ == "__main__":
    main()