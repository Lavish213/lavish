# lavish_core/patreon/trigger.py
from __future__ import annotations
import os, time, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from lavish_core.logger_setup import get_logger
from lavish_core.trading.alert_handler import handle_alert_text
from lavish_core.patreon.patreon_refresh import refresh_patreon_token

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

    seen: set[str] = set()
    base = f"{API}/campaigns/{cid}/posts?fields[post]=title,content,created_at,post_type&page[count]=10&sort=-created"

    while True:
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
                    _handle_post(post)
        except Exception as e:
            log.error("patreon_poll_error: %s", e)
        time.sleep(POLL_SECONDS)

def main():
    poll_loop()

if __name__ == "__main__":
    main()