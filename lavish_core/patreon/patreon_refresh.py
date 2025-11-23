# lavish_core/patreon/patreon_refresh.py
import os
import json
import requests
from dotenv import load_dotenv
from lavish_core.logger_setup import get_logger

log = get_logger("patreon_refresh", log_dir="lavish_core/patreon/logs")
load_dotenv()

PATREON_TOKEN_URL = "https://www.patreon.com/api/oauth2/token"

def refresh_patreon_token():
    client_id = os.getenv("PATREON_CLIENT_ID")
    ***REMOVED*** = os.getenv("PATREON_CLIENT_SECRET")
    refresh_token = os.getenv("PATREON_REFRESH_TOKEN")
    if not all([client_id, ***REMOVED***, refresh_token]):
        log.warning("⚠️ Missing Patreon credentials in env.")
        return None

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        ***REMOVED***: client_id,
        "***REMOVED***": ***REMOVED***,
    }

    try:
        res = requests.post(PATREON_TOKEN_URL, data=payload, timeout=15)
    except requests.RequestException as e:
        log.warning("⚠️ Patreon token refresh failed: %s", e)
        return None

    try:
        data = res.json()
    except Exception:
        data = {}
        log.warning("⚠️ Non-JSON response from Patreon: %s", res.text[:200])

    if res.status_code == 401 or data.get("error") == "invalid_grant":
        log.warning("🔒 Refresh rejected (invalid_grant/401). Re-auth required.")
        return None
    if "error" in data:
        log.warning("⚠️ Patreon refresh error: %s", data)
        return None

    new_access = data.get("access_token")
    new_refresh = data.get("refresh_token")
    if not new_access or not new_refresh:
        log.warning("⚠️ Patreon response missing tokens: %s", data)
        return None

    os.environ["PATREON_ACCESS_TOKEN"] = new_access
    os.environ["PATREON_REFRESH_TOKEN"] = new_refresh
    _update_env("PATREON_ACCESS_TOKEN", new_access)
    _update_env("PATREON_REFRESH_TOKEN", new_refresh)
    log.info("✅ Patreon tokens refreshed.")
    return new_access

def _update_env(key: str, value: str) -> None:
    env_file = ".env"
    lines = []
    found = False
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as f:
        f.writelines(lines)