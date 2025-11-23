#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Lavish Bot Master Orchestrator (Fixed, No Loops)
---------------------------------------------------------
Coordinates one full pass of major modules, then optional cloud sync.
Includes a one-time Patreon self-heal check (non-fatal).

Modules run in order with retries:
  1) lavish_core/data/fetch_all.py
  2) lavish_core/vision/master_vision.py
  3) lavish_core/vision/ensemble.py
  4) lavish_core/news/feeds_news.py
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"

MODULES: List[str] = [
    "lavish_core/data/fetch_all.py",
    "lavish_core/vision/master_vision.py",
    "lavish_core/vision/ensemble.py",
    "lavish_core/news/feeds_news.py",
]
CLOUD_SYNC = "lavish_core/vision/cloud_sync.py"
MAX_RETRIES = 2

# ==========================================
# LOGGING UTILITIES
# ==========================================

def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(LOG_DIR / "lavish_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_module(module_path: str, attempt: int = 1) -> bool:
    """
    Safely execute a Python module as a script with retries.
    """
    py = sys.executable or "python3"
    log(f"▶️ Starting: {module_path} (Attempt {attempt})")
    start_time = time.time()
    try:
        result = subprocess.run(
            [py, module_path],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = time.time() - start_time
        log(f"✅ Completed {module_path} in {duration:.2f}s")
        if result.stdout:
            log(result.stdout.strip())
        if result.stderr:
            # Some libs write warnings to stderr even on success
            log(result.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        err_out = (e.stderr or "").strip() or (e.stdout or "").strip()
        log(f"❌ Error in {module_path} (rc={e.returncode}, {duration:.2f}s): {err_out}")
        if attempt < MAX_RETRIES:
            log(f"🔁 Retrying {module_path}...")
            return run_module(module_path, attempt + 1)
        log(f"🚫 Failed after {MAX_RETRIES} attempts: {module_path}")
        return False
    except FileNotFoundError:
        log(f"❌ Not found: {module_path} — check path/filename.")
        return False
    except Exception as e:
        log(f"❌ Unexpected error in {module_path}: {e}")
        return False

# ==========================================
# OPTIONAL: Patreon self-heal (one-time, non-fatal)
# ==========================================

def _patreon_self_heal_once() -> None:
    """
    Runs a lightweight check against Patreon and attempts token refresh if needed.
    Non-fatal: logs and returns. Safe to keep even if you don't use Patreon.
    """
    try:
        from dotenv import load_dotenv  # optional
        load_dotenv()
    except Exception:
        pass

    import json
    import time as _time

    PATREON_CLIENT_ID = os.getenv("PATREON_CLIENT_ID")
    PATREON_CLIENT_SECRET = os.getenv("PATREON_CLIENT_SECRET")
    PATREON_REFRESH_TOKEN = os.getenv("PATREON_REFRESH_TOKEN")
    PATREON_ACCESS_TOKEN = os.getenv("PATREON_ACCESS_TOKEN")
    PATREON_CAMPAIGN_ID = os.getenv("PATREON_CAMPAIGN_ID")

    # If not configured, skip quietly.
    if not (PATREON_CLIENT_ID and PATREON_CLIENT_SECRET and PATREON_CAMPAIGN_ID):
        log("[AUTOFIX] Patreon not configured — skipping.")
        return

    try:
        import requests
    except Exception:
        log("[AUTOFIX] requests not installed — skipping Patreon autofix.")
        return

    def _refresh_token() -> Optional[str]:
        try:
            url = "https://www.patreon.com/api/oauth2/token"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": PATREON_REFRESH_TOKEN,
                ***REMOVED***: PATREON_CLIENT_ID,
                "***REMOVED***": PATREON_CLIENT_SECRET,
            }
            r = requests.post(url, data=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            new_access = data.get("access_token")
            new_refresh = data.get("refresh_token")

            # update .env locally if present
            env_path = ROOT / ".env"
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
                out_lines = []
                for line in lines:
                    if line.startswith("PATREON_ACCESS_TOKEN="):
                        out_lines.append(f"PATREON_ACCESS_TOKEN={new_access}")
                    elif line.startswith("PATREON_REFRESH_TOKEN="):
                        out_lines.append(f"PATREON_REFRESH_TOKEN={new_refresh}")
                    else:
                        out_lines.append(line)
                env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

            # Update process env
            if new_access:
                os.environ["PATREON_ACCESS_TOKEN"] = new_access
            if new_refresh:
                os.environ["PATREON_REFRESH_TOKEN"] = new_refresh

            log("[AUTOFIX] Patreon tokens refreshed.")
            return new_access
        except Exception as e:
            log(f"[AUTOFIX] Token refresh failed: {e}")
            return None

    def _campaign_ok(access_token: str) -> bool:
        try:
            url = f"https://www.patreon.com/api/oauth2/v2/campaigns/{PATREON_CAMPAIGN_ID}"
            hdr = {"Authorization": f"Bearer {access_token}"}
            r = requests.get(url, headers=hdr, timeout=30)
            return r.status_code == 200
        except Exception:
            return False

    try:
        # quick health check: list posts (or at least hit campaign)
        if PATREON_ACCESS_TOKEN and _campaign_ok(PATREON_ACCESS_TOKEN):
            log("[AUTOFIX] Patreon API healthy ✅")
            return

        log("[AUTOFIX] Patreon needs attention — attempting refresh…")
        new_access = _refresh_token()
        if new_access and _campaign_ok(new_access):
            log("[AUTOFIX] Patreon reconnected ✅")
            # optionally restart a listener:
            # subprocess.Popen([sys.executable, "lavish_core/patreon/poller.py"])
        else:
            log("[AUTOFIX] Patreon still not healthy after refresh.")
    except Exception as e:
        log(f"[AUTOFIX] Patreon self-heal error: {e}")

# ==========================================
# MAIN ORCHESTRATION (single pass)
# ==========================================

def run_once() -> int:
    log("🚀 Lavish Bot Master Run Started")
    log(f"Working Directory: {Path.cwd()}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"MODE: {os.getenv('MODE', 'dev')}  VISION: {os.getenv('ENABLE_VISION') or os.getenv('FORCE_ENABLE_VISION') or 'off'}")

    # Non-fatal helper
    _patreon_self_heal_once()

    success = 0
    for module in MODULES:
        # Resolve relative to repo root if needed
        path = module if Path(module).is_absolute() else str((ROOT / module).resolve())
        if run_module(path):
            success += 1

    log(f"🏁 {success}/{len(MODULES)} modules completed successfully")


    # Cloud sync (optional)
    sync_path = (ROOT / CLOUD_SYNC).resolve()
    if Path(sync_path).exists():
        log("☁️ Initiating iCloud Drive Sync…")
        try:
            rc = subprocess.run(
                [sys.executable or "python3", str(sync_path), str(DATA_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
            if rc.stdout:
                log(rc.stdout.strip())
            if rc.stderr:
                log(rc.stderr.strip())
            log("✅ Cloud sync completed.")
        except subprocess.CalledProcessError as e:
            err_out = (e.stderr or "").strip() or (e.stdout or "").strip()
            log(f"❌ Cloud sync failed (rc={e.returncode}): {err_out}")
        except Exception as e:
            log(f"❌ Cloud sync error: {e}")
    else:
        log("ℹ️ Cloud sync script not found — skipping.")

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"✨ Lavish Bot Run Finished at {end_time}")
    print("\nAll tasks finished — check logs and your outputs.\n")
    # non-zero exit if any module failed
    return 0 if success == len(MODULES) else 1

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    try:
        sys.exit(run_once())
    except KeyboardInterrupt:
        log("🟥 Run interrupted by user.")
        sys.exit(130)
    except Exception as e:
        log(f"🔥 Critical failure: {e}")
        sys.exit(2)