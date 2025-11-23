"""#cloud_sync.py - Lavish Bot iCloud Drive Sync Utility
----------------------------------------------------
This module automatically syncs any local data folder
(e.g. /data) to iCloud Drive, ensuring all logs,
datasets, and stock data are continuously backed up
and available across your devices.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

# =====================================================
# Configuration
# =====================================================

# Default iCloud Drive path
ICLOUD_BASE = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"
ICLOUD_PROJECT = ICLOUD_BASE / "LavishBot_Data"

# Max retries for slow iCloud syncs
MAX_RETRIES = 3
RETRY_DELAY = 2.5  # seconds


# =====================================================
# Helper Functions
# =====================================================

def ensure_icloud_path() -> Path:
    """
    Ensure that the iCloud Drive path exists.
    Create it if needed.
    """
    if not ICLOUD_BASE.exists():
        raise FileNotFoundError(
            f"❌ iCloud Drive not detected. Make sure iCloud Drive is enabled on your Mac.\n"
            f"Expected path: {ICLOUD_BASE}"
        )

    if not ICLOUD_PROJECT.exists():
        print(f"📂 Creating LavishBot iCloud directory at {ICLOUD_PROJECT}")
        ICLOUD_PROJECT.mkdir(parents=True, exist_ok=True)

    return ICLOUD_PROJECT


def copy_file_with_retry(src: Path, dst: Path):
    """
    Copy file with a few retries in case iCloud is slow to sync.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"☁️ Synced: {src.relative_to(Path.cwd())} → {dst.relative_to(ICLOUD_PROJECT)}")
            return
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed to sync {src}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print(f"❌ Failed to sync {src} after {MAX_RETRIES} attempts.")


def sync_folder(local_folder: str):
    """
    Sync all files from the given local folder to iCloud Drive.
    """
    local_path = Path(local_folder).resolve()
    dest_root = ensure_icloud_path()

    if not local_path.exists():
        raise FileNotFoundError(f"❌ Local folder not found: {local_path}")

    print(f"\n🚀 Starting sync from {local_path} → {dest_root}\n")

    file_count = 0
    start_time = time.time()

    for item in local_path.rglob("*"):
        if item.is_file():
            dest_file = dest_root / item.relative_to(local_path)
            copy_file_with_retry(item, dest_file)
            file_count += 1

    duration = time.time() - start_time
    print(f"\n✅ Sync complete — {file_count} files in {duration:.2f}s.")
    print(f"📦 Saved in iCloud Drive: {dest_root}")


# =====================================================
# Main Entry
# =====================================================

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"

    try:
        sync_folder(folder)
    except Exception as e:
        print(f"❌ Cloud sync failed: {e}")