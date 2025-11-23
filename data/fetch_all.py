"""
Lavish Bot — Master Data Fetcher (Auto-Unify Enabled)
-----------------------------------------------------
Runs all API feeds, saves logs, and automatically unifies datasets
once all fetches are complete.

Usage:
    python data/fetch_all.py
"""

import os
import time
import logging
import subprocess
from datetime import datetime

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"fetch_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# ============================================================
# FETCH TASKS
# ============================================================

def run_feed(script_path, name, attempt=1):
    """Run a single feed script and log its result."""
    logging.info(f"🟢 Running {name} feed (attempt {attempt})...")

    try:
        start_time = time.time()
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        duration = time.time() - start_time

        if result.returncode == 0:
            logging.info(f"✅ {name} feed completed successfully in {duration:.2f}s.")
        else:
            logging.error(f"❌ {name} feed failed with exit code {result.returncode}.")
            logging.error(result.stderr)

    except Exception as e:
        logging.error(f"⚠️ Error running {name} feed: {e}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    total_start = time.time()
    logging.info("🚀 Starting all data feeds...")

    feeds = [
        ("data/fetch_polygon.py", "Polygon"),
        ("data/fetch_finnhub.py", "Finnhub"),
        ("data/fetch_crypto.py", "Crypto"),
        ("data/fetch_alpha_vantage.py", "AlphaVantage"),  # optional if you have this one
    ]

    for script_path, name in feeds:
        if os.path.exists(script_path):
            run_feed(script_path, name)
        else:
            logging.warning(f"⚠️ Skipping {name}: {script_path} not found.")

    total_time = time.time() - total_start
    logging.info(f"🏁 All feeds completed in {total_time:.2f}s.")
    logging.info("🪵 Logs saved in /logs")

    # ============================================================
    # 🔁 AUTO-UNIFY ADDITION (NEW SECTION)
    # ============================================================
    # After all feeds finish, this triggers the dataset merger automatically.
    # ------------------------------------------------------------
    unify_script = "data/unify_datasets.py"
    if os.path.exists(unify_script):
        logging.info("🔗 All feeds done — launching unified dataset builder...")
        try:
            unify_start = time.time()
            result = subprocess.run(["python", unify_script], capture_output=True, text=True)

            if result.returncode == 0:
                unify_duration = time.time() - unify_start
                logging.info(f"✅ Unified dataset completed in {unify_duration:.2f}s.")
                logging.info(result.stdout)
            else:
                logging.error("❌ Dataset unification failed.")
                logging.error(result.stderr)

        except Exception as e:
            logging.error(f"⚠️ Error running unify_datasets.py: {e}")
    else:
        logging.warning("⚠️ unify_datasets.py not found. Skipping auto-merge.")

    logging.info("✨ All tasks finished successfully.")
