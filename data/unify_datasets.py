"""
Lavish Bot — Unified Dataset Builder
------------------------------------
Merges all API outputs (Polygon, Alpha Vantage, Finnhub, Crypto, etc.)
into one clean, deduplicated dataset ready for model training.

Usage:
    python data/unify_datasets.py
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"unify_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
# DIRECTORIES AND OUTPUTS
# ============================================================

CACHE_DIR = "data/cache"
OUTPUT_FILE = "data/unified_data.json"

os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================

def load_json_file(filepath):
    """Safely load JSON from a file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"⚠️ Failed to read {filepath}: {e}")
        return None

# ============================================================
# MERGING LOGIC
# ============================================================

def unify_data():
    """Combine all cached JSONs into one DataFrame."""
    all_records = []

    for file in os.listdir(CACHE_DIR):
        if not file.endswith(".json"):
            continue

        path = os.path.join(CACHE_DIR, file)
        data = load_json_file(path)

        if not data:
            continue

        symbol = os.path.splitext(file)[0]
        logging.info(f"📥 Processing {file}...")

        # Handle different JSON structures across APIs
        try:
            # Alpha Vantage style
            if isinstance(data, dict) and "Time Series (Daily)" in data:
                for date, values in data["Time Series (Daily)"].items():
                    values["symbol"] = symbol
                    values["date"] = date
                    values["source"] = "AlphaVantage"
                    all_records.append(values)

            # Polygon / Finnhub style
            elif isinstance(data, dict) and "results" in data:
                for entry in data["results"]:
                    entry["symbol"] = symbol
                    entry["source"] = data.get("status", "Polygon/Finnhub")
                    all_records.append(entry)

            # Raw list style (Crypto, etc.)
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        entry["symbol"] = symbol
                        entry["source"] = "ListFeed"
                        all_records.append(entry)

            else:
                logging.warning(f"❓ Unknown data format in {file}, skipping...")

        except Exception as e:
            logging.error(f"⚠️ Error parsing {file}: {e}")

    if not all_records:
        logging.error("❌ No valid data found to unify.")
        return None

    df = pd.DataFrame(all_records)

    # ============================================================
    # CLEANUP
    # ============================================================

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Sort by date if available
    if "date" in df.columns:
        df.sort_values("date", inplace=True, ascending=False)

    logging.info(f"🧩 Unified {after} records (removed {before - after} duplicates).")

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    df.to_json(OUTPUT_FILE, orient="records")
    logging.info(f"✅ Unified dataset saved → {OUTPUT_FILE}")
    logging.info(f"📜 Logs saved → {log_file}")

    return df

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logging.info("🚀 Starting unified dataset builder...")
    result = unify_data()

    if result is not None:
        print(f"✅ Unified {len(result)} total records saved to {OUTPUT_FILE}")
    else:
        print("❌ No unified data created.")
