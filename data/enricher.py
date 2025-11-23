import os
import json
import pandas as pd
from datetime import datetime

CACHE_DIR = "data/cache"
OUTPUT_FILE = os.path.join(CACHE_DIR, "enriched_data.json")

def load_latest_cache():
    """Load the most recent cached data file."""
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    if not files:
        print("❌ No cache files found.")
        return None

    # Sort files by modification time (latest first)
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(CACHE_DIR, f)))
    path = os.path.join(CACHE_DIR, latest)

    print(f"📂 Loading latest cache: {latest}")
    with open(path, "r") as f:
        return json.load(f)

def enrich_data(raw_data):
    """Convert, clean, and add new metrics."""
    if not raw_data:
        print("❌ No data to enrich.")
        return None

    # Convert nested JSON to flat DataFrame
    df = pd.json_normalize(raw_data)

    # Example enrichment — add derived columns
    if "results" in raw_data:
        df["result_count"] = len(raw_data["results"])
    if "ticker" in raw_data:
        df["ticker_upper"] = raw_data["ticker"].upper()

    # Add timestamp
    df["processed_at"] = datetime.now().isoformat()

    print(f"✅ Enriched {len(df)} rows.")
    return df

def save_enriched(df):
    """Save enriched DataFrame as JSON."""
    if df is not None:
        df.to_json(OUTPUT_FILE, orient="records", indent=4)
        print(f"💾 Saved enriched data to {OUTPUT_FILE}")
    else:
        print("❌ No enriched data to save.")

if __name__ == "__main__":
    raw = load_latest_cache()
    enriched = enrich_data(raw)
    save_enriched(enriched)