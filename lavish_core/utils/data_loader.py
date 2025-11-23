# lavish_bo# lavish_bot/utils/data_loader.py

import os
import pandas as pd

def load_enriched_data():
    """
    Load the enriched dataset for predictions.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "enriched_data.csv")

    if not os.path.exists(data_path):
        print(f"⚠️ Data file not found at {data_path}")
        return None

    try:
        df = pd.read_csv(data_path)
        print("✅ Enriched data loaded successfully.")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None