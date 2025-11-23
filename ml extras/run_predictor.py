import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

CACHE_DIR = "data/cache"
ENRICHED_FILE = os.path.join(CACHE_DIR, "enriched_data.json")

def load_enriched_data():
    """Load the latest enriched dataset."""
    if not os.path.exists(ENRICHED_FILE):
        print("❌ No enriched data found. Run enricher.py first.")
        return None

    with open(ENRICHED_FILE, "r") as f:
        data = json.load(f)
    print(f"📂 Loaded {len(data)} records from {ENRICHED_FILE}")
    return pd.DataFrame(data)

def analyze_trends(df):
    """Perform basic trend prediction using linear regression."""
    if df.empty:
        print("❌ No data to analyze.")
        return None

    # Example: use 'close' prices if they exist
    if 'results.0.c' in df.columns:
        y = df['results.0.c'].values
        X = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)
        trend = model.coef_[0]

        direction = "📈 Uptrend" if trend > 0 else "📉 Downtrend"
        print(f"📊 Linear trend: {direction} (slope = {trend:.4f})")

        # Predict next value
        next_pred = model.predict([[len(y)]])[0]
        print(f"🔮 Predicted next close: {next_pred:.2f}")
        return {"trend": direction, "next_pred": next_pred}
    else:
        print("⚠️ No close price data found in dataset.")
        return None

def save_predictions(predictions):
    """Save prediction results."""
    out_file = os.path.join(CACHE_DIR, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"💾 Saved predictions to {out_file}")

if __name__ == "__main__":
    df = load_enriched_data()
    if df is not None:
        results = analyze_trends(df)
        if results:
            save_predictions(results)