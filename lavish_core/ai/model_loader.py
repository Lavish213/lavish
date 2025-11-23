# lavish_bot/ai/model_loader.py

import joblib
import os

def load_model():
    """
    Load the trained prediction model from disk.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None