"""
📍 lavish_core/config/config.py
Lavish Config — Master Edition (Error-Proof)
--------------------------------------------
Centralized configuration loader for Lavish Core.
Handles environment variables safely with defaults and validation.
"""

import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

def get_settings() -> dict:
    """
    Returns configuration values used by Lavish Core.
    Automatically pulls from environment with safe fallbacks.
    """
    db_url = os.getenv("DB_URL", "sqlite:///lavish_store.db")
    debug_raw = os.getenv("DEBUG", "false").strip().lower()
    debug_mode = debug_raw in ("1", "true", "yes", "on")

    settings = {
        "DB_URL": db_url,
        "DEBUG": debug_mode
    }

    # Sanity check
    if not db_url:
        raise ValueError("❌ Missing DB_URL — cannot initialize database.")

    return settings


if __name__ == "__main__":
    s = get_settings()
    print("✅ Lavish Settings Loaded:")
    for k, v in s.items():
        print(f"   {k} = {v}")