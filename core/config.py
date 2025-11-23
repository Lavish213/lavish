import os
from dotenv import load_dotenv
from pathlib import Path

# === Load Environment Variables ===
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def get_keys(prefix):
    """Return all environment variables that start with a prefix (e.g. POLYGON_API_KEY)."""
    return [v for k, v in os.environ.items() if k.startswith(prefix) and v]

# === API KEYS (Multiple Key Support for Rate Limits) ===
POLYGON_KEYS = get_keys("POLYGON_API_KEY")  # includes all polygon keys & backups
FINNHUB_KEYS = get_keys("FINNHUB_API_KEY")  # includes all finnhub keys & backups
ALPHA_VANTAGE_KEYS = get_keys("ALPHA_VANTAGE_API_KEY")
FRED_KEYS = get_keys("FRED_API_KEY")
NEWSAPI_KEYS = get_keys("NEWSAPI_KEY")
COINMARKETCAP_KEYS = get_keys("COINMARKETCAP_API_KEY")

# === Brokerage / Trading ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# === Discord & Webhook ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# === OpenAI (for Chat/AI integration) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === General Config ===
BOT_REFRESH_INTERVAL_MIN = int(os.getenv("BOT_REFRESH_INTERVAL_MIN", 60))
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# === Paths ===
DATA_DIR = Path('.') / 'data' / 'storage'
LOG_DIR = Path('.') / 'logs'
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Key Rotation Helper ===
def rotate_key(keys, index):
    """Return the next key for a given service (for rate-limit handling)."""
    if not keys:
        return None
    return keys[index % len(keys)]

# === Sanity Check Output ===
print("🔑 Loaded API keys:")
print(f"   Polygon: {len(POLYGON_KEYS)}")
print(f"   Finnhub: {len(FINNHUB_KEYS)}")
print(f"   Alpha Vantage: {len(ALPHA_VANTAGE_KEYS)}")
print(f"   FRED: {len(FRED_KEYS)}")
print(f"   NewsAPI: {len(NEWSAPI_KEYS)}")
print(f"   CoinMarketCap: {len(COINMARKETCAP_KEYS)}")