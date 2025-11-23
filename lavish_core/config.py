# --- add near the top ---
from functools import lru_cache
import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = os.getenv("ENV", "dev")
    TRADE_MODE: str = os.getenv("TRADE_MODE", "paper")   # 'dry'|'paper'|'live'
    DB_URL: str = os.getenv("DB_URL", "sqlite:///lavish_store.db")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")

    # news failover knobs
    NEWS_MIN_ROWS: int = int(os.getenv("NEWS_MIN_ROWS", "120"))
    NEWS_TIMEOUT: int = int(os.getenv("NEWS_TIMEOUT", "10"))
    NEWS_FAIL_CUTOFF: int = int(os.getenv("NEWS_FAIL_CUTOFF", "3"))  # circuit-breaker threshold

    # csv/audit (you already use these paths)
    CSV_ORDERS: Path = Path(os.getenv("CSV_ORDERS", "data/trades.csv"))
    CSV_SIGNALS: Path = Path(os.getenv("CSV_SIGNALS", "data/signal_scores.csv"))

    class Config:
        env_file = ".env"
        case_sensitive = False

_runtime_overrides = {}

def set_runtime_overrides(**kw):
    _runtime_overrides.update(kw)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    for k, v in _runtime_overrides.items():
        setattr(s, k, v)
    return s

FINNHUB_API = "d3pgd5pr01qq6ml8mnfgd3pgd5pr01qq6ml8mng0"

ALPHA_API   = "5BGGGJT1RXG1Z1PU"

FRED_API    = "4b68941ee7b5e4a5dff168c65b3051b9"
PAPER_MODE = True  # Set to False for live trading