from fastapi import APIRouter
from ..config import get_settings

router = APIRouter(prefix="/config", tags=["config"])

@router.get("")
def read_config():
    s = get_settings()
    return {
        "env": s.ENV,
        "mode": s.TRADE_MODE,
        "db_url": s.DB_URL,
        "alpaca_base_url": s.ALPACA_BASE_URL,
        "logs": {"orders": str(s.CSV_ORDERS), "signals": str(s.CSV_SIGNALS)},}