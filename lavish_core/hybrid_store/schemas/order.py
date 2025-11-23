from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, Any
from datetime import datetime

class OrderCreate(BaseModel):
    symbol: str
    side: str                  # "buy" | "sell"
    qty: Optional[float] = None
    notional: Optional[float] = None
    type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None
    comment: Optional[str] = None

class OrderOut(BaseModel):
    id: int
    client_order_id: Optional[str]
    symbol: str
    side: str
    qty: float
    notional: Optional[float]
    type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    status: str
    mode: str
    placed_at: datetime
    filled_at: Optional[datetime]
    updated_at: Optional[datetime]
    error: Optional[str]
    raw_response: Optional[Any]
    comment: Optional[str]

    class Config:
        from_attributes = True
