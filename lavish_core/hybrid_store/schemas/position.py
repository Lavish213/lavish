from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from datetime import datetime

class PositionOut(BaseModel):
    id: int
    symbol: str
    qty: float
    avg_price: float
    market_price: Optional[float]
    value: Optional[float]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True