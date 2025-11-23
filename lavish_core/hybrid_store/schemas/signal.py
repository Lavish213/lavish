from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, Any
from datetime import datetime

class SignalCreate(BaseModel):
    source: str
    symbol: str
    action: str            # buy/sell/hold
    confidence: float = 0.5
    price_hint: Optional[float] = None
    meta: Optional[Any] = None

class SignalOut(BaseModel):
    id: int
    source: str
    symbol: str
    action: str
    confidence: float
    price_hint: Optional[float]
    meta: Optional[Any]
    created_at: datetime

    class Config:
        from_attributes = True