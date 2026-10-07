from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from datetime import datetime

class AccountOut(BaseModel):
    id: int
    mode: str
    buying_power: float
    equity: float
    last_sync: Optional[datetime]

    class Config:
        from_attributes = True
