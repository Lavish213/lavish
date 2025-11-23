from pydantic_settings import BaseModel
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
