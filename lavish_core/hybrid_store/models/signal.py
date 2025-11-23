from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .base import Base

class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)          # "vision", "patreon", "news", "manual"
    symbol = Column(String, index=True)
    action = Column(String)                      # "buy", "sell", "hold"
    confidence = Column(Float, default=0.5)      # 0..1
    price_hint = Column(Float, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())