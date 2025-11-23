from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from .base import Base

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    client_order_id = Column(String, unique=True, nullable=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)             # buy/sell
    qty = Column(Float, nullable=False)
    notional = Column(Float, nullable=True)
    type = Column(String, default="market")           # market/limit/stop
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    time_in_force = Column(String, default="day")
    status = Column(String, default="accepted")       # accepted/filled/canceled/rejected
    mode = Column(String, default="dry-run")          # dry-run/paper/live
    placed_at = Column(DateTime(timezone=True), server_default=func.now())
    filled_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    error = Column(String, nullable=True)
    raw_response = Column(JSON, nullable=True)
    comment = Column(String, nullable=True)