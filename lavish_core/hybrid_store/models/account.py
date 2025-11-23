from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .base import Base

class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, default="dry-run")          # dry-run/paper/live
    buying_power = Column(Float, default=0.0)
    equity = Column(Float, default=0.0)
    last_sync = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
