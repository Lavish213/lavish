from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from lavish_core.hybrid_store.models.base import Base

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    qty = Column(Float, nullable=False, default=0)
    avg_price = Column(Float, nullable=True, default=0)
    market_price = Column(Float, nullable=True, default=0)
    value = Column(Float, nullable=True, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Position(id={self.id}, symbol={self.symbol}, qty={self.qty}, avg_price={self.avg_price})>"
