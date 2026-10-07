from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..database.db import get_db
from ..schemas.order import OrderCreate, OrderOut
from ..models.order import Order
from ..services.order_service import place_order, list_orders

router = APIRouter(prefix="/orders", tags=["orders"])

@router.post("", response_model=OrderOut)
def create_order(payload: OrderCreate, db: Session = Depends(get_db)):
    if not payload.qty and not payload.notional:
        raise HTTPException(status_code=400, detail="Provide qty or notional")
    order = place_order(
        db,
        symbol=payload.symbol,
        side=payload.side,
        qty=payload.qty,
        notional=payload.notional,
        type=payload.type,
        time_in_force=payload.time_in_force,
        limit_price=payload.limit_price,
        stop_price=payload.stop_price,
        client_order_id=payload.client_order_id,
        comment=payload.comment,
    )
    return order

@router.get("", response_model=List[OrderOut])
def get_orders(limit: int = 100, db: Session = Depends(get_db)):
    return list_orders(db, limit=limit)