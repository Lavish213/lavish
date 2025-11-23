from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from ..database.db import get_db
from ..schemas.position import PositionOut
from ..models.position import Position

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

@router.get("/positions", response_model=List[PositionOut])
def list_positions(db: Session = Depends(get_db)):
    return db.query(Position).order_by(Position.symbol.asc()).all()
