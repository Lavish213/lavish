from lavish_core.hybrid_store.backend.database import engine, Base
from lavish_core.hybrid_store.models.position import Position
from sqlalchemy.orm import Session

# Drop & recreate table to sync structure
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Insert a test record
with Session(engine) as session:
    new_pos = Position(symbol="AAPL", qty=10, avg_price=190.25)
    session.add(new_pos)
    session.commit()

    results = session.query(Position).all()
    for r in results:
        print(r.id, r.symbol, r.qty, r.avg_price)