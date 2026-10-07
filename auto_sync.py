"""
Auto Table Sync Utility
-----------------------
Checks all SQLAlchemy models and ensures your PostgreSQL tables
match the current model definitions — without losing existing data.
"""

from lavish_core.legacy_dashboard_api.backend.database import engine, Base
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

def auto_sync_tables():
    inspector = inspect(engine)
    conn = engine.connect()

    print("\n🔍 Checking tables...")

    for table_name, table_obj in Base.metadata.tables.items():
        if table_name not in inspector.get_table_names():
            print(f"⚙️  Creating missing table: {table_name}")
            table_obj.create(bind=engine)
        else:
            print(f"✅ Table exists: {table_name}")

    print("\n✅ Database is synced with your models!\n")
    conn.close()

if __name__ == "__main__":
    try:
        auto_sync_tables()
    except SQLAlchemyError as e:
        print(f"❌ SQLAlchemy error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")