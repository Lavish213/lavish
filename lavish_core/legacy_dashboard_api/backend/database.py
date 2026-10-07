"""
📍 lavish_core/hybrid_store/backend/database.py
Lavish Hybrid Store — Database Connector (Master Edition)
---------------------------------------------------------
Production-safe ORM engine and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lavish_core.config.config import get_settings
from lavish_core.legacy_dashboard_api.models.base import Base

# ─────────────────────────────────────────────────────────────
# 1. Load Configuration
# ─────────────────────────────────────────────────────────────
try:
    settings = get_settings()
    DB_URL = settings.get("DB_URL")
    DEBUG_MODE = settings.get("DEBUG", False)
except Exception as e:
    raise RuntimeError(f"⚠️ Failed to load settings: {e}")

if not DB_URL:
    raise RuntimeError("❌ Database URL not provided in configuration.")


# ─────────────────────────────────────────────────────────────
# 2. Create Engine
# ─────────────────────────────────────────────────────────────
try:
    engine = create_engine(
        DB_URL,
        echo=DEBUG_MODE,
        pool_pre_ping=True,       # Detect broken connections automatically
        pool_recycle=1800,        # Recycle every 30 mins
        future=True
    )
except Exception as e:
    raise RuntimeError(f"❌ Failed to create database engine: {e}")


# ─────────────────────────────────────────────────────────────
# 3. Session Factory
# ─────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True
)


# ─────────────────────────────────────────────────────────────
# 4. ORM Initialization
# ─────────────────────────────────────────────────────────────
def init_db(verify: bool = True):
    """Initializes ORM and creates all tables."""
    print("🧩 Initializing Lavish database...")
    try:
        Base.metadata.create_all(bind=engine)
        if verify:
            tables = list(Base.metadata.tables.keys())
            print(f"✅ ORM ready with {len(tables)} tables: {tables if tables else 'No models yet.'}")
    except Exception as e:
        raise RuntimeError(f"❌ Database initialization failed: {e}")


# ─────────────────────────────────────────────────────────────
# 5. Safe Session Context
# ─────────────────────────────────────────────────────────────
def get_session():
    """Context-managed session provider."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────
# 6. Manual Verification Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧠 Verifying Lavish ORM connection...")
    init_db()
    print("✅ Database connector verified and ready.")