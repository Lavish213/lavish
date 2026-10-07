"""
📍 lavish_core/hybrid_store/models/base.py
Lavish Core — SQLAlchemy Declarative Base (Master Edition)
----------------------------------------------------------
Defines the ORM base class for all Lavish models with safe initialization
and built-in metadata consistency check.
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData

# Explicit metadata naming conventions prevent Alembic conflicts
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Declarative Base
Base = declarative_base(metadata=metadata)

def verify_base(verbose: bool = True):
    """Verifies Base structure integrity."""
    try:
        assert hasattr(Base, "metadata"), "❌ Base missing metadata"
        assert isinstance(Base.metadata, MetaData), "❌ Invalid metadata type"
        if verbose:
            print("✅ Lavish Base verified — ready for ORM binding.")
        return True
    except AssertionError as e:
        raise RuntimeError(f"⚠️ Base verification failed: {e}")


if __name__ == "__main__":
    verify_base()