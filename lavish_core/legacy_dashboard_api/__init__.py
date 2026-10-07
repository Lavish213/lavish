# Renamed from lavish_core/hybrid_store/ (repo audit) - this is a separate,
# unused FastAPI+SQLAlchemy+Postgres dashboard backend, unrelated to
# lavish_core/db/hybrid_store.py (the real DuckDB store the live trading
# pipeline actually uses). Same old name on two unrelated things was a
# real risk of editing/importing the wrong one. Nothing in run_ingest.py's
# import chain touches this package; only other already-orphaned files
# (auto_sync.py, lavish_gateway.py, main_launcher.py, ui.py, test_db.py,
# lavish_core/api/api_handler.py) reference it.
