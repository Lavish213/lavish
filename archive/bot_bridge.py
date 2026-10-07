# bot_bridge.py  (patched)
from __future__ import annotations
from lavish_core.logger_setup import get_logger
log = get_logger("bot_bridge", log_dir="logs")

# Optional dependency: super_vector_sql
try:
    import super_vector_sql  # type: ignore
except Exception:
    super_vector_sql = None
    log.info("super_vector_sql not installed. Vector features disabled.")

# Ensure df_ohlcv is defined to silence pylance warnings if referenced
def df_ohlcv(*_args, **_kwargs):
    """Placeholder OHLCV loader; replace with your real implementation."""
    log.warning("df_ohlcv placeholder called. Provide a real implementation.")
    return None