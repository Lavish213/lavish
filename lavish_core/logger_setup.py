# lavish_core/logger_setup.py
from __future__ import annotations
import logging, os, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    # This logger gets its own stdout+file handlers below - without
    # propagate=False, the message also travels up to the root logger,
    # which configure_root_logging() (or trade_agent.py's import-time call
    # to it) gives its own stdout handler too, so every line printed twice.
    # Modules that just do logging.getLogger(name) with no handlers of
    # their own (broker_alpaca, circuit_breaker, reconcile, etc.) are
    # unaffected - they rely on propagation to root and should keep it.
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # add tiny CSV helper
    def csv_line(event: str, data: str, level: str = "INFO"):
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts},{event},{data}"
        getattr(logger, level.lower(), logger.info)(line)

    logger.csv_line = csv_line  # type: ignore[attr-defined]
    return logger


_root_configured = False

def configure_root_logging(log_dir: str | Path = "logs", level: Optional[str] = None) -> None:
    """
    Configures the ROOT logger once, with both a stdout handler and a file
    handler (logs/lavish.log). Modules that call plain logging.getLogger(name)
    without ever calling get_logger() above - broker_alpaca, circuit_breaker,
    reconcile, options_broker, options_exit_monitor, portfolio_risk,
    account_monitor, trade_agent - have no handlers of their own and
    propagate here by default. Previously only trade_agent.py's
    logging.basicConfig() (stdout-only, no file) happened to configure root
    as a side effect of being imported - meaning circuit breaker trips,
    reconciliation recoveries, and every real broker order/error from those
    modules were never written to a log file at all, only container
    stdout. Idempotent - safe to call from multiple modules/entrypoints.
    """
    global _root_configured
    if _root_configured:
        return
    _root_configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "lavish.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)