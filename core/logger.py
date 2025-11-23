# core/logger.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any


SECRET_HINTS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "ACCESS", "API", "WEBHOOK",
    "CLIENT_ID", "CLIENT_SECRET", "AUTH", "BEARER"
)


def _mask(v: str) -> str:
    if not v:
        return ""
    if len(v) <= 8:
        return "*" * len(v)
    return v[:4] + "…" + v[-4:]


class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return ct.strftime(datefmt or "%Y-%m-%dT%H:%M:%S.%fZ")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        # include extras
        for k, v in record.__dict__.items():
            if k in (
                "msg","args","levelname","levelno","pathname","filename","module",
                "exc_info","exc_text","stack_info","lineno","funcName","created",
                "msecs","relativeCreated","thread","threadName","processName","process"
            ):
                continue
            base[k] = v
        return json.dumps(base, ensure_ascii=False)


class SecretMaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for k, v in os.environ.items():
            if any(h in k.upper() for h in SECRET_HINTS):
                if v and v in msg:
                    msg = msg.replace(v, _mask(v))
        record.msg = msg
        record.args = ()
        return True


def build_logger(
    name: str = "lavish.core",
    level: str | int = "INFO",
    log_dir: Optional[Path] = None,
    jsonl_file: Optional[str] = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5
) -> logging.Logger:
    """
    Unified UTC logger with optional JSONL structured file.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()

    lvl = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(lvl)

    fmt = UTCFormatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    logger.addFilter(SecretMaskFilter())

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File logs
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        plain_path = log_dir / f"{name.replace('.','_')}.log"
        fh = RotatingFileHandler(plain_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if jsonl_file:
            jpath = log_dir / jsonl_file
            jh = RotatingFileHandler(jpath, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            jh.setLevel(lvl)
            jh.setFormatter(JSONFormatter())
            logger.addHandler(jh)

    return logger