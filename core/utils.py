# core/utils.py
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
DEFAULT_STOP = {"THE","AND","FOR","WITH","FROM","ABOUT","HTTPS","HTTP","WWW","NEWS","MARKET"}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def infer_tickers(text: str, stop: Optional[set] = None, max_n: int = 8) -> List[str]:
    stop = stop or DEFAULT_STOP
    found = {t for t in TICKER_RE.findall(text.upper()) if t not in stop}
    return sorted(found)[:max_n]


def safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_json_save(path: Path, obj: Any, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def dedup_by_id(items: Iterable[Dict[str, Any]], id_field: str = "id") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        nid = it.get(id_field)
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out.append(it)
    return out