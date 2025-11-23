# lavish_core/patreon/normalizer.py
import os, re, json, logging, pathlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

log = logging.getLogger("patreon_normalizer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    log.addHandler(h)
log.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))

OUTPUT_PATH = pathlib.Path("lavish_core/patreon/latest_signals.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SYM_RE = re.compile(r"\b([A-Z]{1,5})\b")

def _symbols(text: str) -> List[str]:
    return sorted({m.group(1).upper() for m in SYM_RE.finditer(text or "")})

def _detect_action(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("sell", "short", "put")): return "sell"
    if any(k in t for k in ("buy", "long", "call")):   return "buy"
    return "hold"

def _confidence(text: str) -> float:
    t = text.lower()
    if "strong" in t or "very high" in t: return 0.9
    if "moderate" in t or "medium" in t:  return 0.65
    if "low" in t or "watch" in t:        return 0.45
    return 0.75

def _price(text: str) -> Optional[float]:
    m = re.search(r"\$?(\d+(?:\.\d{1,2})?)", text or "")
    return float(m.group(1)) if m else None

def normalize_posts(raw_posts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for post in raw_posts:
        ts = post.get("created_at") or datetime.now(timezone.utc).isoformat()
        content = (post.get("content") or post.get("body") or "").strip()
        syms = _symbols(content)
        if not syms:
            continue
        action = _detect_action(content)
        conf = _confidence(content)
        price = _price(content)
        for s in syms:
            out[s] = {
                "symbol": s,
                "action": action,
                "confidence": conf,
                "price": price,
                "timestamp": ts,
                "source": "patreon",
            }
            # allow override fields
            for k in ("price", "action", "confidence"):
                if k in post:
                    out[s][k] = post[k]
    return out

def save_latest_signals(posts: List[Dict[str, Any]]):
    signals = normalize_posts(posts)
    OUTPUT_PATH.write_text(json.dumps(signals, indent=2))
    log.info("[Patreon] Wrote %d normalized signals → %s", len(signals), OUTPUT_PATH)

if __name__ == "__main__":
    example_posts = [{
        "title":"Strong Buy: NVDA at $457",
        "content":"Strong Buy: NVDA at $457 breakout confirmed!",
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    save_latest_signals(example_posts)
