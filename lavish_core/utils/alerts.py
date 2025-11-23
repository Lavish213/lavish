"""
Lavish Bot — Alerts (Discord)
=============================
Unified helpers to send rich Discord messages with safe fallbacks.
"""

from __future__ import annotations
import os, json, requests
from typing import Optional, Dict, Any

WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

def _send(payload: Dict[str, Any]) -> None:
    if not WEBHOOK:
        print("[WARN] No DISCORD_WEBHOOK_URL set.\n", json.dumps(payload, indent=2))
        return
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=15)
        if r.status_code >= 300:
            print("[Discord ERROR]", r.status_code, r.text[:300])
    except Exception as e:
        print("[Discord ERROR]", e)

def post_discord(msg: str, embed: Optional[dict] = None):
    payload = {"content": msg}
    if embed: payload["embeds"] = [embed]
    _send(payload)

def send_trade_alert(action: str, symbol: str, note: str = "", confidence: Optional[float] = None):
    color = 0x2ecc71 if action.upper() == "BUY" else 0xe74c3c
    embed = {
        "title": f"{'🟢 BUY' if action.upper()=='BUY' else '🔴 SELL'} {symbol}",
        "color": color,
        "fields": [
            {"name": "Action", "value": action.upper(), "inline": True},
            {"name": "Symbol", "value": symbol.upper(), "inline": True},
        ],
    }
    if confidence is not None:
        embed["fields"].append({"name": "Confidence", "value": f"{confidence:.2f}", "inline": True})
    if note:
        embed["fields"].append({"name": "Note", "value": note[:256], "inline": False})
    _send({"content": "", "embeds": [embed]})

def send_vision_alert(symbols, action, src_file, conf):
    txt = f"🧠 Vision parsed: {', '.join(symbols)} | {action} | conf={conf:.2f} | src={src_file}"
    post_discord(txt)