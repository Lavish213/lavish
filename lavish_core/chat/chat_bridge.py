# lavish_core/chat/chat_bridge.py
# Ask the bot questions about OCR events & trade decisions — no LLM required.
# Optional: if OPENAI_API_KEY exists, we’ll summarize answers more naturally.

from __future__ import annotations
import os, sqlite3, textwrap, json, re
from typing import List, Tuple, Optional

DB_PATH = os.getenv("LAVISH_DB", "data/lavish_vision.sqlite")
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))

def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def _rows(q: str, args: tuple=()) -> List[tuple]:
    with _conn() as c:
        cur = c.execute(q, args)
        return cur.fetchall()

def last_ocr(n: int=5) -> List[dict]:
    rows = _rows("""SELECT ts_utc, source, src_id, engine, conf, substr(text,1,800)
                    FROM ocr_events ORDER BY id DESC LIMIT ?""", (n,))
    return [
        {"ts":r[0], "src":r[1], "id":r[2], "engine":r[3], "conf":r[4], "text":r[5]}
        for r in rows
    ]

def last_trades(n: int=10) -> List[dict]:
    rows = _rows("""SELECT ts_utc, ticker, side, confidence, qty, dry_run, paper, sent, extra
                    FROM trade_decisions ORDER BY id DESC LIMIT ?""", (n,))
    out = []
    for r in rows:
        out.append({
            "ts":r[0], "ticker":r[1], "side":r[2], "conf":r[3], "qty":r[4],
            "dry_run":bool(r[5]), "paper":bool(r[6]), "sent":bool(r[7]), "detail":r[8]
        })
    return out

def why_ticker(ticker: str, limit: int=10) -> str:
    ticker = ticker.upper()
    rows = _rows("""SELECT ts_utc, side, confidence, qty, dry_run, paper, sent, extra
                    FROM trade_decisions WHERE UPPER(ticker)=? ORDER BY id DESC LIMIT ?""",
                 (ticker, limit))
    if not rows:
        return f"No history for {ticker}."
    lines = [f"Recent decisions for {ticker}:"]
    for r in rows:
        ts, side, conf, qty, dry, pap, sent, extra = r
        lines.append(f"- {ts} | {side} qty={qty} conf={conf:.2f} "
                     f"{'DRY' if dry else 'LIVE'} {'PAPER' if pap else 'LIVE'} "
                     f"{'SENT' if sent else 'BLOCKED'} | {extra or ''}")
    return "\n".join(lines)

def search_text(query: str, n: int=5) -> List[dict]:
    # naive LIKE search across OCR text
    q = f"%{query}%"
    rows = _rows("""SELECT ts_utc, source, src_id, engine, conf, substr(text,1,800)
                    FROM ocr_events WHERE text LIKE ? ORDER BY id DESC LIMIT ?""", (q, n))
    return [
        {"ts":r[0], "src":r[1], "id":r[2], "engine":r[3], "conf":r[4], "text":r[5]}
        for r in rows
    ]

# Optional: a super simple router that turns questions into actions
def ask(question: str) -> str:
    q = question.strip()
    # Commands
    m = re.match(r"/why\s+([A-Za-z]{1,5})", q, flags=re.I)
    if m: return why_ticker(m.group(1))

    if re.search(r"last (ocr|images?)", q, flags=re.I):
        items = last_ocr(5)
        if not items: return "No OCR events yet."
        return "\n\n".join([f"{it['ts']} [{it['engine']} conf={it['conf']:.2f}] {it['text']}" for it in items])

    if re.search(r"last (trades?|orders?)", q, flags=re.I):
        items = last_trades(8)
        if not items: return "No trade decisions yet."
        return "\n".join([f"{it['ts']} {it['ticker']} {it['side']} qty={it['qty']} sent={it['sent']} ({'DRY' if it['dry_run'] else 'LIVE'})" for it in items])

    # Fallback keyword search
    hits = search_text(q, 5)
    if hits:
        return "Matches in OCR:\n" + "\n\n".join([f"{h['ts']} [{h['src']}] {h['text']}" for h in hits])

    return "I didn’t find anything for that yet. Try: /why AAPL  •  'last ocr'  •  'last trades'  •  keyword search."

# Optional LLM polish (only if you want; not required)
def answer(question: str) -> str:
    if not USE_LLM:
        return ask(question)
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        base = ask(question)
        prompt = f"Rewrite this answer clearly in one short paragraph:\n\n{base}"
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return ask(question)
    
# add inside your existing chat_bridge.py
def classify(text: str):
    try:
        from lavish_core.ml.predictor import predict_text
        res = predict_text(text)
        return f"{res['label']}  (conf={res['confidence']})  probs={res['probs']}"
    except Exception as e:
        return f"Classifier unavailable: {e}"