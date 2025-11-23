"""
Brain Vector — Hybrid Memory Core (Wall Street Edition)

• Lightweight embedding store backed by SQLite (DuckDB optional)
• Text/JSON ingestion with dedup by content hash
• Vector ops via cosine similarity (pluggable embedder)
• Daily insight writer to state/insights.json
"""

from __future__ import annotations
import os, sys, json, sqlite3, hashlib, math, pathlib
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

ROOT = pathlib.Path(__file__).resolve().parent
STATE = ROOT / "state"
STATE.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE / "brain.db"

def now_iso(): return datetime.now(timezone.utc).isoformat()

# ── Simple embedder (pluggable) ───────────────────────────────────────────────
def embed(text: str) -> List[float]:
    """
    Placeholder embedder: stable 384-dim hashing-based vector.
    Replace with your model (e.g., sentence-transformers) when ready.
    """
    import random
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]

def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    return 0.0 if da == 0 or db == 0 else num/(da*db)

# ── Storage ───────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  created_at TEXT,
  kind TEXT,
  meta TEXT,
  text TEXT,
  vec BLOB
);
CREATE INDEX IF NOT EXISTS idx_docs_kind ON docs(kind);
"""

class Brain:
    def __init__(self, db_path: pathlib.Path = DB_PATH):
        self.db_path = db_path
        self.con = sqlite3.connect(str(self.db_path))
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.executescript(SCHEMA)

    def _hash(self, text: str, kind: str, meta: Dict) -> str:
        return hashlib.sha1(f"{kind}|{json.dumps(meta,sort_keys=True)}|{text}".encode("utf-8","ignore")).hexdigest()

    def upsert(self, text: str, kind: str = "note", meta: Optional[Dict] = None) -> Tuple[int, bool]:
        meta = meta or {}
        h = self._hash(text, kind, meta)
        cur = self.con.execute("SELECT id FROM docs WHERE hash=?", (h,))
        row = cur.fetchone()
        if row:
            return int(row[0]), False
        v = embed(text)
        self.con.execute(
            "INSERT INTO docs(hash, created_at, kind, meta, text, vec) VALUES(?,?,?,?,?,?)",
            (h, now_iso(), kind, json.dumps(meta), text, json.dumps(v).encode("utf-8"))
        )
        self.con.commit()
        return int(self.con.execute("SELECT id FROM docs WHERE hash=?", (h,)).fetchone()[0]), True

    def search(self, query: str, k: int = 8, filter_kind: Optional[str] = None) -> List[Dict]:
        qv = embed(query)
        rows = self.con.execute("SELECT id, kind, meta, text, vec FROM docs" + (" WHERE kind=?" if filter_kind else ""),
                                ((filter_kind,) if filter_kind else ()))
        scored = []
        for i, kind, meta, text, vblob in rows:
            vec = json.loads(vblob.decode("utf-8"))
            scored.append((cosine(qv, vec), i, kind, json.loads(meta), text))
        scored.sort(key=lambda x: -x[0])
        out = [{"score": round(s,4), "id":i, "kind":k, "meta":m, "text":t} for s,i,k,m,t in scored[:k]]
        return out

    def daily_insights(self, top_n: int = 10) -> Dict:
        rows = self.con.execute("SELECT id, created_at, kind, meta, text FROM docs ORDER BY id DESC LIMIT ?", (top_n,))
        items = [{"id":i, "ts":ts, "kind":k, "meta":json.loads(m), "text":t} for i,ts,k,m,t in rows]
        out = {"generated": now_iso(), "latest": items}
        (STATE / "insights.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

# ── Convenience singletons ────────────────────────────────────────────────────
_brain: Optional[Brain] = None
def get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain

# ── CLI (quick test) ─────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Brain Vector")
    ap.add_argument("--ingest", type=str, default="", help="Text file to ingest (or raw text)")
    ap.add_argument("--kind", type=str, default="note")
    ap.add_argument("--query", type=str, default="")
    args = ap.parse_args()

    br = get_brain()
    if args.ingest:
        text = pathlib.Path(args.ingest).read_text(encoding="utf-8") if os.path.exists(args.ingest) else args.ingest
        _id, created = br.upsert(text, kind=args.kind, meta={})
        print(("↑ updated" if not created else "↑ created"), _id)
    if args.query:
        hits = br.search(args.query, k=5)
        for h in hits:
            print(h["score"], h["kind"], str(h["meta"])[:40], "→", h["text"][:80].replace("\n"," "))
    br.daily_insights()

if __name__ == "__main__":
    main()
