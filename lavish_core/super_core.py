# lavish_core/super_core.py
# Lavish "Brain": chat + retrieval + trading hooks + decision logs
# Hybrid vector store (Chroma > FAISS > TF-IDF) and hybrid embedder (SentenceTransformer > TF-IDF)
# Persona: witty, confident, Wall-Street mentor
# Dry-run trading by default; auto-upgrades to live if trade_agent available and LIVE_TRADE=true

import os, sys, json, time, uuid, glob, logging, traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

# ------------ Logging ------------
os.makedirs("memory/logs", exist_ok=True)
LOGFILE = os.path.join("memory", "logs", f"super_core_{datetime.now(timezone.utc).date()}.jsonl")

logger = logging.getLogger("lavish_super_core")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)

def log_event(kind: str, payload: Dict):
    payload = dict(payload)
    payload["ts_utc"] = datetime.now(timezone.utc).isoformat()
    payload["kind"] = kind
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ------------ Embedder (Hybrid) ------------
class HybridEmbedder:
    """
    Prefers sentence-transformers; falls back to TF-IDF if unavailable.
    """
    def __init__(self):
        self.mode = "tfidf"
        self.model = None
        self.vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            # Light, widely cached model; replace if you have a local one
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.mode = "sbert"
            logger.info("Embedder: sentence-transformers ready.")
        except Exception as e:
            logger.warning(f"Embedder fallback to TF-IDF ({e})")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1,2))

    def fit(self, texts: List[str]):
        if self.mode == "tfidf":
            self.vectorizer.fit(texts)

    def encode(self, texts: List[str]):
        if self.mode == "sbert":
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            # lazy-fit if empty
            if not hasattr(self.vectorizer, "vocabulary_") or not self.vectorizer.vocabulary_:
                self.vectorizer.fit(texts)
            return self.vectorizer.transform(texts).toarray()

# ------------ Store (Hybrid) ------------
class HybridStore:
    """
    Chroma > FAISS > InMemory cosine with brute force.
    Schema: each doc = {id, text, meta}
    """
    def __init__(self, embedder: HybridEmbedder, persist_dir="memory/vector_store"):
        self.embedder = embedder
        self.backend = "inmem"
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._texts: List[str] = []
        self._metas: List[Dict] = []
        self._ids: List[str] = []
        self._emb = None

        # Try Chroma
        try:
            import chromadb
            from chromadb.config import Settings
            self.chroma_client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
            self.collection = self.chroma_client.get_or_create_collection("lavish_core")
            self.backend = "chroma"
            logger.info("Vector store: Chroma ready.")
            return
        except Exception as e:
            logger.warning(f"Chroma unavailable, trying FAISS ({e})")

        # Try FAISS
        try:
            import faiss  # noqa: F401
            self.backend = "faiss"
            self._faiss_index = None
            logger.info("Vector store: FAISS ready (in-memory).")
            return
        except Exception as e:
            logger.warning(f"FAISS unavailable, in-memory store only ({e})")

    def _ensure_faiss(self):
        if self._emb is None and self._texts:
            self._emb = self.embedder.encode(self._texts)
        if getattr(self, "_faiss_index", None) is None and self._emb is not None:
            import faiss
            d = self._emb.shape[1]
            self._faiss_index = faiss.IndexFlatIP(d)  # cosine via normalized vectors (sbert) or dot proxy
            self._faiss_index.add(self._emb.astype("float32"))

    def add(self, texts: List[str], metas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        metas = metas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        if self.backend == "chroma":
            # Chroma handles embedding internally if you pass embeddings, but we control embeddings to keep hybrid
            embs = self.embedder.encode(texts)
            self.collection.add(documents=texts, embeddings=[e.tolist() for e in embs], metadatas=metas, ids=ids)
            return

        # local modes
        self._texts.extend(texts)
        self._metas.extend(metas)
        self._ids.extend(ids)
        self._emb = None  # will recompute lazily for FAISS

    def persist(self):
        if self.backend == "chroma":
            return  # persistent
        # You could add pickle persist for FAISS/inmem if desired.

    def _search_inmem(self, query: str, k=5):
        import numpy as np
        if not self._texts:
            return []
        # compute (or reuse) embeddings
        q_emb = self.embedder.encode([query])[0]
        if self._emb is None:
            self._emb = self.embedder.encode(self._texts)
        # cosine
        def cos(a,b): 
            na = (a**2).sum()**0.5 + 1e-12
            nb = (b**2).sum()**0.5 + 1e-12
            return float((a@b)/(na*nb))
        sims = [cos(q_emb, x) for x in self._emb]
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [{"id": self._ids[i], "text": self._texts[i], "meta": self._metas[i], "score": sims[i]} for i in idx]

    def search(self, query: str, k=5):
        if self.backend == "chroma":
            q_emb = self.embedder.encode([query])[0].tolist()
            res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances","ids"])
            out = []
            if res and res.get("documents"):
                for i in range(len(res["documents"][0])):
                    out.append({
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i],
                        "meta": res["metadatas"][0][i],
                        "score": 1 - float(res["distances"][0][i]) if res.get("distances") else None
                    })
            return out

        if self.backend == "faiss":
            import numpy as np, faiss
            if not self._texts:
                return []
            q_emb = self.embedder.encode([query]).astype("float32")
            if self._emb is None:
                self._emb = self.embedder.encode(self._texts).astype("float32")
            if getattr(self, "_faiss_index", None) is None:
                self._ensure_faiss()
            D, I = self._faiss_index.search(q_emb, k)
            out = []
            for j, idx in enumerate(I[0]):
                if idx == -1: 
                    continue
                out.append({"id": self._ids[idx], "text": self._texts[idx], "meta": self._metas[idx], "score": float(D[0][j])})
            return out

        # in-memory brute force
        return self._search_inmem(query, k=k)

# ------------ Styler / Persona ------------
def lavish_reply_style(raw: str) -> str:
    """
    Witty + confident mentor tone. Keep it punchy, clear, and useful.
    """
    return (
        raw
        .replace("I think", "My take")
        .replace("maybe", "likely")
    )

# ------------ Trading Bridge ------------
class TradeBridge:
    def __init__(self):
        self.live = os.getenv("LIVE_TRADE", "false").lower() == "true"
        self.loaded = False
        self.agent = None
        try:
            # Your file: lavish_core/trade/trade_agent.py
            from lavish_core.trade.trade_agent import TradeAgent  # type: ignore
            self.agent = TradeAgent()
            self.loaded = True
            logger.info("TradeAgent loaded.")
        except Exception as e:
            logger.warning(f"TradeAgent not available, running in DRY mode. ({e})")

    def place_order(self, symbol: str, side: str, qty: float, meta: Dict):
        ev = {"symbol": symbol, "side": side, "qty": qty, "meta": meta, "live": self.live, "bridge_loaded": self.loaded}
        if not self.loaded or not self.live:
            log_event("order_dry_run", ev)
            return {"status": "dry_run", "detail": ev}
        try:
            res = self.agent.place_order(symbol=symbol, side=side, qty=qty, meta=meta)
            log_event("order_live", {"result": res, **ev})
            return {"status": "live_ok", "result": res}
        except Exception as e:
            log_event("order_error", {"error": str(e), **ev})
            return {"status": "error", "error": str(e)}

# ------------ Core Brain ------------
class LavishCore:
    def __init__(self, store: HybridStore):
        self.store = store
        self.trader = TradeBridge()

    def _retrieve(self, query: str, k=6) -> List[Dict]:
        hits = self.store.search(query, k=k)
        log_event("retrieve", {"query": query, "hits": [{"score": h.get("score"), "meta": h.get("meta")} for h in hits]})
        return hits

    def answer(self, query: str) -> str:
        """
        Retrieval-augmented answer with persona.
        """
        hits = self._retrieve(query, k=6)
        context = "\n---\n".join([f"[{i+1}] {h['text']}" for i,h in enumerate(hits)])
        if not context:
            base = "No prior notes on this yet. Shoot me details or drop files into /raw and run the trainer — I’ll ingest and advise."
            log_event("answer", {"query": query, "response": base})
            return lavish_reply_style(base)

        # Simple rank fusion + extractive summary heuristic
        answer = f"I pulled {len(hits)} notes that matter:\n\n"
        for i, h in enumerate(hits, 1):
            short = h["text"]
            if len(short) > 300:
                short = short[:300] + "..."
            src = h.get("meta", {}).get("source", "memory")
            answer += f"{i}. ({src}) {short}\n"
        answer += "\nMy take: Based on the retrieved notes, what do you want me to execute — scan, explain, or place a dry-run trade?"

        answer = lavish_reply_style(answer)
        log_event("answer", {"query": query, "response": answer})
        return answer

    def decide_and_trade(self, signal: Dict) -> Dict:
        """
        Example: signal -> {"symbol":"AAPL","bias":"buy","confidence":0.78,"qty":5}
        Applies layered checks and either dry-runs or places real order (if enabled).
        """
        layers = []
        ok = True

        # L1: sanity
        sym = signal.get("symbol", "").upper()
        side = "buy" if signal.get("bias","").lower().startswith("b") else "sell"
        qty = float(signal.get("qty", 0))
        conf = float(signal.get("confidence", 0.0))
        layers.append(("sanity", ok and sym != "" and qty > 0))
        ok = ok and (sym != "" and qty > 0)

        # L2: confidence threshold
        thr = float(os.getenv("TRADE_CONF_THRESHOLD", "0.65"))
        layers.append(("confidence", conf >= thr))
        ok = ok and (conf >= thr)

        # L3: risk toggle
        risk = os.getenv("RISK_MODE", "normal")  # "low","normal","high"
        if risk == "low": qty = max(1.0, qty*0.5)
        if risk == "high": qty = qty*1.5
        layers.append(("risk_adjust", True))

        # L4: memory veto (recent warnings)
        veto_hits = self._retrieve(f"risk warning {sym} downgrade halt")
        veto = any("halt" in h["text"].lower() or "downgrade" in h["text"].lower() for h in veto_hits)
        layers.append(("memory_veto", not veto))
        ok = ok and (not veto)

        # L5: cooldown per symbol
        cooldown_s = int(os.getenv("TRADE_COOLDOWN_SEC", "60"))
        recent = False
        try:
            # check recent orders in log
            cutoff = time.time() - cooldown_s
            with open(LOGFILE, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("kind","").startswith("order_") and obj.get("symbol")==sym:
                        ts = datetime.fromisoformat(obj["ts_utc"]).timestamp()
                        if ts >= cutoff: recent = True; break
        except Exception:
            pass
        layers.append(("cooldown", not recent))
        ok = ok and (not recent)

        decision = {"ok": ok, "layers": layers, "normalized_qty": qty}
        if not ok:
            log_event("trade_blocked", {"signal": signal, **decision})
            return {"status": "blocked", **decision}

        # Execute
        meta = {"confidence": conf, "risk": risk, "layers": layers}
        res = self.trader.place_order(symbol=sym, side=side, qty=qty, meta=meta)
        log_event("trade_decision", {"signal": signal, "result": res, **decision})
        return {"status": "submitted", "broker": res, **decision}

# ------------ Bootstrapping ------------
def load_store(embedder: HybridEmbedder) -> HybridStore:
    store = HybridStore(embedder=embedder, persist_dir="memory/vector_store")
    # Best-effort hot-load for FAISS/TFIDF modes from last trainer export (optional)
    cache_glob = "memory/vector_store_cache/*.jsonl"
    if glob.glob(cache_glob):
        texts, metas, ids = [], [], []
        for path in glob.glob(cache_glob):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    texts.append(o["text"]); metas.append(o.get("meta", {})); ids.append(o.get("id", str(uuid.uuid4())))
        try:
            embedder.fit(texts)
        except Exception:
            pass
        store.add(texts, metas=metas, ids=ids)
        logger.info(f"Preloaded {len(texts)} chunks into store.")
    return store

# ------------ CLI ------------
def _print_help():
    print("""
Lavish Super Core — commands:
  answer "your question"         -> Retrieval QA with persona
  trade SYMBOL SIDE QTY CONF     -> Decide & (dry) trade, e.g., trade AAPL buy 5 0.77
  mode                            -> Show live/dry and thresholds
  help                            -> This message
Examples:
  python -m lavish_core.super_core answer "What do we know about NVDA earnings?"
  python -m lavish_core.super_core trade TSLA buy 3 0.72
""".strip())

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lavish Super Core (brain)")
    parser.add_argument("cmd", type=str, nargs="?", default="help")
    parser.add_argument("rest", nargs="*", default=[])
    args = parser.parse_args()

    emb = HybridEmbedder()
    store = load_store(emb)
    core = LavishCore(store)

    if args.cmd == "help":
        _print_help(); return

    if args.cmd == "mode":
        print(json.dumps({
            "live_trading": core.trader.live and core.trader.loaded,
            "bridge_loaded": core.trader.loaded,
            "conf_threshold": float(os.getenv("TRADE_CONF_THRESHOLD", "0.65")),
            "risk_mode": os.getenv("RISK_MODE", "normal"),
        }, indent=2)); return

    if args.cmd == "answer":
        q = " ".join(args.rest).strip().strip('"').strip("'")
        if not q:
            print("Give me a question, chief."); return
        out = core.answer(q)
        print("\n" + out); return

    if args.cmd == "trade":
        if len(args.rest) < 4:
            print("Usage: trade SYMBOL SIDE QTY CONF"); return
        sym, side, qty, conf = args.rest[0], args.rest[1], float(args.rest[2]), float(args.rest[3])
        sig = {"symbol": sym, "bias": side, "qty": qty, "confidence": conf}
        res = core.decide_and_trade(sig)
        print(json.dumps(res, indent=2)); return

    _print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error:\n" + traceback.format_exc())
        log_event("fatal", {"error": str(e), "trace": traceback.format_exc()})
        raise

