# lavish_core/ml/predictor.py (BUFFED + PRODUCTION READY)
from __future__ import annotations
import os, json, joblib, numpy as np
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

REGISTRY = "models/registry.json"

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def _safe_path(base: str, p: str) -> str:
    """Resolve path safely relative to model directory."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base, p))

def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY):
        raise FileNotFoundError("Registry missing: train a model first.")
    with open(REGISTRY, "r") as f:
        return json.load(f)

def _choose_best_meta(reg: Dict[str, Any]) -> Dict[str, Any]:
    best_key, best_score = None, -1
    for key, meta in reg.items():
        score = float(meta.get("val_macro_f1", meta.get("cv_macro_f1", 0.0)))
        if score > best_score:
            best_score, best_key = score, key
    return reg[best_key]

# ───────────────────────────────────────────────────────────────
# Cached Model Loader (FAST)
# ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_bundle() -> Tuple[Any, Any, list, float]:
    """
    Loads model, vectorizer, labels, and threshold ONCE.
    Future predictions re-use cached objects (massive speed boost).
    """
    reg = _load_registry()
    meta = _choose_best_meta(reg)

    model_path = meta["path"]
    base = os.path.dirname(model_path)

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        est = bundle["estimator"]
        th = float(bundle.get("threshold", meta.get("threshold", 0.5)))
    else:
        est = bundle
        th = float(meta.get("threshold", 0.5))

    # Vectorizer path stability
    vec_path = _safe_path(base, meta["vectorizer"])
    vec = joblib.load(vec_path)

    labels = meta.get("labels", ["BUY", "SELL", "CALL", "PUT", "UNKNOWN"])
    if "UNKNOWN" not in labels:
        labels.append("UNKNOWN")

    return est, vec, labels, th

# ───────────────────────────────────────────────────────────────
# Text Cleaner (helps your OCR pipeline)
# ───────────────────────────────────────────────────────────────

def _clean_text(t: str) -> str:
    if not isinstance(t, str): 
        return ""
    t = t.strip()
    t = t.replace("\n", " ")
    t = " ".join(t.split())  # compress whitespace
    return t

# ───────────────────────────────────────────────────────────────
# Prediction (BUFFED)
# ───────────────────────────────────────────────────────────────

def predict_text(text: str) -> Dict[str, Any]:
    est, vec, labels, th = _load_bundle()

    clean = _clean_text(text)
    X = vec.transform([clean])

    # Probability block
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)[0]
    else:
        # Manual softmax fallback
        d = est.decision_function(X)
        if d.ndim == 1:  # binary
            d = np.vstack([-d, d]).T
        e = np.exp(d - d.max(axis=1, keepdims=True))
        p = (e / e.sum(axis=1, keepdims=True))[0]

    idx = int(np.argmax(p))
    conf = float(p[idx])

    # Confidence smoothing
    conf = round(conf, 4)

    # Threshold enforcement → UNKNOWN
    if conf < th:
        idx = labels.index("UNKNOWN")

    out = {
        "label": labels[idx],
        "confidence": round(conf, 3),
        "threshold": th,
        "probs": {labels[i]: float(p[i]) for i in range(len(labels))}
    }
    return out

# ───────────────────────────────────────────────────────────────
# OCR Record Prediction (used in candle_reader + live monitor)
# ───────────────────────────────────────────────────────────────

def predict_ocr_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    txt = (rec or {}).get("ocr_text") or ""
    out = predict_text(txt)
    out["image_path"] = rec.get("image_path", "")
    out["cache_key"] = rec.get("cache_key", "")
    return out