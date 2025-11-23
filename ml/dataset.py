# lavish_core/ml/dataset.py
from __future__ import annotations
import os, json, logging, joblib
import numpy as np, pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# =========================
# CONFIG & PATHS
# =========================
DATA_DIR = "data/ocr"
DATA_PATH = os.path.join(DATA_DIR, "dataset.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CANON_LABELS = ["BUY", "SELL", "CALL", "PUT", "UNKNOWN"]

log = logging.getLogger("LavishDataset")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# =========================
# HELPER UTILITIES
# =========================
def _normalize_label(lbl: str) -> str:
    if not lbl:
        return "UNKNOWN"
    lbl = lbl.upper().strip()
    for c in CANON_LABELS:
        if c in lbl:
            return c
    if "BTO" in lbl or "BUY" in lbl:
        return "BUY"
    if "STO" in lbl or "SELL" in lbl:
        return "SELL"
    return "UNKNOWN"

def _safe_float(x):
    try: return float(x)
    except Exception: return 0.0

# =========================
# LOAD RAW DATA
# =========================
def _load_csv() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run ingest_images_in.py first.")
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
    df["label"] = df["label"].apply(_normalize_label)
    return df

# =========================
# FEATURE EXTRACTION
# =========================
def _vectorize_texts(texts: list[str], save_path: str):
    if os.path.exists(save_path):
        vec = joblib.load(save_path)
        X = vec.transform(texts)
        return X, vec
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True
    )
    X = vec.fit_transform(texts)
    joblib.dump(vec, save_path)
    return X, vec

def _build_numeric(df: pd.DataFrame) -> np.ndarray:
    df_num = pd.DataFrame()
    df_num["strike"] = df["strike"].apply(_safe_float)
    df_num["conf"] = df["confidence"].apply(_safe_float)
    df_num["is_call"] = df["option_type"].str.contains("CALL", case=False).astype(int)
    df_num["is_put"] = df["option_type"].str.contains("PUT", case=False).astype(int)
    return df_num.values.astype(float)

# =========================
# MAIN DATASET BUILDER
# =========================
class Dataset:
    def __init__(self, X_train, X_valid, y_train, y_valid, vectorizer, scaler, class_weights):
        self.X_train, self.X_valid = X_train, X_valid
        self.y_train, self.y_valid = y_train, y_valid
        self.vectorizer, self.scaler = vectorizer, scaler
        self.class_weights = class_weights

def load_dataset() -> Dataset:
    df = _load_csv()
    if len(df) < 5:
        raise ValueError(f"Not enough samples ({len(df)}) in dataset.")

    # --- Features ---
    text_data = df["ocr_text"].astype(str).tolist()
    X_text, vec = _vectorize_texts(text_data, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    X_num = csr_matrix(_build_numeric(df))
    X_full = hstack([X_text, X_num]).tocsr()

    # --- Labels ---
    label_map = {c: i for i, c in enumerate(CANON_LABELS)}
    y = np.array([label_map.get(l, label_map["UNKNOWN"]) for l in df["label"]])

    # --- Split ---
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Scaling numeric features ---
    scaler = StandardScaler(with_mean=False)
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # --- Class weights ---
    uniq = np.unique(y, return_counts=False)
    cw = compute_class_weight("balanced", classes=uniq, y=y)
    class_weights = {int(k): float(v) for k, v in zip(uniq, cw)}

    log.info(f"Dataset built: {X_tr.shape[0]} train / {X_va.shape[0]} val / {len(CANON_LABELS)} classes")
    return Dataset(X_tr_scaled, X_va_scaled, y_tr, y_va, vec, scaler, class_weights)