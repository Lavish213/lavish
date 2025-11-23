# lavish_core/ml/train_loop.py
from __future__ import annotations
import os, json, time, joblib, logging, argparse, numpy as np
from typing import Dict, Any, Tuple, List

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from ml.dataset import load_dataset, CANON_LABELS, MODELS_DIR

log = logging.getLogger("LavishTrainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(logging.INFO)

REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
REPORT_PATH   = os.path.join("logs", "train_report.txt")
os.makedirs("logs", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def _maybe_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None

def _save_registry(name: str, payload: Dict[str, Any]) -> None:
    reg = {}
    if os.path.exists(REGISTRY_PATH):
        try: reg = json.load(open(REGISTRY_PATH,"r"))
        except Exception: reg = {}
    reg[name] = payload
    with open(REGISTRY_PATH,"w") as f: json.dump(reg,f,indent=2)

def _tune_thresholds(probs: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Per-class thresholds (one-vs-rest) to maximize macro-F1 on validation.
    For models without predict_proba, we pass calibrated scores.
    """
    K = probs.shape[1]
    th = np.full(K, 0.5)
    best_f1 = -1.0
    best_th = th.copy()
    # quick grid
    grid = np.linspace(0.3, 0.7, 9)
    for t in grid:
        y_hat = probs.argmax(axis=1)
        # uniform threshold variant: suppress low confidence to UNKNOWN (last class)
        y_hat_adj = []
        for i, cls in enumerate(y_hat):
            if probs[i, cls] < t:
                # map to UNKNOWN index if present
                unk = CANON_LABELS.index("UNKNOWN")
                y_hat_adj.append(unk)
            else:
                y_hat_adj.append(cls)
        y_hat_adj = np.asarray(y_hat_adj)
        f1 = f1_score(y_true, y_hat_adj, average="macro")
        if f1 > best_f1:
            best_f1, best_th = f1, np.full(K, t)
    return best_th, float(best_f1)

def _calibrated_svm(class_weights: dict):
    base = LinearSVC(C=2.0, class_weight=class_weights, max_iter=5000)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)

def _logreg(class_weights: dict, n_jobs:int=8):
    return LogisticRegression(
        penalty="l2", C=2.0, max_iter=400, class_weight=class_weights, n_jobs=n_jobs, verbose=0
    )

def _xgb(class_weights: dict, n_jobs:int=8, n_estimators:int=600, max_depth:int=6, lr:float=0.07):
    XGB = _maybe_xgb()
    if XGB is None: return None
    # convert class_weights to sample_weight in fit
    return XGB(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42
    )

def _cv_score(estimator, X, y, folds=5) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores=[]
    for tr, va in skf.split(X, y):
        est = estimator
        est.fit(X[tr], y[tr])
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X[va])
        else:
            # decision_function -> softmax-ish
            d = est.decision_function(X[va])
            if d.ndim == 1: d = np.vstack([-d, d]).T
            e = np.exp(d - d.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
        th,_ = _tune_thresholds(probs, y[va])
        y_hat = probs.argmax(axis=1)
        y_hat = np.where(probs.max(axis=1) < th[0], CANON_LABELS.index("UNKNOWN"), y_hat)
        scores.append(f1_score(y[va], y_hat, average="macro"))
    return float(np.mean(scores))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", choices=["auto","svm","logreg","xgb"])
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.07)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    data = load_dataset()
    X_tr, X_va = data.X_train, data.X_valid
    y_tr, y_va = data.y_train, data.y_valid

    candidates = []
    if args.model == "auto":
        candidates = ["svm","logreg"] + (["xgb"] if _maybe_xgb() else [])
    else:
        candidates = [args.model]

    best = None
    best_cv = -1.0
    log.info(f"Candidates: {candidates}")

    for name in candidates:
        if name == "svm":
            est = _calibrated_svm(data.class_weights)
        elif name == "logreg":
            est = _logreg(data.class_weights, n_jobs=args.n_jobs)
        elif name == "xgb":
            est = _xgb(data.class_weights, n_jobs=args.n_jobs,
                       n_estimators=args.n_estimators, max_depth=args.max_depth, lr=args.lr)
            if est is None:
                log.warning("XGBoost unavailable; skipping.")
                continue
        else:
            continue

        log.info(f"[CV] scoring {name} ...")
        cv = _cv_score(est, X_tr, y_tr, folds=5)
        log.info(f"[CV] {name} macro-F1={cv:.4f}")
        if cv > best_cv:
            best_cv, best = cv, (name, est)

    name, est = best
    t0=time.time(); est.fit(X_tr, y_tr); dur = time.time()-t0
    # Validation with tuned thresholds
    if hasattr(est,"predict_proba"):
        probs=est.predict_proba(X_va)
    else:
        d=est.decision_function(X_va)
        if d.ndim == 1: d = np.vstack([-d, d]).T
        e=np.exp(d-d.max(axis=1,keepdims=True))
        probs=e/e.sum(axis=1,keepdims=True)
    th, tuned_f1 = _tune_thresholds(probs, y_va)
    y_hat = probs.argmax(axis=1)
    y_hat = np.where(probs.max(axis=1) < th[0], CANON_LABELS.index("UNKNOWN"), y_hat)

    f1 = f1_score(y_va, y_hat, average="macro")
    cm = confusion_matrix(y_va, y_hat, labels=list(range(len(CANON_LABELS))))
    report = classification_report(y_va, y_hat, target_names=CANON_LABELS, digits=3)

    out_path = os.path.join(MODELS_DIR, f"vision_{name}.joblib")
    joblib.dump({"estimator": est, "threshold": float(th[0])}, out_path)
    log.info(f"Saved model → {out_path} | val macro-F1={f1:.4f} (tuned={tuned_f1:.4f}) | train_time={dur:.1f}s")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Model: {name}\nMacro-F1: {f1:.4f}\nThreshold: {float(th[0]):.3f}\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n\n")
        f.write(report)
    log.info(f"Report → {REPORT_PATH}")

    # Registry entry
    reg_payload = {
        "path": out_path,
        "labels": CANON_LABELS,
        "threshold": float(th[0]),
        "vectorizer": os.path.join(MODELS_DIR, "vectorizer.pkl"),
        "cv_macro_f1": round(best_cv,4),
        "val_macro_f1": round(f1,4),
    }
    _save_registry(f"vision_{name}", reg_payload)

if __name__ == "__main__":
    main()