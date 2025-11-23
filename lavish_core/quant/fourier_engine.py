# lavish_core/signals/fourier_engine.py
# Fourier/Cycle/Volatility engine pulling bars from Alpaca Market Data v2.
# Env:
#   ALPACA_KEY  or APCA_API_KEY_ID
#   ALPACA_SECRET or APCA_API_SECRET_KEY
#   ALPACA_DATA_URL (optional, default: https://data.alpaca.markets)

from __future__ import annotations
import os, time, logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from scipy.signal import get_window

# ---------- logging ----------
log = logging.getLogger("fourier_engine")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    log.addHandler(_h)

# ---------- config / env ----------
ALPACA_BASE = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
# support both naming styles
ALPACA_KEY   = os.getenv("ALPACA_KEY") or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET= os.getenv("ALPACA_SECRET") or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

if not (ALPACA_KEY and ALPACA_SECRET):
    log.warning("Alpaca credentials not found. Set ALPACA_KEY & ALPACA_SECRET (or APCA_API_KEY_ID & APCA_API_SECRET_KEY).")

# ---------- dataclass ----------
@dataclass
class FourierSignal:
    symbol: str
    timeframe: str
    n: int
    last_ts: str
    price: float
    trend_slope: float
    trend_z: float
    cycle_period_bars: float
    cycle_strength: float
    vol_z: float
    reversal_risk: float
    notes: str

# ---------- helpers ----------
def _normalize_timeframe(tf: str) -> str:
    tf = tf.strip()
    # allow 1m, 5m, 15m, 1h, 1D, etc.
    map_simple = {
        "1m":"1Min","5m":"5Min","15m":"15Min","30m":"30Min","45m":"45Min",
        "1h":"1Hour","2h":"2Hour","4h":"4Hour","6h":"6Hour","8h":"8Hour",
        "1d":"1Day","1D":"1Day","day":"1Day","1wk":"1Week","1mo":"1Month"
    }
    return map_simple.get(tf, tf)

def _detrend(x: np.ndarray) -> np.ndarray:
    n = len(x); t = np.arange(n)
    A = np.vstack([t, np.ones(n)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m*t + b)

def _spectral_features(prices: np.ndarray, window: str="hann") -> Tuple[float,float,float]:
    x = _detrend(prices.astype(float))
    w = get_window(window, len(x), fftbins=True)
    mag = np.abs(np.fft.rfft(x*w))
    if len(mag) == 0:
        return (np.nan, 0.0, 1.0)
    mag[0] = 0.0
    if mag.max() <= 1e-12:
        return (np.nan, 0.0, 1.0)
    k = int(np.argmax(mag))
    dom_period = (len(x)/k) if k else np.nan
    peak_strength = float(mag[k]/(mag.mean()+1e-12))
    gm = np.exp(np.mean(np.log(mag+1e-12)))
    sf = float(gm/(mag.mean()+1e-12))
    return (dom_period, peak_strength, sf)

def _lowpass(prices: np.ndarray, cutoff_period: int=80) -> np.ndarray:
    n = len(prices); x = prices.astype(float)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    cutoff = 1.0 / max(2, cutoff_period)
    X[freqs > cutoff] = 0.0
    return np.fft.irfft(X, n=n)

def _zscore(a: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(a), np.nanstd(a)
    return np.zeros_like(a) if sd < 1e-12 else (a - mu) / sd

# ---------- Alpaca pull ----------
def _headers() -> dict:
    return {"APCA-API-KEY-ID": ALPACA_KEY or "", "APCA-API-SECRET-KEY": ALPACA_SECRET or ""}

def fetch_bars(symbol: str, timeframe: str="1Min", limit: int=1500,
               start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    """
    Try per-symbol bars endpoint first. If that fails/empty, try multi-symbol endpoint.
    If intraday returns empty/blocked, auto-fallback to 1Day once.
    """
    tf = _normalize_timeframe(timeframe)
    params = {"timeframe": tf, "limit": int(limit)}
    if start: params["start"] = start
    if end:   params["end"] = end

    # candidate endpoints to tolerate account/plan differences
    base = ALPACA_BASE
    single = f"{base}/v2/stocks/{symbol}/bars"
    multi  = f"{base}/v2/stocks/bars"  # ?symbols=SYM

    last_err = None
    # 1) single-symbol endpoint
    try:
        r = requests.get(single, headers=_headers(), params=params, timeout=30)
        if r.status_code == 200:
            js = r.json()
            bars = js.get("bars", [])
            if bars:
                df = pd.DataFrame(bars)
                df.rename(columns={"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df.sort_values("ts", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df[["ts","open","high","low","close","volume"]]
            else:
                log.error(f"[fetch_bars] {symbol} via {single} -> No bars in response.")
        else:
            last_err = f"{r.status_code} {r.text}"
            log.error(f"[fetch_bars] {symbol} failed via {single} -> {last_err}")
    except Exception as e:
        last_err = str(e)
        log.exception(f"[fetch_bars] {symbol} exception via {single}: {e}")

    # 2) multi-symbol endpoint with symbols=AAPL
    try:
        p2 = dict(params); p2["symbols"] = symbol
        r = requests.get(multi, headers=_headers(), params=p2, timeout=30)
        if r.status_code == 200:
            js = r.json()
            # response shape can be {"bars": [{"T":"AAPL", ...}, ...]} or {"bars":{"AAPL":[...]}}
            bars = js.get("bars", [])
            if isinstance(bars, dict):
                bars = bars.get(symbol, [])
            if bars:
                df = pd.DataFrame(bars)
                # handle both shapes
                rename_map = {"t":"ts","o":"open","h":"high","l":"low","c":"close","v":"volume",
                              "S":"symbol","T":"symbol","o":"open"}
                df.rename(columns=rename_map, inplace=True)
                if "ts" not in df and "t" in df: df["ts"] = df["t"]
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df.sort_values("ts", inplace=True)
                df.reset_index(drop=True, inplace=True)
                cols = [c for c in ["ts","open","high","low","close","volume"] if c in df.columns]
                return df[cols]
            else:
                log.error(f"[fetch_bars] {symbol} via {multi} -> No bars in response.")
        else:
            last_err = f"{r.status_code} {r.text}"
            log.error(f"[fetch_bars] {symbol} failed via {multi} -> {last_err}")
    except Exception as e:
        last_err = str(e)
        log.exception(f"[fetch_bars] {symbol} exception via {multi}: {e}")

    # 3) automatic fallback to 1Day if intraday blocked/empty
    if tf != "1Day":
        log.warning(f"[fetch_bars] {symbol}: intraday '{tf}' unavailable/empty. Falling back to 1Day.")
        return fetch_bars(symbol, timeframe="1Day", limit=min(limit, 1000), start=start, end=end)

    raise RuntimeError(f"All endpoints failed for {symbol}. Last error: {last_err or 'unknown'}")

# ---------- analysis ----------
def analyze_series(symbol: str, df: pd.DataFrame,
                   trend_period: int=80, vol_lookback: int=100, slope_lookback: int=60) -> FourierSignal:
    closes = df["close"].astype(float).values
    if len(closes) < max(trend_period, vol_lookback, slope_lookback) + 32:
        raise ValueError(f"Not enough bars for {symbol} (have {len(closes)})")

    trend = _lowpass(closes, cutoff_period=trend_period)
    slope = np.gradient(trend)
    slope_hist = slope[-slope_lookback:]
    trend_slope = float(slope[-1])
    trend_z = float(_zscore(slope_hist)[-1])

    look = min(512, len(closes))
    dom_period, peak_strength, _sf = _spectral_features(closes[-look:])

    rets = np.diff(np.log(closes))
    vol = pd.Series(rets).rolling(window=20).std().values
    vol_hist = vol[-vol_lookback:]
    vol_z = float(_zscore(vol_hist)[-1]) if len(vol_hist) else 0.0

    ext_hist = closes[-slope_lookback:] - trend[-slope_lookback:]
    ext_z = float(_zscore(ext_hist)[-1]) if len(ext_hist) else 0.0
    reversal_risk = float(np.clip(0.15*abs(trend_z) + 0.55*abs(ext_z) + 0.30*max(0.0, vol_z), 0.0, 3.0)/3.0)

    notes = []
    if not np.isnan(dom_period): notes.append(f"domP≈{dom_period:.1f} bars")
    if peak_strength > 2.0: notes.append("strong cycle")
    if trend_z > 1.5: notes.append("trend↑")
    if trend_z < -1.5: notes.append("trend↓")
    if reversal_risk > 0.7: notes.append("rev-risk↑")

    return FourierSignal(
        symbol=symbol,
        timeframe="",
        n=len(df),
        last_ts=df["ts"].iloc[-1].isoformat(),
        price=float(closes[-1]),
        trend_slope=float(trend_slope),
        trend_z=float(trend_z),
        cycle_period_bars=float(dom_period) if not np.isnan(dom_period) else np.nan,
        cycle_strength=float(np.clip(peak_strength/5.0, 0.0, 1.0)),
        vol_z=float(vol_z),
        reversal_risk=float(reversal_risk),
        notes="; ".join(notes),
    )

def scan_symbols(symbols: List[str], timeframe: str="15Min", limit: int=500,
                 out_csv: str="signals/fourier_signals.csv") -> List[FourierSignal]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results: List[FourierSignal] = []
    tf = _normalize_timeframe(timeframe)

    for sym in symbols:
        try:
            df = fetch_bars(sym, timeframe=tf, limit=limit)
            sig = analyze_series(sym, df)
            sig.timeframe = tf
            results.append(sig)
            log.info(f"[{sym}] px={sig.price:.2f} trend_z={sig.trend_z:+.2f} "
                     f"cycleP={sig.cycle_period_bars:.1f} strength={sig.cycle_strength:.2f} "
                     f"vol_z={sig.vol_z:+.2f} rev={sig.reversal_risk:.2f} | {sig.notes}")
            time.sleep(0.15)
        except Exception as e:
            log.error(f"{sym} failed: {e}")

    if results:
        pd.DataFrame([s.__dict__ for s in results]).to_csv(out_csv, index=False)
        log.info(f"Wrote {len(results)} rows -> {out_csv}")
    return results

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fourier engine (Alpaca Market Data v2)")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated list, e.g., AAPL,MSFT,NVDA,SPY")
    p.add_argument("--timeframe", type=str, default="15Min", help="e.g., 15m, 1h, 1Day")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", type=str, default="signals/fourier_signals.csv")
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    scan_symbols(syms, timeframe=args.timeframe, limit=args.limit, out_csv=args.out)