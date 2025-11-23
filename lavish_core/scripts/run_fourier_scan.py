# scripts/run_fourier_scan.py
import os
import sys

# allow running from repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from quant.fourier_engine import scan_symbols

if __name__ == "__main__":
    # Default starter set; you can override via CLI if you want
    symbols = os.getenv("FOURIER_SYMBOLS", "SPY,AAPL,NVDA,TSLA,AMD").split(",")
    timeframe = os.getenv("FOURIER_TIMEFRAME", "1Min")
    limit = int(os.getenv("FOURIER_LIMIT", "1500"))
    out = os.getenv("FOURIER_OUT", "signals/fourier_signals.csv")
    scan_symbols([s.strip().upper() for s in symbols], timeframe=timeframe, limit=limit, out_csv=out)