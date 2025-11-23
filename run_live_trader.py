# run_live_trader.py
import argparse, logging, os
from lavish_core.trade.trade_agent import run_loop

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL","INFO")))
    ap = argparse.ArgumentParser(description="Lavish Live Trader (exec-only, Patreon+Math)")
    ap.add_argument("--symbols", type=str, required=True, help="Comma list, e.g. AAPL,NVDA,SPY")
    ap.add_argument("--poll", type=int, default=10, help="Polling seconds")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_loop(symbols, poll_seconds=args.poll)