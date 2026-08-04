# system_boot.py
# Pre-flight check run by start_bot.sh before launching run_ingest.py.
#
# This used to check imports/paths for lavish_core.vision.king_master,
# lavish_core.trade.alpaca_handler, and lavish_core.trading.auto_signal_runner -
# none of which are the real live pipeline (that's run_ingest.py, which
# itself doesn't even use alpaca_handler.py or auto_signal_runner.py -
# start_bot.sh's own comment already flags auto_signal_runner.py as
# orphaned). A pre-flight check that validates the wrong modules is worse
# than no check at all - it either prints false failures for things that
# don't matter, or false confidence ("all imports OK!") while missing a
# real problem in the modules that actually run. Rewritten to check what
# actually matters for a real deployment, and to exit non-zero on a real
# problem so start_bot.sh can abort instead of launching into a broken run.
from __future__ import annotations
import importlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

print("Running pre-flight check...\n")
ok = True

# 1. .env - anchored to this script's own directory (see run_ingest.py's
# identical comment), not load_dotenv()'s cwd-dependent default.
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    before = set(os.environ)
    load_dotenv(_env_path)
    loaded = len(set(os.environ) - before)
    print(f"[ok] .env found at {_env_path} ({loaded} new key(s) loaded from it).")
else:
    print(f"[warn] No .env at {_env_path} - relying on whatever's already in the shell environment.")

# 2. Required credentials actually present (not just "the var exists with an empty default")
required = {
    "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
}
has_patreon_static = bool(os.getenv("PATREON_ACCESS_TOKEN"))
has_patreon_refresh = all(os.getenv(k) for k in ("PATREON_CLIENT_ID", "PATREON_CLIENT_SECRET", "PATREON_REFRESH_TOKEN"))

for name, val in required.items():
    if val:
        print(f"[ok] {name} is set.")
    else:
        print(f"[FAIL] {name} is missing - copy env.example to .env and fill it in.")
        ok = False

if has_patreon_static or has_patreon_refresh:
    print("[ok] Patreon credentials present.")
else:
    print("[FAIL] No Patreon credentials - need either PATREON_ACCESS_TOKEN, or all three of "
          "PATREON_CLIENT_ID/PATREON_CLIENT_SECRET/PATREON_REFRESH_TOKEN.")
    ok = False

trade_mode = os.getenv("TRADE_MODE", "dry").lower()
if trade_mode not in ("dry", "paper", "live"):
    print(f"[FAIL] TRADE_MODE={trade_mode!r} is not dry/paper/live.")
    ok = False
else:
    print(f"[ok] TRADE_MODE={trade_mode}"
          + (" - will submit REAL orders with real money." if trade_mode == "live" else ""))
if trade_mode in ("paper", "live") and "paper-api" not in os.getenv("ALPACA_BASE_URL", "") and trade_mode == "paper":
    print(f"[warn] TRADE_MODE=paper but ALPACA_BASE_URL doesn't look like the paper endpoint "
          f"({os.getenv('ALPACA_BASE_URL')!r}) - double check this isn't pointed at live by mistake.")

# 3. The real live modules actually import cleanly
modules = [
    "run_ingest",
    "lavish_core.patreon.patreon_trigger",
    "lavish_core.discord_ingest.listener",
    "lavish_core.trading.alert_handler",
    "lavish_core.trading.trade_handler",
    "lavish_core.trade.broker_alpaca",
    "lavish_core.trade.options_broker",
    "lavish_core.trade.circuit_breaker",
    "lavish_core.trade.reconcile",
    "lavish_core.vision.extract_signal",
    "lavish_core.db.hybrid_store",
]
for module in modules:
    try:
        importlib.import_module(module)
        print(f"[ok] import {module}")
    except Exception as e:
        print(f"[FAIL] import {module}: {e}")
        ok = False

# 4. tesseract the system BINARY, not just the pytesseract python package -
# a very common real-deployment gap (pip install succeeds, OCR silently
# fails at runtime because the actual `tesseract` executable isn't on PATH)
if shutil.which("tesseract"):
    print("[ok] tesseract binary found on PATH.")
else:
    print("[FAIL] tesseract binary not found - apt install tesseract-ocr (or your distro's equivalent). "
          "pip installing pytesseract alone is not enough, it's a wrapper around this binary.")
    ok = False

print()
if ok:
    print("Pre-flight check passed - ready for launch.")
    sys.exit(0)
else:
    print("Pre-flight check FAILED - fix the [FAIL] items above before starting the bot.")
    sys.exit(1)
