#!/bin/bash
set -e
echo "Starting Lavish Bot System..."

# Activate virtual environment
source venv/bin/activate

# Pre-flight check - aborts (non-zero exit) on a real problem (missing
# credentials, a broken import, tesseract not installed) instead of
# launching into a run that's going to fail or silently do nothing.
if ! python system_boot.py; then
    echo "Pre-flight check failed - not starting the bot. See errors above."
    exit 1
fi

# Run bot: watches Patreon + Discord (if configured) and executes trades.
# NOTE: this used to run lavish_core/trading/auto_signal_runner.py - an
# older, separate implementation with no options support and no relation
# to the trade_handler/broker_alpaca/circuit_breaker pipeline this repo
# actually uses now. That script still exists but nothing points at it
# anymore; run_ingest.py is the real entry point.
python run_ingest.py