#!/bin/bash
echo "🚀 Starting Lavish Bot System..."


# Activate virtual environment
source venv/bin/activate

# Run system check
python system_boot.py

# Run bot: watches Patreon + Discord (if configured) and executes trades.
# NOTE: this used to run lavish_core/trading/auto_signal_runner.py - an
# older, separate implementation with no options support and no relation
# to the trade_handler/broker_alpaca/circuit_breaker pipeline this repo
# actually uses now. That script still exists but nothing points at it
# anymore; run_ingest.py is the real entry point.
python run_ingest.py