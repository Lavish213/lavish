#!/bin/bash
echo "🚀 Starting Lavish Bot System..."


# Activate virtual environment
source venv/bin/activate

# Run system check
python system_boot.py

# Run bot
python lavish_core/trading/auto_signal_runner.py