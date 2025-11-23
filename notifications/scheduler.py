import sys
import os
import logging
from datetime import datetime, timedelta
import time

# --- FIX: Ensure Lavish_bot project root is on the import path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import your prediction function from ml/predictor.py ---
from ml.predictor import predict_tomorrow_close

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)

def run_daily_task():
    """Run the stock predictor once per day."""
    logging.info("Starting daily scheduler for stock prediction...")

    while True:
        now = datetime.now()
        # Run every day at 6:00 PM (adjust as needed)
        run_time = now.replace(hour=18, minute=0, second=0, microsecond=0)

        # If it’s past 6:00 PM, schedule for next day
        if now > run_time:
            run_time += timedelta(days=1)

        wait_seconds = (run_time - now).total_seconds()
        logging.info(f"Next prediction run scheduled at {run_time} (in {wait_seconds/3600:.2f} hours)")

        time.sleep(wait_seconds)

        try:
            logging.info("Running daily prediction task...")
            predict_tomorrow_close()
            logging.info("Prediction task completed successfully.")
        except Exception as e:
            logging.error(f"Error running prediction task: {e}", exc_info=True)

        # Wait 24 hours before running again
        time.sleep(86400)

if __name__ == "__main__":
    run_daily_task()