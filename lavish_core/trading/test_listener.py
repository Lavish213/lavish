from __future__ import annotations
import time, threading
from pathlib import Path
from lavish_core.logger_setup import get_logger
from lavish_core.trading.alpaca_handler import execute_trade_from_post

# --- CONFIG ---
TEST_DIR = Path("lavish_core/trading/test_drops")
POLL_SECONDS = 5
TRADE_AMOUNT = 50
SYMBOL = "TEST"

# --- LOGGING ---
log = get_logger("test_listener", log_dir="lavish_core/logs")
SEEN = set()

def handle_image(img_path: Path):
    """Trigger a simulated trade whenever a new image appears."""
    if img_path.name in SEEN:
        return  # prevent double-triggers for the same file
    SEEN.add(img_path.name)

    log.info(f"📸 Triggered trade from: {img_path.name}")
    print(f"\033[92m💸 BUY signal triggered from {img_path.name}\033[0m")

    # Paper trade trigger
    execute_trade_from_post({
        "source": "test_listener",
        "action": "BUY",
        "symbol": SYMBOL,
        "confidence": 0.95,
        "amount_usd": TRADE_AMOUNT,
        "note": f"image_trigger:{img_path.name}"
    })

def listener_loop():
    log.info("👀 Test Listener active – drop a pic into test_drops/")
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            for img_path in TEST_DIR.iterdir():
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    handle_image(img_path)
        except Exception as e:
            log.warning(f"Listener loop error: {e}")
        time.sleep(POLL_SECONDS)

def start_test_listener():
    """Run listener in background."""
    t = threading.Thread(target=listener_loop, daemon=True)
    t.start()
    log.info("🚀 Test listener started successfully.")

def main():
    start_test_listener()
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()