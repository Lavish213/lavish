import os, re, time, pytesseract
from PIL import Image
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =============================
# 🔐 Load Environment Variables
# =============================
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise EnvironmentError("❌ Missing Alpaca API keys in .env file!")

# Initialize trading client
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

RAW_PATH = "lavish_core/king_master/raw"


def ocr_extract(file_path):
    """Extracts text from image and finds stock tickers."""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        print(f"🧠 OCR found in {os.path.basename(file_path)} → {tickers}")
        return tickers
    except Exception as e:
        print(f"❌ OCR failed on {file_path}: {e}")
        return []


def buy_stock(symbol, qty=1):
    """Submits a paper buy order via Alpaca."""
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"✅ Mock trade: {symbol} ({qty} shares)")
    except Exception as e:
        print(f"⚠️ Trade failed for {symbol}: {e}")


def process_images():
    """Continuously scans for new images in raw/ and processes them."""
    print("👑 King Master watching for new images...")
    seen = set()
    while True:
        try:
            files = [f for f in os.listdir(RAW_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                full_path = os.path.join(RAW_PATH, f)
                if f not in seen:
                    tickers = ocr_extract(full_path)
                    for t in tickers:
                        buy_stock(t)
                    seen.add(f)
            time.sleep(10)
        except Exception as e:
            print(f"💥 Error in watcher loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    process_images()