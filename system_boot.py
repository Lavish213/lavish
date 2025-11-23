import os
import importlib
from dotenv import load_dotenv

print("🧠 Running System Boot Check...\n")

# 1. Check environment file
if os.path.exists(".env"):
    load_dotenv()
    keys = [k for k in os.environ.keys() if k.isupper()]
    print(f"✅ .env found with {len(keys)} keys.")
else:
    print("❌ .env file missing!")

# 2. Verify key imports
modules = [
    "lavish_core.vision.king_master",
    "lavish_core.patreon.patreon_trigger",
    "lavish_core.trade.alpaca_handler",
    "lavish_core.trading.auto_signal_runner",
]

for module in modules:
    try:
        importlib.import_module(module)
        print(f"✅ Import OK: {module}")
    except Exception as e:
        print(f"❌ Import failed for {module}: {e}")

# 3. Confirm critical paths
paths = [
    "lavish_core/trading/auto_signal_runner.py",
    "lavish_core/trade/alpaca_handler.py",
]

for p in paths:
    if os.path.exists(p):
        print(f"📁 Found: {p}")
    else:
        print(f"⚠️ Missing: {p}")

print("\n✅ System Boot Check complete — ready for launch!")