from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

account = api.get_account()
print("Account Status:", account.status)
print("Alpaca base URL:", os.getenv("ALPACA_BASE_URL"))

# Submitting an order here is destructive (real paper/live order), so it only
# runs when explicitly requested and never against a live (non-paper) account.
if os.getenv("RUN_DEMO_ORDER", "0") == "1":
    base_url = (os.getenv("ALPACA_BASE_URL") or "").lower()
    if "paper-api" not in base_url:
        raise SystemExit(
            "Refusing to submit demo order: ALPACA_BASE_URL does not look like "
            "a paper-trading endpoint. Set RUN_DEMO_ORDER=1 only against paper-api.alpaca.markets."
        )
    order = api.submit_order(
        symbol="AAPL",
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc",
    )
    print("Demo order submitted:", order)
else:
    print("RUN_DEMO_ORDER not set to 1 — skipping demo order submission.")
