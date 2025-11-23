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
# Example: Buy 1 share of AAPL in paper trading
api.submit_order(
    symbol="AAPL",
    qty=1,
    side="buy",
    type="market",
    time_in_force="gtc"
)
print("✅ Test trade submitted!")
# ---- Test Trade ----
order = api.submit_order(
    symbol="AAPL",      # stock to trade
    qty=1,              # number of shares
    side="buy",         # buy or sell
    type="market",      # market order
    time_in_force="gtc" # good till canceled
)
print("✅ Test order sent:", order)
