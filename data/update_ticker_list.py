"""
Lavish Bot — Auto Ticker Updater
--------------------------------
Fetches all available tickers from Polygon, Alpha Vantage, and Finnhub APIs,
and saves them to data/tickers.csv and data/tickers.json.
"""

import os
import json
import csv
import requests
from datetime import datetime

POLYGON_API = os.getenv("POLYGON_API_KEY")
FINNHUB_API = os.getenv("FINNHUB_API_KEY")
ALPHAVANTAGE_API = os.getenv("ALPHAVANTAGE_API_KEY")

OUTPUT_CSV = "data/tickers.csv"
OUTPUT_JSON = "data/tickers.json"


def fetch_polygon_tickers():
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&apiKey={POLYGON_API}"
    tickers = []
    while url:
        r = requests.get(url)
        data = r.json()
        results = data.get("results", [])
        for t in results:
            tickers.append(t["ticker"])
        url = data.get("next_url")
        if url and "apiKey" not in url:
            url += f"&apiKey={POLYGON_API}"
    return tickers


def fetch_finnhub_tickers():
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_API}"
    data = requests.get(url).json()
    return [item["symbol"] for item in data]


def fetch_alpha_tickers():
    # Alpha Vantage doesn’t provide a full list, so use a backup source.
    # (You can later replace this with your own maintained list.)
    return ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]


def save_tickers(all_tickers):
    unique = sorted(set(all_tickers))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol"])
        for t in unique:
            writer.writerow([t])

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"tickers": unique}, f, indent=2)

    print(f"✅ Saved {len(unique)} unique tickers.")
    print(f"📁 CSV: {OUTPUT_CSV}")
    print(f"📁 JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    all_tickers = []
    print("🔍 Fetching Polygon tickers...")
    all_tickers += fetch_polygon_tickers()

    print("🔍 Fetching Finnhub tickers...")
    all_tickers += fetch_finnhub_tickers()

    print("🔍 Adding Alpha Vantage defaults...")
    all_tickers += fetch_alpha_tickers()

    save_tickers(all_tickers)
    print(f"🕒 Completed at {datetime.now()}")
