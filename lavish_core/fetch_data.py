import os, csv, json, requests, datetime, concurrent.futures
from config import FINNHUB_API, ALPHA_API, FRED_API
from pathlib import Path

# === Setup ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def log(msg):
    with open(DATA_DIR / "logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")

# === Finnhub ===
def fetch_finnhub_tickers(exchange="US"):
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={FINNHUB_API}"
    res = requests.get(url, timeout=15)
    data = res.json()
    tickers = [{"symbol": d["symbol"], "description": d["description"]} for d in data]
    save_csv(DATA_DIR / "tickers.csv", ["symbol", "description"], tickers)
    log(f"✅ Saved {len(tickers)} tickers from Finnhub.")
    return tickers

def fetch_finnhub_news():
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API}"
    res = requests.get(url, timeout=15)
    data = res.json()
    rows = [
        {"headline": n["headline"], "datetime": n["datetime"], "source": n["source"], "url": n["url"]}
        for n in data
    ]
    save_csv(DATA_DIR / "news.csv", ["headline", "datetime", "source", "url"], rows)
    log(f"📰 Saved {len(rows)} news articles.")
    return rows

# === FRED ===
def fetch_fred_data(series="GDP"):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_API}&file_type=json"
    res = requests.get(url, timeout=15)
    data = res.json().get("observations", [])
    rows = [{"date": d["date"], "value": d["value"]} for d in data]
    save_csv(DATA_DIR / "fred_data.csv", ["date", "value"], rows)
    log(f"🏦 Saved FRED data for {series}.")
    return rows

# === Alpha Vantage ===
def fetch_alpha_data(symbol="AAPL"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API}"
    res = requests.get(url, timeout=15)
    data = res.json().get("Time Series (Daily)", {})
    rows = [
        {
            "date": k,
            "open": v["1. open"],
            "high": v["2. high"],
            "low": v["3. low"],
            "close": v["4. close"],
            "volume": v["5. volume"]
        }
        for k, v in data.items()
    ]
    save_csv(DATA_DIR / f"{symbol}_alpha.csv", ["date", "open", "high", "low", "close", "volume"], rows)
    log(f"📊 Saved AlphaVantage data for {symbol}.")
    return rows

# === Run all in parallel ===
def update_all():
    try:
        log("🚀 Starting daily data update...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = {
                executor.submit(fetch_finnhub_tickers): "Tickers",
                executor.submit(fetch_finnhub_news): "News",
                executor.submit(fetch_fred_data): "FRED",
                executor.submit(fetch_alpha_data): "Alpha"
            }
            for future in concurrent.futures.as_completed(tasks):
                name = tasks[future]
                try:
                    future.result()
                    log(f"✅ {name} data updated successfully.")
                except Exception as e:
                    log(f"❌ {name} failed: {e}")
        log("🎯 All data updated successfully!\n")
    except Exception as e:
        log(f"💥 Fatal error: {e}")

# === Auto-run when executed directly ===
if __name__ == "__main__":
    update_all()
    print("✅ All data fetched. Logs saved in data/logs.txt")