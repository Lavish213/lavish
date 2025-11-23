import asyncio, csv, json, os, time
from datetime import datetime
from pathlib import Path

import aiohttp
from config import DATA_DIR, FINNHUB_API, ALPHA_API, FRED_API, ALPHA_REQS_PER_MIN, FINNHUB_CONCURRENCY, HTTP_TIMEOUT

LOG_FILE = DATA_DIR / "logs.txt"

def log(msg: str):
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else "") + f"[{datetime.now()}] {msg}\n")

def save_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

async def _get_json(session: aiohttp.ClientSession, url: str, params=None, max_retries=3):
    params = params or {}
    for attempt in range(1, max_retries+1):
        try:
            async with session.get(url, params=params, timeout=HTTP_TIMEOUT) as resp:
                if resp.status == 429:  # rate limit
                    retry_after = int(resp.headers.get("Retry-After", "2"))
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return await resp.json(content_type=None)
        except Exception as e:
            if attempt == max_retries:
                log(f"ERROR GET {url} params={params} -> {e}")
                return None
            await asyncio.sleep(1.5 * attempt)

# ------------- FINNHUB -----------------

async def fetch_finnhub_tickers(session: aiohttp.ClientSession, exchange="US"):
    url = "https://finnhub.io/api/v1/stock/symbol"
    data = await _get_json(session, url, {"exchange": exchange, "token": FINNHUB_API})
    rows = []
    if isinstance(data, list):
        for d in data:
            sym = d.get("symbol")
            if sym and sym.isalpha():  # crude filter
                rows.append({
                    "symbol": sym,
                    "description": d.get("description", ""),
                    "type": d.get("type", ""),
                    "currency": d.get("currency", "")
                })
    save_csv(DATA_DIR / "tickers.csv", ["symbol","description","type","currency"], rows)
    log(f"Finnhub: saved {len(rows)} tickers")
    return [r["symbol"] for r in rows]

async def fetch_finnhub_news(session: aiohttp.ClientSession, category="general", per_page=200):
    url = "https://finnhub.io/api/v1/news"
    data = await _get_json(session, url, {"category": category, "token": FINNHUB_API})
    rows = []
    if isinstance(data, list):
        for n in data[:per_page]:
            rows.append({
                "headline": n.get("headline",""),
                "datetime": n.get("datetime",""),
                "source": n.get("source",""),
                "url": n.get("url",""),
                "summary": n.get("summary","")
            })
    save_csv(DATA_DIR / "news.csv", ["headline","datetime","source","url","summary"], rows)
    log(f"Finnhub: saved {len(rows)} news rows")
    return rows

# ------------- FRED -----------------

async def fetch_fred_series(session: aiohttp.ClientSession, series_id="GDP"):
    url = "https://api.stlouisfed.org/fred/series/observations"
    data = await _get_json(session, url, {"series_id": series_id, "api_key": FRED_API, "file_type": "json"})
    obs = (data or {}).get("observations", [])
    rows = [{"date": o.get("date",""), "value": o.get("value","")} for o in obs]
    save_csv(DATA_DIR / "fred_data.csv", ["date","value"], rows)
    log(f"FRED: saved {len(rows)} observations ({series_id})")
    return rows

# ------------- ALPHA VANTAGE -----------------

class AlphaLimiter:
    """Simple minute bucket limiter for Alpha Vantage."""
    def __init__(self, per_minute=ALPHA_REQS_PER_MIN):
        self.per_min = per_minute
        self.ts = []
    async def acquire(self):
        now = time.time()
        # drop old
        self.ts = [t for t in self.ts if now - t < 60]
        if len(self.ts) >= self.per_min:
            await asyncio.sleep(60 - (now - self.ts[0]) + 0.1)
        self.ts.append(time.time())

async def fetch_alpha_daily_one(session: aiohttp.ClientSession, limiter: AlphaLimiter, symbol: str):
    await limiter.acquire()
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": ALPHA_API}
    data = await _get_json(session, url, params)
    ts = (data or {}).get("Time Series (Daily)", {})
    rows = []
    for day, v in ts.items():
        rows.append({
            "symbol": symbol,
            "date": day,
            "open": v.get("1. open",""),
            "high": v.get("2. high",""),
            "low": v.get("3. low",""),
            "close": v.get("4. close",""),
            "volume": v.get("5. volume","")
        })
    return rows

async def fetch_alpha_daily_batch(session: aiohttp.ClientSession, symbols: list[str]):
    limiter = AlphaLimiter()
    results = []
    # keep it modest concurrency due to Alpha anyway
    sem = asyncio.Semaphore(3)
    async def one(sym):
        async with sem:
            rows = await fetch_alpha_daily_one(session, limiter, sym)
            if rows:
                results.extend(rows)
    await asyncio.gather(*(one(s) for s in symbols[:50]))  # cap to first N to avoid huge runs
    if results:
        save_csv(DATA_DIR / "alpha_data.csv",
                 ["symbol","date","open","high","low","close","volume"], results)
        log(f"Alpha: saved {len(results)} daily rows (batch)")
    return results

# ------------- MASTER UPDATE -----------------

async def update_all_async(symbols_for_alpha: list[str] | None = None):
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT + 10)
    connector = aiohttp.TCPConnector(limit=FINNHUB_CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # always get new tickers/news
        tickers_task = asyncio.create_task(fetch_finnhub_tickers(session, "US"))
        news_task    = asyncio.create_task(fetch_finnhub_news(session, "general"))
        fred_task    = asyncio.create_task(fetch_fred_series(session, "GDP"))

        tickers = await tickers_task
        await asyncio.gather(news_task, fred_task)

        # Decide which symbols to pull from Alpha:
        if symbols_for_alpha is None:
            # take a small diversified slice to respect Alpha limits
            symbols_for_alpha = (tickers or ["AAPL","MSFT","NVDA","TSLA","AMZN"])[:10]

        await fetch_alpha_daily_batch(session, symbols_for_alpha)

def update_all():
    try:
        asyncio.run(update_all_async())
        log("✅ Daily async refresh complete.\n")
    except Exception as e:
        log(f"❌ update_all_async failed: {e}")

if __name__ == "__main__":
    update_all()