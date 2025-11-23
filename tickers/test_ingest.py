import logging
from datetime import datetime, timezone
from mass_ingest_bot import fetch_recent_1m_bars, upsert_bars, ingest_once
import sqlite3

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_single_ticker(symbol="AAPL"):
    log.info(f"Testing ingest for: {symbol}")

    # Fetch 1-minute bar data for the last 2 hours
    df = fetch_recent_1m_bars([symbol], minutes=120)

    if df.empty:
        log.warning("⚠️ No data returned — maybe API limit or invalid symbol.")
        return

    # Insert into database
    n = upsert_bars(df)
    log.info(f"✅ Inserted {n} rows for {symbol}")

    # Verify saved data
    con = sqlite3.connect("../lavish_ingest.db")
    cur = con.cursor()
    cur.execute(f"SELECT * FROM bars WHERE symbol = ? ORDER BY timestamp DESC LIMIT 3", (symbol,))
    rows = cur.fetchall()
    con.close()

    if rows:
        log.info("✅ Data successfully saved:")
        for r in rows:
            print(r)
    else:
        log.warning("⚠️ No data found in database for that symbol.")

if __name__ == "__main__":
    test_single_ticker("AAPL")
