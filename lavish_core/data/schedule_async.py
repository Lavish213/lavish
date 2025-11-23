from datetime import datetime
import asyncio, time, zoneinfo
from fetch_data_async import update_all

ET = zoneinfo.ZoneInfo("America/New_York")

def should_run_now():
    now = datetime.now(ET)
    # Weekdays, run at 09:00 ET (≈30 min before cash open)
    return now.weekday() < 5 and now.hour == 9 and now.minute == 0

print("⏰ Async scheduler running… (Ctrl+C to stop)")
while True:
    if should_run_now():
        print(f"Running daily data update at {datetime.now(ET)}")
        update_all()
        # sleep ~60s so we don’t re-trigger within the minute
        time.sleep(65)
    time.sleep(5)