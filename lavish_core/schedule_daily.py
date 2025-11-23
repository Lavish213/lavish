import schedule, time
from fetch_data import update_all
from datetime import datetime

def job():
    print(f"Running daily data update at {datetime.now()}")
    update_all()

# Run every weekday (Mon–Fri) at 09:00 ET (06:00 PT)
schedule.every().monday.at("09:00").do(job)
schedule.every().tuesday.at("09:00").do(job)
schedule.every().wednesday.at("09:00").do(job)
schedule.every().thursday.at("09:00").do(job)
schedule.every().friday.at("09:00").do(job)

print("⏰ Scheduler running... Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(30)
