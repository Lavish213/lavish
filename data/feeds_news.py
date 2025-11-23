# data/feeds_news.py
import asyncio
from lavish_core.data.news_failover import NewsFailover
from lavish_core.config import get_settings

def main():
    s = get_settings()
    nf = NewsFailover()
    asyncio.run(nf.fetch_and_save_min(min_rows=s.NEWS_MIN_ROWS))

if __name__ == "__main__":
    main()