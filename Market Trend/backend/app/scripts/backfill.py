import os
import sys
import time
import argparse

# Add the backend directory to the python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.db import SessionLocal
from app.data.tickers import GLOBAL_INDICES, US_SECTOR_ETFS
from app.repository.data_repository import DataRepository
from app.services.logging_service import LoggingService


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill historical data for tickers")
    parser.add_argument("--type", choices=["indices", "etfs", "all"], default="all", help="Which tickers to backfill")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g., 5d, 1y, max")
    parser.add_argument("--interval", default="1d", help="yfinance interval, e.g., 1d, 1h, 1m")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between tickers to respect rate limits")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers (0 means no limit)")
    parser.add_argument(
        "--disable-redis-cache",
        action="store_true",
        help="Bypass Redis cache during backfill to ensure DB writes",
    )
    return parser.parse_args()


def main():
    """
    Seeds the database with historical market data for all tickers.
    Fetches 5 years of daily data for each ticker.
    """
    logger = LoggingService()
    logger.log_info("Starting historical data backfill...")
    args = parse_args()
    db = SessionLocal()
    if args.disable_redis_cache:
        # Construct a repository with an in-memory cache only (no Redis), avoiding cross-process cache hits
        from app.services.cache import CacheService
        repo = DataRepository(cache=CacheService("redis://disabled"))
    else:
        repo = DataRepository()

    if args.type == "indices":
        tickers_to_seed = GLOBAL_INDICES
    elif args.type == "etfs":
        tickers_to_seed = US_SECTOR_ETFS
    else:
        tickers_to_seed = GLOBAL_INDICES + US_SECTOR_ETFS

    if args.limit and args.limit > 0:
        tickers_to_seed = tickers_to_seed[:args.limit]
    total_tickers = len(tickers_to_seed)

    for i, ticker_info in enumerate(tickers_to_seed):
        ticker = ticker_info["ticker"]
        name = ticker_info["name"]
        logger.log_info(
            f"[{i + 1}/{total_tickers}] Backfilling data for {ticker} ({name})..."
        )
        try:
            # Force-fetch from provider and persist, ensuring full coverage beyond existing DB rows.
            repo.backfill_historical(db, ticker=ticker, period=args.period, interval=args.interval)
            logger.log_info(f"Successfully backfilled data for {ticker}.")
            # Be respectful to the yfinance API to avoid getting blocked.
            time.sleep(args.sleep)
        except Exception as e:
            logger.log_error(f"Failed to backfill data for {ticker}", e)

    db.close()
    logger.log_info("Historical data backfill complete.")


if __name__ == "__main__":
    main()
