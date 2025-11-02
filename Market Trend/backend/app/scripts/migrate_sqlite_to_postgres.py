import os
import sys
import argparse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Add the backend directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.models.historical import HistoricalPrice
from app.crud.historical import bulk_upsert_prices
from app.services.logging_service import LoggingService


def parse_args():
    p = argparse.ArgumentParser(description="Migrate historical data from SQLite to PostgreSQL")
    p.add_argument("--sqlite", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "market_trend.db")), help="Path to source SQLite file")
    p.add_argument("--postgres", required=True, help="Destination PostgreSQL DATABASE_URL (sqlalchemy URL)")
    p.add_argument("--batch", type=int, default=5000, help="Batch size for upserts")
    return p.parse_args()


def main():
    args = parse_args()
    log = LoggingService()

    log.log_info(f"Connecting to SQLite at {args.sqlite}")
    sqlite_engine = create_engine(f"sqlite:///{args.sqlite}", connect_args={"check_same_thread": False})
    SQLiteSession = sessionmaker(bind=sqlite_engine)

    log.log_info(f"Connecting to Postgres at {args.postgres}")
    pg_engine = create_engine(args.postgres)
    PostgresSession = sessionmaker(bind=pg_engine)

    src = SQLiteSession()
    dst = PostgresSession()

    try:
        # Ensure destination table exists
        HistoricalPrice.metadata.create_all(bind=pg_engine)

        # Stream rows from SQLite
        stmt = select(HistoricalPrice)
        result = src.execute(stmt).scalars()
        batch = []
        total = 0
        for row in result:
            batch.append({
                "ticker": row.ticker,
                "date": row.date,
                "interval": row.interval,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            })
            if len(batch) >= args.batch:
                bulk_upsert_prices(dst, batch)
                total += len(batch)
                log.log_info(f"Migrated {total} rows...")
                batch.clear()
        if batch:
            bulk_upsert_prices(dst, batch)
            total += len(batch)
        log.log_info(f"Migration complete. Total rows migrated: {total}")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()

