import os
import sys
from datetime import date, timedelta

# Add the backend directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.db import SessionLocal
from app.data.tickers import GLOBAL_INDICES, US_SECTOR_ETFS
from app.crud.historical import get_prices
from app.services.logging_service import LoggingService


def _count_trading_days(start: date, end: date) -> int:
    days = 0
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Monday=0 .. Friday=4
            days += 1
        cur += timedelta(days=1)
    return days


def validate_ticker(db, ticker: str, days: int = 365, interval: str = "1d") -> dict:
    start_date = date.today() - timedelta(days=days)
    rows = get_prices(db, ticker=ticker, start_date=start_date, interval=interval)
    count = len(rows)
    expected_trading = _count_trading_days(start_date, date.today())
    coverage_pct = round(100.0 * count / expected_trading, 2) if expected_trading > 0 else 0.0
    first_date = rows[0].date.isoformat() if rows else None
    last_date = rows[-1].date.isoformat() if rows else None
    return {
        "ticker": ticker,
        "rows": count,
        "days": days,
        "coverage_pct": coverage_pct,
        "first_date": first_date,
        "last_date": last_date,
    }


def main():
    log = LoggingService()
    db = SessionLocal()
    try:
        tickers = [t["ticker"] for t in (GLOBAL_INDICES + US_SECTOR_ETFS)]
        log.log_info(f"Validating {len(tickers)} tickers for last 1y trading-day coverage")
        summary = []
        for t in tickers:
            res = validate_ticker(db, t, days=252, interval="1d")
            summary.append(res)
            log.log_info(
                f"{t}: rows={res['rows']} coverage={res['coverage_pct']}% first={res['first_date']} last={res['last_date']}"
            )
        # Simple reconciliation: flag any ticker with < 80% coverage
        flagged = [r for r in summary if r["coverage_pct"] < 80.0]
        if flagged:
            log.log_warn(f"Found {len(flagged)} tickers with low coverage (<80%).")
            for r in flagged:
                log.log_warn(f"Low coverage: {r['ticker']} {r['coverage_pct']}%")
        else:
            log.log_info("All tickers have acceptable coverage.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
