from datetime import date
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..models.historical import HistoricalPrice


def get_prices(
    db: Session,
    ticker: str,
    start_date: Optional[date] = None,
    interval: str = "1d",
) -> List[HistoricalPrice]:
    stmt = select(HistoricalPrice).where(HistoricalPrice.ticker == ticker)
    if start_date:
        stmt = stmt.where(HistoricalPrice.date >= start_date)
    stmt = stmt.where(HistoricalPrice.interval == interval).order_by(HistoricalPrice.date.asc())
    return list(db.execute(stmt).scalars().all())


def bulk_upsert_prices(db: Session, records: List[dict]):
    if not records:
        return
    # Detect dialect from the session's bind
    dialect_name = getattr(getattr(db, "bind", None), "dialect", None)
    dialect_name = getattr(dialect_name, "name", "")

    if dialect_name == "postgresql":
        # Use ON CONFLICT DO NOTHING with batch insert for performance
        stmt = pg_insert(HistoricalPrice).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date", "interval"])
        db.execute(stmt)
        db.commit()
    else:
        # Fallback: naive per-row upsert for SQLite
        for r in records:
            hp = HistoricalPrice(
                ticker=r["ticker"],
                date=r["date"],
                interval=r.get("interval", "1d"),
                open=r.get("open"),
                high=r.get("high"),
                low=r.get("low"),
                close=r.get("close"),
                volume=r.get("volume"),
            )
            try:
                db.add(hp)
                db.commit()
            except Exception:
                db.rollback()
                # existing row or other error; skip this row and continue
