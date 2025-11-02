from sqlalchemy import Column, Integer, String, Date, Float, Index
from ..core.db import Base


class HistoricalPrice(Base):
    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    interval = Column(String, default="1d", index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        Index("ix_hist_unique", "ticker", "date", "interval", unique=True),
    )