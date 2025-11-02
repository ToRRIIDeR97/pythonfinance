from fastapi import APIRouter, HTTPException, Query, Depends
from ...data.tickers import US_SECTOR_ETFS
from ...core.db import get_db
from ...repository.data_repository import DataRepository

router = APIRouter()

@router.get("/sectors/etfs")
def list_sector_etfs():
    return US_SECTOR_ETFS

@router.get("/sectors/etfs/{etf_ticker}/historical")
def get_etf_historical(
    etf_ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        data = repo.get_historical(db, ticker=etf_ticker, period=period, interval=interval)
        return {
            "ticker": etf_ticker,
            "period": period,
            "interval": interval,
            "data": data,
        }
    except Exception as exc:
        # Graceful degradation: return empty data with error info instead of 502
        return {
            "ticker": etf_ticker,
            "period": period,
            "interval": interval,
            "data": [],
            "error": f"Failed to fetch historical data: {exc}"
        }
