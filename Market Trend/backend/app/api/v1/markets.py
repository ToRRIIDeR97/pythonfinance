from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional

from ...data.tickers import GLOBAL_INDICES
from ...core.db import get_db
from ...repository.data_repository import DataRepository

router = APIRouter()

@router.get("/indices")
def list_indices():
    return GLOBAL_INDICES

@router.get("/indices/{index_ticker}/historical")
def get_index_historical(
    index_ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        data = repo.get_historical(db, ticker=index_ticker, period=period, interval=interval)
        return {
            "ticker": index_ticker,
            "period": period,
            "interval": interval,
            "data": data,
        }
    except Exception as exc:
        # Graceful degradation: return empty data with error info instead of 502
        return {
            "ticker": index_ticker,
            "period": period,
            "interval": interval,
            "data": [],
            "error": f"Failed to fetch historical data: {exc}"
        }
