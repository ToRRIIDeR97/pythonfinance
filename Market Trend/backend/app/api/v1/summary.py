from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict

from ...core.db import get_db
from ...repository.data_repository import DataRepository

router = APIRouter()


@router.get("/summary/{ticker}")
def get_summary(
    ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        result: Dict = repo.get_summary(db, ticker=ticker, period=period, interval=interval)
        if not result:
            # Return explicit null so clients can show "No data" without error banners
            return JSONResponse(content=None, status_code=200)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        # Graceful degradation: return null instead of 5xx to avoid UI error banners
        return JSONResponse(content=None, status_code=200)
