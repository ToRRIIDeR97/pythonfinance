from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict

from ...core.db import get_db
from ...repository.data_repository import DataRepository

router = APIRouter()


def compute_sma(closes: List[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float] = []
    running_sum = 0.0
    for i, c in enumerate(closes):
        running_sum += c if c is not None else 0.0
        if i >= window:
            prev = closes[i - window]
            running_sum -= prev if prev is not None else 0.0
        if i + 1 >= window:
            out.append(running_sum / window)
        else:
            out.append(None)
    return out


def compute_rsi(closes: List[float], window: int = 14) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(closes)):
        change = (closes[i] or 0.0) - (closes[i - 1] or 0.0)
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = []
    avg_loss = []
    for i in range(len(gains)):
        if i < window:
            avg_gain.append(None)
            avg_loss.append(None)
        elif i == window:
            avg_gain.append(sum(gains[1:window + 1]) / window)
            avg_loss.append(sum(losses[1:window + 1]) / window)
        else:
            prev_avg_gain = avg_gain[-1] if avg_gain[-1] is not None else 0.0
            prev_avg_loss = avg_loss[-1] if avg_loss[-1] is not None else 0.0
            avg_gain.append((prev_avg_gain * (window - 1) + gains[i]) / window)
            avg_loss.append((prev_avg_loss * (window - 1) + losses[i]) / window)
    rsi = []
    for ag, al in zip(avg_gain, avg_loss):
        if ag is None or al is None or al == 0:
            rsi.append(None)
        else:
            rs = ag / al
            rsi.append(100 - (100 / (1 + rs)))
    return rsi


@router.get("/indicators/{ticker}/sma")
def get_sma(
    ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    window: int = Query(14, ge=1, le=365, description="SMA window length"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        data: List[Dict] = repo.get_historical(db, ticker=ticker, period=period, interval=interval)
        closes = [row.get("close") for row in data]
        sma_vals = compute_sma(closes, window)
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "window": window,
            "sma": [
                {"date": data[i].get("date"), "value": sma_vals[i]} for i in range(len(data))
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute SMA: {exc}")


@router.get("/indicators/{ticker}/rsi")
def get_rsi(
    ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    window: int = Query(14, ge=1, le=365, description="RSI window length"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        data: List[Dict] = repo.get_historical(db, ticker=ticker, period=period, interval=interval)
        closes = [row.get("close") for row in data]
        rsi_vals = compute_rsi(closes, window)
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "window": window,
            "rsi": [
                {"date": data[i].get("date"), "value": rsi_vals[i]} for i in range(len(data))
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute RSI: {exc}")


@router.get("/indicators/{ticker}/macd")
def get_macd(
    ticker: str,
    period: str = Query("1y", description="e.g., 1y, 5d, max"),
    interval: str = Query("1d", description="e.g., 1d, 1h, 1m"),
    fast: int = Query(12, ge=1, le=50, description="Fast EMA period"),
    slow: int = Query(26, ge=1, le=100, description="Slow EMA period"),
    signal: int = Query(9, ge=1, le=50, description="Signal line EMA period"),
    db=Depends(get_db),
):
    try:
        repo = DataRepository()
        data: List[Dict] = repo.get_historical(db, ticker=ticker, period=period, interval=interval)
        closes = [row.get("close") for row in data]
        macd_data = repo._macd(closes, fast=fast, slow=slow, signal=signal)
        
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "fast": fast,
            "slow": slow,
            "signal": signal,
            "macd": [
                {
                    "date": data[i].get("date"),
                    "macd": macd_data["macd"][i],
                    "signal": macd_data["signal"][i],
                    "histogram": macd_data["histogram"][i]
                } for i in range(len(data))
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute MACD: {exc}")

