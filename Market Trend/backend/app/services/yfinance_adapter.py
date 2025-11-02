from typing import List, Dict
import datetime

import yfinance as yf
import pandas as pd
from ..services.logging_service import LoggingService
from ..services.rate_limiter import TokenBucket
from ..core.config import settings
import time

_logger = LoggingService()
_bucket = TokenBucket(rate=getattr(settings, "yfinance_rate", 30), per_seconds=getattr(settings, "yfinance_per_seconds", 60))


def download_historical(symbol: str, period: str = "1y", interval: str = "1d") -> List[Dict]:
    """Download OHLCV historical data for a symbol using yfinance.

    Returns a list of records: {date, open, high, low, close, volume}
    """
    try:
        # Rate limit to avoid getting blocked by upstream
        while not _bucket.consume():
            time.sleep(0.1)

        # Retryable primary fetch
        df = None
        max_retries = getattr(settings, "yfinance_max_retries", 3)
        backoff_base = getattr(settings, "yfinance_backoff_base_seconds", 0.5)

        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(symbol, period=period, interval=interval, progress=False)
                if df is not None and not df.empty:
                    break
                _logger.log_warn(f"Empty dataframe from yf.download for {symbol} (attempt {attempt}/{max_retries})")
            except Exception as e:
                _logger.log_warn(f"Error in yf.download for {symbol} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** (attempt - 1)))

        # Fallback to per-ticker history API with retries if primary failed or returned empty
        if df is None or df.empty:
            tkr = yf.Ticker(symbol)
            for attempt in range(1, max_retries + 1):
                try:
                    df = tkr.history(period=period, interval=interval, auto_adjust=False)
                    if df is not None and not df.empty:
                        break
                    _logger.log_warn(f"Empty dataframe from Ticker.history for {symbol} (attempt {attempt}/{max_retries})")
                except Exception as e:
                    _logger.log_warn(f"Error in Ticker.history for {symbol} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(backoff_base * (2 ** (attempt - 1)))
            if df is None or df.empty:
                return []
        # Normalize columns: handle MultiIndex from yfinance when using download()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # If the ticker symbol is one of the column levels, select it
                level_values = [str(v) for v in df.columns.get_level_values(-1)]
                if symbol in level_values:
                    df = df.xs(symbol, axis=1, level=-1, drop_level=True)
                else:
                    # Fallback: flatten by joining levels
                    df.columns = ["_".join([str(x) for x in tup]) for tup in df.columns]
            except Exception:
                # If normalization fails, proceed with current structure
                pass
        # Use index as the date source to avoid column name variability
        if df is None or df.empty:
            return []
        records = []
        for idx, row in df.iterrows():
            date_val = idx
            if isinstance(date_val, (datetime.date, datetime.datetime, pd.Timestamp)):
                date_str = date_val.isoformat()
            else:
                # If date is missing or invalid, skip row
                if date_val in (None, "NaT") or (hasattr(pd, "isna") and pd.isna(date_val)):
                    continue
                date_str = str(date_val)
            records.append(
                {
                    "date": date_str,
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                }
            )
        return records
    except Exception as exc:
        # Let callers translate to appropriate HTTP errors
        raise exc
