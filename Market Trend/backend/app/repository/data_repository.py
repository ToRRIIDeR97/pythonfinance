from datetime import date, timedelta
from typing import List, Dict, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.db import get_db
from ..crud.historical import get_prices, bulk_upsert_prices
from ..services.cache import CacheService
from ..services.logging_service import LoggingService
from ..services.yfinance_adapter import download_historical
from ..services.logging_service import LoggingService


def period_to_days(period: str) -> int:
    mapping = {
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "max": 36500,
    }
    return mapping.get(period, 365)


class DataRepository:
    def __init__(self, cache: Optional[CacheService] = None, logger: Optional[LoggingService] = None):
        self.cache = cache or CacheService(settings.redis_url)
        self.logger = logger or LoggingService()

    def get_historical(self, db: Session, ticker: str, period: str = "1y", interval: str = "1d") -> List[Dict]:
        cache_key = f"hist:{ticker}:{period}:{interval}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # DB lookup
        start_days = period_to_days(period)
        start_date = date.today() - timedelta(days=start_days)
        db_rows = get_prices(db, ticker=ticker, start_date=start_date, interval=interval)
        if db_rows:
            result = [
                {
                    "date": r.date.isoformat(),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in db_rows
            ]
            self.cache.set(cache_key, result, ttl=300)
            return result

        # Fetch via yfinance and persist
        try:
            self.logger.log_info(f"Fetching yfinance data for {ticker} {period} {interval}")
            records = download_historical(ticker, period=period, interval=interval)
            to_store = []
            for row in records:
                ds = row.get("date")
                try:
                    parsed_date = date.fromisoformat(ds[:10]) if ds else None
                except Exception:
                    self.logger.log_warn(f"Skipping row with invalid date: {ds}")
                    continue
                if not parsed_date:
                    self.logger.log_warn("Skipping row due to missing date")
                    continue
                to_store.append(
                    {
                        "ticker": ticker,
                        "date": parsed_date,
                        "interval": interval,
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "volume": row.get("volume"),
                    }
                )
            if to_store:
                bulk_upsert_prices(db, to_store)
            self.cache.set(cache_key, records, ttl=300)
            return records
        except Exception as exc:
            self.logger.log_error("yfinance fetch failed", exc)
            # Graceful degradation: cache and return an empty dataset instead of raising.
            # This avoids 5xx responses upstream and lets clients render a fallback UI.
            self.cache.set(cache_key, [], ttl=120)
            return []

    def _sma_last(self, values: List[Optional[float]], window: int) -> Optional[float]:
        if window <= 0:
            return None
        running_sum = 0.0
        last_val: Optional[float] = None
        for i, c in enumerate(values):
            running_sum += c if c is not None else 0.0
            if i >= window:
                prev = values[i - window]
                running_sum -= prev if prev is not None else 0.0
            if i + 1 >= window:
                last_val = running_sum / window
        return last_val

    def _rsi_last(self, values: List[Optional[float]], window: int = 14) -> Optional[float]:
        if window <= 0:
            return None
        # Build gains/losses
        gains = [0.0]
        losses = [0.0]
        for i in range(1, len(values)):
            prev = values[i - 1] or 0.0
            curr = values[i] or 0.0
            change = curr - prev
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain: List[Optional[float]] = []
        avg_loss: List[Optional[float]] = []
        for i in range(len(gains)):
            if i < window:
                avg_gain.append(None)
                avg_loss.append(None)
            elif i == window:
                avg_gain.append(sum(gains[1:window + 1]) / window)
                avg_loss.append(sum(losses[1:window + 1]) / window)
            else:
                pg = avg_gain[-1] if avg_gain[-1] is not None else 0.0
                pl = avg_loss[-1] if avg_loss[-1] is not None else 0.0
                avg_gain.append((pg * (window - 1) + gains[i]) / window)
                avg_loss.append((pl * (window - 1) + losses[i]) / window)
        if not avg_gain or not avg_loss:
            return None
        ag = avg_gain[-1]
        al = avg_loss[-1]
        if ag is None or al is None or al == 0:
            return None
        rs = ag / al
        return 100 - (100 / (1 + rs))

    def _ema(self, values: List[Optional[float]], window: int) -> List[Optional[float]]:
        """Calculate Exponential Moving Average for a series of values."""
        if window <= 0:
            return [None] * len(values)
        
        alpha = 2.0 / (window + 1)
        ema_values = []
        
        for i, value in enumerate(values):
            if value is None:
                ema_values.append(None)
                continue
                
            if i == 0:
                ema_values.append(value)
            else:
                prev_ema = ema_values[-1]
                if prev_ema is None:
                    ema_values.append(value)
                else:
                    ema_values.append(alpha * value + (1 - alpha) * prev_ema)
        
        return ema_values

    def _macd(self, values: List[Optional[float]], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence) indicator."""
        if len(values) < slow:
            return {"macd": [], "signal": [], "histogram": []}
        
        # Calculate EMAs
        ema_fast = self._ema(values, fast)
        ema_slow = self._ema(values, slow)
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = []
        for i in range(len(values)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line.append(ema_fast[i] - ema_slow[i])
            else:
                macd_line.append(None)
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self._ema(macd_line, signal)
        
        # Calculate histogram (MACD - Signal)
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(None)
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    def get_summary(self, db: Session, ticker: str, period: str = "1y", interval: str = "1d") -> Dict:
        """Compute key statistics from historical data and cache the result."""
        cache_key = f"summary:{ticker}:{period}:{interval}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self.get_historical(db, ticker=ticker, period=period, interval=interval)
        if not data:
            return {}

        closes: List[Optional[float]] = [row.get("close") for row in data]
        highs: List[Optional[float]] = [row.get("high") for row in data]
        lows: List[Optional[float]] = [row.get("low") for row in data]
        vols: List[Optional[float]] = [row.get("volume") for row in data]

        last_close = None
        prev_close = None
        for i in range(len(closes) - 1, -1, -1):
            if closes[i] is not None and last_close is None:
                last_close = closes[i]
                continue
            if closes[i] is not None and prev_close is None:
                prev_close = closes[i]
                break

        change = None
        change_pct = None
        if last_close is not None and prev_close is not None:
            change = last_close - prev_close
            change_pct = (change / prev_close * 100.0) if prev_close != 0 else None

        # 52-week high/low approximated from 1y highs/lows
        high_52w = max([h for h in highs if h is not None], default=None)
        low_52w = min([l for l in lows if l is not None], default=None)

        sma_20 = self._sma_last(closes, 20)
        sma_50 = self._sma_last(closes, 50)
        sma_200 = self._sma_last(closes, 200)
        rsi_14 = self._rsi_last(closes, 14)

        # Average volume over last 20 trading days
        last_20_vols = [v for v in vols if v is not None][-20:]
        volume_avg_20d = (sum(last_20_vols) / len(last_20_vols)) if last_20_vols else None

        result = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "price": last_close,
            "change": change,
            "change_pct": change_pct,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "volume_avg_20d": volume_avg_20d,
        }
        # Cache for 2 minutes
        self.cache.set(cache_key, result, ttl=120)
        return result
    def backfill_historical(self, db: Session, ticker: str, period: str = "1y", interval: str = "1d") -> List[Dict]:
        """
        Force fetch from provider and persist, bypassing DB short-circuit.
        Does not read from cache; writes directly to DB using bulk upsert.
        """
        try:
            self.logger.log_info(f"Backfill (force) yfinance data for {ticker} {period} {interval}")
            records = download_historical(ticker, period=period, interval=interval)
            to_store = []
            for row in records:
                ds = row.get("date")
                try:
                    parsed_date = date.fromisoformat(ds[:10]) if ds else None
                except Exception:
                    self.logger.log_warn(f"Skipping row with invalid date: {ds}")
                    continue
                if not parsed_date:
                    self.logger.log_warn("Skipping row due to missing date")
                    continue
                to_store.append(
                    {
                        "ticker": ticker,
                        "date": parsed_date,
                        "interval": interval,
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "volume": row.get("volume"),
                    }
                )
            if to_store:
                bulk_upsert_prices(db, to_store)
            return records
        except Exception as exc:
            self.logger.log_error("yfinance backfill fetch failed", exc)
            raise
