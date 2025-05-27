import yfinance as yf
import pandas as pd
from utils import logger

def fetch_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetches historical OHLCV data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'BTC-USD' for Bitcoin).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1d' for daily, '1h' for hourly).
                        Note: Intraday data availability varies by ticker and period.

    Returns:
        pandas.DataFrame: DataFrame with OHLCV data, or None if an error occurs.
    """
    try:
        logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date} with interval {interval}.")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            logger.warning(f"No data found for {ticker} for the given period/interval.")
            return None
        logger.info(f"Successfully fetched {len(data)} data points for {ticker}.")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage:
    btc_data = fetch_historical_data('BTC-USD', '2023-01-01', '2023-12-31', '1d')
    if btc_data is not None:
        print("Bitcoin Daily Data:")
        print(btc_data.head())

    eth_data_hourly = fetch_historical_data('ETH-USD', '2023-12-01', '2023-12-07', '1h') # Note: 1h data limited to recent past
    if eth_data_hourly is not None:
        print("\nEthereum Hourly Data:")
        print(eth_data_hourly.head())
