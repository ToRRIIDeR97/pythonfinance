import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time
from collections import OrderedDict

# --- Global Settings ---
INITIAL_CAPITAL = 100000.0
TRANSACTION_FEE_PERCENT = 0.001 # 0.1% per trade (applied on buy and sell)
RISK_FREE_RATE = 0.02 # Annualized, for Sharpe Ratio

# --- 1. Data Handling & Indicator Calculation ---
def download_data(ticker, start_date, end_date):
    """Downloads historical stock data and ensures core columns are Series."""
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    downloaded_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if downloaded_df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker or date range.")
    print("Data download complete.")

    # Uncomment for debugging yfinance output structure:
    # print(f"Initial downloaded_df columns: {downloaded_df.columns}")
    # print(f"Initial dtypes:\n{downloaded_df.dtypes}")
    # for col_inspect in downloaded_df.columns:
    #     print(f"  Type of downloaded_df['{col_inspect}']: {type(downloaded_df[col_inspect])}, Shape: {getattr(downloaded_df[col_inspect], 'shape', 'N/A')}")

    processed_data = OrderedDict() 
    core_columns_set = {'Open', 'High', 'Low', 'Close', 'Volume'}
    
    for original_col_name in downloaded_df.columns:
        current_col_data = downloaded_df[original_col_name]
        
        standardized_name = original_col_name[0] if isinstance(original_col_name, tuple) else original_col_name

        series_to_store = None
        if isinstance(current_col_data, pd.DataFrame):
            if current_col_data.shape[1] == 1:
                series_to_store = current_col_data.iloc[:, 0]
                # print(f"Converted DataFrame column '{original_col_name}' (as '{standardized_name}') to Series.")
            else:
                print(f"Warning: Column '{original_col_name}' (as '{standardized_name}') is a multi-column DataFrame (shape {current_col_data.shape}). Taking first sub-column.")
                series_to_store = current_col_data.iloc[:, 0]
        elif isinstance(current_col_data, pd.Series):
            series_to_store = current_col_data
            # print(f"Column '{original_col_name}' (as '{standardized_name}') is already a Series.")
        else:
            print(f"Warning: Column '{original_col_name}' (as '{standardized_name}') is of unexpected type {type(current_col_data)}. Skipping.")
            continue

        # Ensure the extracted/original series has the standardized name
        if series_to_store.name != standardized_name:
            series_to_store = series_to_store.rename(standardized_name)
        
        processed_data[standardized_name] = series_to_store

    final_df = pd.DataFrame(processed_data, index=downloaded_df.index)

    for core_col in core_columns_set:
        if core_col not in final_df.columns:
            print(f"Warning: Core column '{core_col}' was not found/processed. Adding as NaN Series.")
            final_df[core_col] = pd.Series(np.nan, index=final_df.index)
        elif not isinstance(final_df[core_col], pd.Series):
            print(f"FATAL Error post-processing: Column '{core_col}' is {type(final_df[core_col])}, not Series. Correcting to NaN Series.")
            final_df[core_col] = pd.Series(np.nan, index=final_df.index)
            
    # print(f"Cleaned df dtypes before return:\n{final_df.dtypes}")
    return final_df

def calculate_indicators(df, params=None):
    """
    Calculates all specified technical indicators.
    """
    if params is None: 
        params = {
            'SMA_short_period': 20, 'SMA_long_period': 50,
            'EMA_short_period': 12, 'EMA_long_period': 26,
            'RSI_period': 14, 'RSI_oversold': 30, 'RSI_overbought': 70,
            'ROC_period': 10,
            'MACD_fast': 12, 'MACD_slow': 26, 'MACD_signal': 9,
            'ATR_period': 14,
            'BB_period': 20, 'BB_std_dev': 2,
            'OBV_on': True, 
            'Volatility_period': 20,
            'Trend_Strength_short_sma': 20, 'Trend_Strength_long_sma': 40, 'Trend_Strength_atr': 14,
            'Vol_of_Vol_period': 20,
            'Return_Skewness_period': 60
        }

    print("Calculating indicators...")
    data = df.copy()
    epsilon = 1e-10 

    core_columns_for_calc = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_columns_for_calc:
        if col not in data.columns or not isinstance(data[col], pd.Series) or data[col].isnull().all():
            print(f"Warning: Core column '{col}' is missing, not a Series, or all NaN at start of calculate_indicators. Filling with NaNs.")
            data[col] = pd.Series(np.nan, index=data.index)


    # SMAs
    data['SMA_short'] = data['Close'].rolling(window=params.get('SMA_short_period', 20)).mean()
    data['SMA_long'] = data['Close'].rolling(window=params.get('SMA_long_period', 50)).mean()

    # EMAs
    data['EMA_short'] = data['Close'].ewm(span=params.get('EMA_short_period', 12), adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=params.get('EMA_long_period', 26), adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/params.get('RSI_period', 14), min_periods=params.get('RSI_period', 14), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/params.get('RSI_period', 14), min_periods=params.get('RSI_period', 14), adjust=False).mean()
    rs = avg_gain / (avg_loss + epsilon)
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(50) 
    
    # ROC
    data['ROC'] = data['Close'].pct_change(params.get('ROC_period', 10), fill_method=None) * 100
    
    # MACD
    data['MACD_line'] = data['Close'].ewm(span=params.get('MACD_fast', 12), adjust=False).mean() - \
                        data['Close'].ewm(span=params.get('MACD_slow', 26), adjust=False).mean()
    data['MACD_signal_line'] = data['MACD_line'].ewm(span=params.get('MACD_signal', 9), adjust=False).mean()
    data['MACD_hist'] = data['MACD_line'] - data['MACD_signal_line']

    # ATR
    high_low = data['High'] - data['Low']
    high_close_prev = abs(data['High'] - data['Close'].shift(1))
    low_close_prev = abs(data['Low'] - data['Close'].shift(1))
    tr_df = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    data['TR'] = tr_df.max(axis=1)
    data['ATR'] = data['TR'].ewm(alpha=1/params.get('ATR_period', 14), adjust=False, min_periods=params.get('ATR_period', 14)).mean()

    # Bollinger Bands
    data['BB_SMA'] = data['Close'].rolling(window=params.get('BB_period', 20)).mean()
    data['BB_StdDev'] = data['Close'].rolling(window=params.get('BB_period', 20)).std()
    data['BB_Upper'] = data['BB_SMA'] + (data['BB_StdDev'] * params.get('BB_std_dev', 2))
    data['BB_Lower'] = data['BB_SMA'] - (data['BB_StdDev'] * params.get('BB_std_dev', 2))
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / (data['BB_SMA'] + epsilon)
    
    data['STD20'] = data['Close'].rolling(window=params.get('Volatility_period', 20)).std()

    if params.get('OBV_on', True):
        data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 
                              np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()
    
    data['Daily_Ret'] = data['Close'].pct_change(fill_method=None)
    data['Volatility'] = data['Daily_Ret'].rolling(window=params.get('Volatility_period', 20)).std()

    # Trend Strength
    ts_sma_short_raw = data['SMA_short'] 
    ts_sma_long_raw = data['SMA_long']   
    ts_atr_raw = data['ATR'] 

    def ensure_series(input_val, name_debug, base_index):
        if isinstance(input_val, pd.DataFrame):
            if input_val.shape[1] == 1: return input_val.squeeze()
            elif input_val.shape[1] > 1: return input_val.iloc[:, 0]
            else: return pd.Series(np.nan, index=base_index)
        return input_val if isinstance(input_val, pd.Series) else pd.Series(np.nan, index=base_index)

    ts_sma_short = ensure_series(ts_sma_short_raw, "ts_sma_short", data.index)
    ts_sma_long = ensure_series(ts_sma_long_raw, "ts_sma_long", data.index)
    ts_atr = ensure_series(ts_atr_raw, "ts_atr", data.index)

    if ts_sma_short.isnull().all() or ts_sma_long.isnull().all() or ts_atr.isnull().all():
        data["Trend_Strength"] = np.nan
    else:
        num_aligned, den_aligned = (ts_sma_short - ts_sma_long).abs().align(ts_atr + epsilon, join='left')
        calculated_trend_strength = num_aligned / den_aligned
        
        if isinstance(calculated_trend_strength, pd.DataFrame):
             if calculated_trend_strength.shape[1] > 0: data['Trend_Strength'] = calculated_trend_strength.iloc[:,0]
             else: data['Trend_Strength'] = np.nan
        else: data['Trend_Strength'] = calculated_trend_strength

    data['Vol_of_Vol'] = data['Volatility'].rolling(window=params.get('Vol_of_Vol_period', 20)).std()
    data['Return_Skewness'] = data['Daily_Ret'].rolling(window=params.get('Return_Skewness_period', 60)).skew()
    
    print("\nNaN counts per column BEFORE final dropna():")
    print(data.isnull().sum().sort_values(ascending=False))
    print(f"DataFrame shape before final dropna(): {data.shape}")

    data_cleaned = data.dropna() 
    print(f"DataFrame shape AFTER final dropna(): {data_cleaned.shape}")
    print("Indicator calculation complete.")
    return data_cleaned


# --- 2. Strategy Definitions ---
class Strategy:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params if params is not None else {}
        self.name = "BaseStrategy"

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0 
        return signals

class RsiStrategy(Strategy):
    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.name = "RSI Strategy"
        self.oversold = self.params.get('RSI_oversold', 30)
        self.overbought = self.params.get('RSI_overbought', 70)

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        if 'RSI' not in self.data.columns or self.data['RSI'].isnull().all(): return signals 
        signals.loc[(self.data['RSI'].shift(1) < self.oversold) & (self.data['RSI'] > self.oversold), 'signal'] = 1
        signals.loc[(self.data['RSI'].shift(1) > self.overbought) & (self.data['RSI'] < self.overbought), 'signal'] = -1
        return signals

class MacdStrategy(Strategy):
    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.name = "MACD Crossover Strategy"

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        if not all(col in self.data.columns and not self.data[col].isnull().all() for col in ['MACD_line', 'MACD_signal_line']): return signals
        signals.loc[(self.data['MACD_line'].shift(1) < self.data['MACD_signal_line'].shift(1)) & \
                    (self.data['MACD_line'] > self.data['MACD_signal_line']), 'signal'] = 1
        signals.loc[(self.data['MACD_line'].shift(1) > self.data['MACD_signal_line'].shift(1)) & \
                    (self.data['MACD_line'] < self.data['MACD_signal_line']), 'signal'] = -1
        return signals

class MaCrossStrategy(Strategy):
    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.name = "Moving Average Crossover Strategy"

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        if not all(col in self.data.columns and not self.data[col].isnull().all() for col in ['SMA_short', 'SMA_long']): return signals
        signals.loc[(self.data['SMA_short'].shift(1) < self.data['SMA_long'].shift(1)) & \
                    (self.data['SMA_short'] > self.data['SMA_long']), 'signal'] = 1
        signals.loc[(self.data['SMA_short'].shift(1) > self.data['SMA_long'].shift(1)) & \
                    (self.data['SMA_short'] < self.data['SMA_long']), 'signal'] = -1
        return signals

class ConvergingSignalsStrategy(Strategy):
    def __init__(self, data, strategies, params=None):
        super().__init__(data, params)
        self.strategies = strategies 
        self.min_signals_to_converge = self.params.get('min_signals_to_converge', len(strategies))
        self.name = f"Converging Signals ({self.min_signals_to_converge}/{len(strategies)})"

    def generate_signals(self):
        print(f"\n--- Debugging {self.name} --- Min signals to converge: {self.min_signals_to_converge}")
        if not self.strategies: 
            print("No strategies provided to ConvergingSignalsStrategy.")
            return pd.DataFrame(0, index=self.data.index, columns=['signal'])
        
        all_signals_df = pd.DataFrame(index=self.data.index)
        print("Individual strategy signals:")
        for i, strategy in enumerate(self.strategies):
            strategy_signals = strategy.generate_signals()
            signal_col_name = f'signal_{strategy.name.replace(" ", "_")}'
            if not strategy_signals.empty:
                all_signals_df[signal_col_name] = strategy_signals['signal']
                # Print summary of individual strategy signals
                print(f"  {strategy.name}: Buy signals: {(strategy_signals['signal'] == 1).sum()}, Sell signals: {(strategy_signals['signal'] == -1).sum()}")
            else: 
                all_signals_df[signal_col_name] = 0
                print(f"  {strategy.name}: No signals generated (empty DataFrame).")

        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        
        if all_signals_df.empty: 
            print("all_signals_df is empty after collecting individual signals.")
            return signals

        print("\nCombined signals from all strategies (all_signals_df head):")
        print(all_signals_df[(all_signals_df != 0).any(axis=1)].head())

        buy_convergence = (all_signals_df == 1).sum(axis=1)
        sell_convergence = (all_signals_df == -1).sum(axis=1)
        
        print(f"\nTotal days with potential buy signals (any strategy): {(buy_convergence > 0).sum()}")
        print(f"Total days with potential sell signals (any strategy): {(sell_convergence > 0).sum()}")

        signals.loc[buy_convergence >= self.min_signals_to_converge, 'signal'] = 1
        signals.loc[sell_convergence >= self.min_signals_to_converge, 'signal'] = -1
        
        # Log how many convergence signals were generated before conflict resolution
        initial_converged_buys = (signals['signal'] == 1).sum()
        initial_converged_sells = (signals['signal'] == -1).sum()
        print(f"Converged buy signals (before conflict resolution): {initial_converged_buys}")
        print(f"Converged sell signals (before conflict resolution): {initial_converged_sells}")

        conflicting_days = (buy_convergence >= self.min_signals_to_converge) & \
                           (sell_convergence >= self.min_signals_to_converge)
        signals.loc[conflicting_days, 'signal'] = 0
        
        print(f"Days with conflicting signals (resolved to 0): {conflicting_days.sum()}")
        print(f"Final converged buy signals: {(signals['signal'] == 1).sum()}")
        print(f"Final converged sell signals: {(signals['signal'] == -1).sum()}")
        print("--- End Debugging ConvergingSignalsStrategy ---\n")
        return signals

# --- 3. Backtesting Engine ---
def backtest_strategy(data_with_indicators, signals_df, initial_capital, transaction_fee_percent):
    print(f"Starting backtest...")
    if signals_df.empty:
        print("Warning: Signals DataFrame is empty. Returning empty portfolio.")
        return pd.DataFrame(columns=['holdings', 'cash', 'total', 'position', 'returns']), pd.DataFrame()
        
    common_index = data_with_indicators.index.intersection(signals_df.index)
    if not common_index.is_unique:
        print("Warning: Common index for backtest is not unique. Deduplicating (keeping first).")
        common_index = common_index[~common_index.duplicated(keep='first')]

    if common_index.empty:
        print("Error: No common dates between data and signals. Backtest cannot proceed.")
        return pd.DataFrame(columns=['holdings', 'cash', 'total', 'position', 'returns']), pd.DataFrame()

    data = data_with_indicators.loc[common_index]
    signals = signals_df.loc[common_index]['signal']
    
    portfolio = pd.DataFrame(index=common_index)
    portfolio['holdings'] = 0.0  
    portfolio['cash'] = initial_capital
    portfolio['total'] = initial_capital
    portfolio['position'] = 0 

    trade_log = []

    for i in range(len(common_index)): 
        current_date = common_index[i]
        
        current_price_val = data.loc[current_date, 'Close']
        if isinstance(current_price_val, pd.Series): 
            current_price_val = current_price_val.item() if len(current_price_val) == 1 else (current_price_val.iloc[0] if not current_price_val.empty else np.nan)

        if pd.isna(current_price_val) or current_price_val <= 0: 
            if i > 0: 
                prev_date = common_index[i-1]
                portfolio.loc[current_date, ['cash', 'holdings', 'position', 'total']] = portfolio.loc[prev_date, ['cash', 'holdings', 'position', 'total']]
            else: 
                portfolio.loc[current_date, ['cash', 'holdings', 'position', 'total']] = [initial_capital, 0.0, 0, initial_capital]
            if i == len(common_index) -1 and 'total' in portfolio.columns: 
                 portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
            continue

        signal = signals.loc[current_date]

        if i == 0:
            portfolio.loc[current_date, 'cash'] = initial_capital
            portfolio.loc[current_date, 'holdings'] = 0.0
            portfolio.loc[current_date, 'position'] = 0
        else:
            prev_date = common_index[i-1]
            portfolio.loc[current_date, 'cash'] = portfolio.loc[prev_date, 'cash']
            portfolio.loc[current_date, 'holdings'] = portfolio.loc[prev_date, 'holdings'] 
            portfolio.loc[current_date, 'position'] = portfolio.loc[prev_date, 'position']
            
            if portfolio.loc[prev_date, 'position'] == 1:
                 prev_price_val = data.loc[prev_date, 'Close']
                 if isinstance(prev_price_val, pd.Series): prev_price_val = prev_price_val.item() if len(prev_price_val)==1 else (prev_price_val.iloc[0] if not prev_price_val.empty else 0)
                 if prev_price_val != 0: 
                    portfolio.loc[current_date, 'holdings'] *= (current_price_val / prev_price_val)

        current_position_at_date = portfolio.loc[current_date, 'position'] 
        cash_on_hand = portfolio.loc[current_date, 'cash']
        if isinstance(cash_on_hand, pd.Series): cash_on_hand = cash_on_hand.item() if len(cash_on_hand)==1 else cash_on_hand.iloc[0]
        
        if signal == 1 and current_position_at_date == 0:  # Buy signal and not already in position
            if cash_on_hand > 0:
                # Calculate how much stock value can be acquired with available cash, accounting for fees
                amount_for_shares = cash_on_hand / (1 + transaction_fee_percent)
                
                # Check if the amount is too small to be meaningful (e.g., buying fractions of a cent of stock)
                if amount_for_shares < 1e-6: # Avoids issues with near-zero investments
                    pass # Not enough cash for a meaningful purchase, or log this event
                else:
                    fee = amount_for_shares * transaction_fee_percent
                    
                    portfolio.loc[current_date, 'cash'] -= (amount_for_shares + fee) # Deduct total spent (value of shares + fee)
                    portfolio.loc[current_date, 'holdings'] = amount_for_shares     # Store the value of stock acquired
                    portfolio.loc[current_date, 'position'] = 1                     # Update position to indicate holding stock
                    trade_log.append({
                        'Date': current_date, 
                        'Type': 'Buy', 
                        'Price': current_price_val, 
                        'Amount (Value)': amount_for_shares, # Log the value of shares bought
                        'Fee': fee
                    })

        elif signal == -1 and current_position_at_date == 1: 
            value_of_holdings_to_sell = portfolio.loc[current_date, 'holdings'] 
            fee = value_of_holdings_to_sell * transaction_fee_percent
            
            portfolio.loc[current_date, 'cash'] += (value_of_holdings_to_sell - fee)
            portfolio.loc[current_date, 'holdings'] = 0.0
            portfolio.loc[current_date, 'position'] = 0
            trade_log.append({'Date': current_date, 'Type': 'Sell', 'Price': current_price_val, 'Amount (Value)': value_of_holdings_to_sell, 'Fee': fee})

        portfolio.loc[current_date, 'total'] = portfolio.loc[current_date, 'cash'] + portfolio.loc[current_date, 'holdings']
        
    if not portfolio.empty:
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
    else: 
        portfolio['returns'] = pd.Series(dtype=float)

    print("Backtest complete.")
    return portfolio, pd.DataFrame(trade_log)

# --- 4. Performance Metrics & Plotting ---
def calculate_performance_metrics(portfolio_df, risk_free_rate):
    epsilon = 1e-10 # Prevent division by zero
    if portfolio_df.empty or 'total' not in portfolio_df.columns or portfolio_df['total'].isnull().all():
        print("Portfolio DataFrame is too short to calculate metrics.")
        return {"Total Return": 0, "Sharpe Ratio": 0, "Max Drawdown": 0, "Daily Win Rate": 0, "Final Portfolio Value": INITIAL_CAPITAL}

    total_return = (portfolio_df['total'].iloc[-1] / portfolio_df['total'].iloc[0]) - 1 if portfolio_df['total'].iloc[0] != 0 else 0
    daily_returns = portfolio_df['returns']
    if daily_returns.std() == 0 or np.isnan(daily_returns.std()): 
        sharpe_ratio = 0
    else:
        excess_returns = daily_returns - (risk_free_rate / 252) 
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / (excess_returns.std() + epsilon))
    
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    
    profitable_days = (daily_returns > 0).sum()
    total_days_with_change = (daily_returns != 0).sum()
    win_rate_daily = profitable_days / total_days_with_change if total_days_with_change > 0 else 0
    
    metrics = {
        "Total Return": total_return, "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown,
        "Daily Win Rate": win_rate_daily, "Final Portfolio Value": portfolio_df['total'].iloc[-1]
    }
    print("\nPerformance Metrics:")
    for k, v in metrics.items(): print(f"  {k}: {v:.4f}")
    return metrics

def plot_results(portfolio_df, data_with_indicators, strategy_name, signals_df=None, trade_log=None):
    if portfolio_df.empty:
        print(f"Skipping plot for {strategy_name}: Portfolio data is empty.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    common_plot_index = data_with_indicators.index.intersection(portfolio_df.index)
    if common_plot_index.empty:
        print(f"Skipping plot for {strategy_name}: No common dates for plotting.")
        plt.close(fig) 
        return
        
    data_plot = data_with_indicators.loc[common_plot_index]
    portfolio_plot = portfolio_df.loc[common_plot_index]

    axs[0].plot(portfolio_plot.index, portfolio_plot['total'], label='Strategy Equity', color='blue')
    
    if not data_plot.empty and 'Close' in data_plot.columns:
        close_series_for_plot = data_plot['Close']
        # Ensure close_series_for_plot is a Series
        if isinstance(close_series_for_plot, pd.DataFrame): 
            if close_series_for_plot.shape[1] > 0: close_series_for_plot = close_series_for_plot.iloc[:, 0].copy() # Use .copy()
            else: close_series_for_plot = pd.Series(dtype=float, index=data_plot.index)
        
        if not close_series_for_plot.empty and not close_series_for_plot.isnull().all():
            # Ensure first_close_val is a scalar
            first_close_val_series = close_series_for_plot.dropna()
            first_close_val = first_close_val_series.iloc[0] if not first_close_val_series.empty else np.nan
            if isinstance(first_close_val, pd.Series): # Should not happen if above is correct
                 first_close_val = first_close_val.item() if len(first_close_val) == 1 else np.nan

            if not pd.isna(first_close_val) and first_close_val != 0:
                buy_hold_equity = (INITIAL_CAPITAL / first_close_val) * close_series_for_plot
                axs[0].plot(close_series_for_plot.index, buy_hold_equity, label='Buy & Hold Equity', color='grey', linestyle='--')
    axs[0].set_title(f'{strategy_name} - Equity Curve'); axs[0].set_ylabel('Portfolio Value'); axs[0].legend(); axs[0].grid(True)

    if 'Close' in data_plot.columns:
        price_series_for_plot = data_plot['Close']
        if isinstance(price_series_for_plot, pd.DataFrame):
            if price_series_for_plot.shape[1] > 0: price_series_for_plot = price_series_for_plot.iloc[:,0].copy() # Use .copy()
            else: price_series_for_plot = pd.Series(dtype=float, index=data_plot.index)
        
        if not price_series_for_plot.empty and not price_series_for_plot.isnull().all():
            axs[1].plot(price_series_for_plot.index, price_series_for_plot, label='Close Price', color='black', alpha=0.7)
            if trade_log is not None and not trade_log.empty:
                trade_log_plot = trade_log[trade_log['Date'].isin(common_plot_index)]
                buy_signals = trade_log_plot[trade_log_plot['Type'] == 'BUY']
                sell_signals = trade_log_plot[trade_log_plot['Type'] == 'SELL']
                if not buy_signals.empty and not buy_signals['Date'].isnull().all():
                    valid_buy_dates = buy_signals['Date'].dropna()
                    valid_buy_dates_in_plot = valid_buy_dates[valid_buy_dates.isin(price_series_for_plot.index)]
                    if not valid_buy_dates_in_plot.empty:
                        axs[1].plot(valid_buy_dates_in_plot, price_series_for_plot.loc[valid_buy_dates_in_plot], '^', markersize=10, color='green', lw=0, label='Buy Signal')
                if not sell_signals.empty and not sell_signals['Date'].isnull().all():
                    valid_sell_dates = sell_signals['Date'].dropna()
                    valid_sell_dates_in_plot = valid_sell_dates[valid_sell_dates.isin(price_series_for_plot.index)]
                    if not valid_sell_dates_in_plot.empty:
                        axs[1].plot(valid_sell_dates_in_plot, price_series_for_plot.loc[valid_sell_dates_in_plot], 'v', markersize=10, color='red', lw=0, label='Sell Signal')
    axs[1].set_title('Price and Trade Signals'); axs[1].set_ylabel('Price'); axs[1].legend(); axs[1].grid(True)

    indicator_plotted = False
    if 'RSI' in data_plot.columns and "RSI" in strategy_name:
        axs[2].plot(data_plot.index, data_plot['RSI'], label='RSI', color='purple')
        axs[2].axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        axs[2].axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        axs[2].set_title('RSI Indicator'); axs[2].set_ylabel('RSI Value'); axs[2].legend(); indicator_plotted = True
    elif 'MACD_line' in data_plot.columns and "MACD" in strategy_name:
        axs[2].plot(data_plot.index, data_plot['MACD_line'], label='MACD Line', color='blue')
        axs[2].plot(data_plot.index, data_plot['MACD_signal_line'], label='Signal Line', color='orange')
        axs[2].bar(data_plot.index, data_plot['MACD_hist'], label='Histogram', color='grey', alpha=0.5)
        axs[2].set_title('MACD Indicator'); axs[2].legend(); indicator_plotted = True
    elif 'SMA_short' in data_plot.columns and "Moving Average" in strategy_name:
        axs[2].plot(data_plot.index, data_plot['SMA_short'], label='SMA Short', color='blue')
        axs[2].plot(data_plot.index, data_plot['SMA_long'], label='SMA Long', color='orange')
        axs[2].set_title('Moving Averages'); axs[2].legend(); indicator_plotted = True
    
    if not indicator_plotted:
        axs[2].text(0.5, 0.5, 'Indicator plot not configured for this strategy.', ha='center', va='center')
    axs[2].grid(True)
    plt.xlabel('Date'); 
    try:
        fig.tight_layout(pad=1.0) # Added pad argument
    except Exception as e:
        print(f"Note: tight_layout failed: {e}")
    plt.show()

# --- 5. Parameter Optimization ---
def objective_function(params_to_optimize_values, params_to_optimize_names, raw_data_for_opt, strategy_class, base_indicator_params_for_opt, base_strategy_logic_params, metric_to_optimize="Sharpe Ratio"):
    obj_func_start_time = time.time()
    
    current_trial_indicator_params = base_indicator_params_for_opt.copy()
    current_trial_strategy_logic_params = base_strategy_logic_params.copy()

    param_dict_for_print = {}
    for name, value in zip(params_to_optimize_names, params_to_optimize_values):
        is_indicator_param = name.endswith("_period") or name.endswith("_window") or \
                             name.endswith("_fast") or name.endswith("_slow") or \
                             name.endswith("_signal") or name == 'BB_std_dev'
        
        if is_indicator_param:
            rounded_value = int(round(value)) if name != 'BB_std_dev' else value
            current_trial_indicator_params[name] = rounded_value
            param_dict_for_print[name] = rounded_value
        else:
            current_trial_strategy_logic_params[name] = value
            param_dict_for_print[name] = value
            
    # print(f"Objective func call with: {param_dict_for_print}") # Can be too verbose

    try:
        calc_indicators_start_time = time.time()
        data_for_trial = calculate_indicators(raw_data_for_opt.copy(), current_trial_indicator_params)
        calc_indicators_time = time.time() - calc_indicators_start_time
        
        if data_for_trial.empty:
            # print(f"Objective func: data_for_trial empty. Params: {param_dict_for_print}. Calc Ind Time: {calc_indicators_time:.4f}s. Total Time: {time.time() - obj_func_start_time:.4f}s. Returning inf.")
            return np.inf

        full_strategy_params = {**current_trial_indicator_params, **current_trial_strategy_logic_params}
        
        strategy_instance = strategy_class(data_for_trial, full_strategy_params)
        signals = strategy_instance.generate_signals()
        
        common_idx = data_for_trial.index.intersection(signals.index)
        if common_idx.empty:
            # print(f"Objective func: common_idx empty. Params: {param_dict_for_print}. Calc Ind Time: {calc_indicators_time:.4f}s. Total Time: {time.time() - obj_func_start_time:.4f}s. Returning inf.")
            return np.inf

        backtest_start_time = time.time()
        portfolio_df, _ = backtest_strategy(data_for_trial.loc[common_idx], signals.loc[common_idx], INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
        backtest_time = time.time() - backtest_start_time

        if portfolio_df.empty:
            # print(f"Objective func: portfolio_df empty. Params: {param_dict_for_print}. Calc Ind Time: {calc_indicators_time:.4f}s. Backtest Time: {backtest_time:.4f}s. Total Time: {time.time() - obj_func_start_time:.4f}s. Returning inf.")
            return np.inf

        metrics = calculate_performance_metrics(portfolio_df, RISK_FREE_RATE)
        value_to_optimize = metrics.get(metric_to_optimize, -np.inf if metric_to_optimize != "Max Drawdown" else np.inf)
        
        result_val = abs(value_to_optimize) if metric_to_optimize == "Max Drawdown" else -value_to_optimize
        # print(f"Objective func: Success. Params: {param_dict_for_print}. Metric '{metric_to_optimize}': {metrics.get(metric_to_optimize, 'N/A')}. Result: {result_val:.4f}. Calc Ind Time: {calc_indicators_time:.4f}s. Backtest Time: {backtest_time:.4f}s. Total Time: {time.time() - obj_func_start_time:.4f}s")
        return result_val
            
    except Exception as e:
        # print(f"Objective func: EXCEPTION '{e}' with params {param_dict_for_print}. Total Time: {time.time() - obj_func_start_time:.4f}s. Returning inf.")
        return np.inf 

def optimize_strategy_params(raw_data_df, strategy_class, param_config, 
                             initial_indicator_params, 
                             initial_strategy_logic_params, 
                             metric_to_optimize="Sharpe Ratio"):
    print(f"\nOptimizing {strategy_class.__name__} for {metric_to_optimize}...")
    param_names = [p[0] for p in param_config]
    bounds = [(p[1], p[2]) for p in param_config]
    
    result = differential_evolution(
        objective_function, bounds,
        args=(param_names, raw_data_df, strategy_class, 
              initial_indicator_params, initial_strategy_logic_params, metric_to_optimize),
        strategy='best1bin', maxiter=30, popsize=10, tol=0.01, disp=True, polish=False, workers=-1
    )
    
    optimized_param_values = result.x
    combined_optimized_params = initial_indicator_params.copy() 
    combined_optimized_params.update(initial_strategy_logic_params) 

    print("\nOptimization Complete."); print(f"Best Objective Value: {result.fun:.4f}")
    print("Optimized Parameters (merged):")
    for name, value in zip(param_names, optimized_param_values):
        is_indicator_param = name.endswith("_period") or name.endswith("_window") or \
                             name.endswith("_fast") or name.endswith("_slow") or \
                             name.endswith("_signal") or name == 'BB_std_dev'
        
        if is_indicator_param:
            if name == 'BB_std_dev':
                 combined_optimized_params[name] = value
                 print(f"  Indicator Param - {name}: {value:.2f}")
            else:
                 combined_optimized_params[name] = int(round(value))
                 print(f"  Indicator Param - {name}: {int(round(value))}")
        else: 
            combined_optimized_params[name] = value
            print(f"  Strategy Logic Param - {name}: {value:.2f}")
            
    return combined_optimized_params


# --- 6. Main Execution Block ---
if __name__ == "__main__":
    TICKER = 'MSFT'
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'

    raw_df_main = download_data(TICKER, START_DATE, END_DATE) 
    
    base_indicator_params = {
        'SMA_short_period': 20, 'SMA_long_period': 50, 'EMA_short_period': 12, 'EMA_long_period': 26,
        'RSI_period': 14, 'ROC_period': 10, 'MACD_fast': 12, 'MACD_slow': 26, 'MACD_signal': 9,
        'ATR_period': 14, 'BB_period': 20, 'BB_std_dev': 2.0, 'OBV_on': True, 'Volatility_period': 20,
        'Trend_Strength_short_sma': 20, 'Trend_Strength_long_sma': 40, 
        'Vol_of_Vol_period': 20, 'Return_Skewness_period': 60
    }
    # Note: Trend_Strength_atr will use ATR_period from the main params.
    
    data_with_indicators_main = calculate_indicators(raw_df_main.copy(), base_indicator_params) 

    if data_with_indicators_main.empty:
        print("No data after initial indicator calculation for main execution. Exiting.")
        exit()

    print("\n--- Running Single RSI Strategy (Default Params) ---")
    rsi_default_logic_params = {'RSI_oversold': 30, 'RSI_overbought': 70}
    rsi_full_default_params = {**base_indicator_params, **rsi_default_logic_params}
    rsi_strat = RsiStrategy(data_with_indicators_main, rsi_full_default_params)
    rsi_signals = rsi_strat.generate_signals()
    rsi_portfolio, rsi_tradelog = backtest_strategy(data_with_indicators_main, rsi_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
    if not rsi_portfolio.empty:
        calculate_performance_metrics(rsi_portfolio, RISK_FREE_RATE)
        plot_results(rsi_portfolio, data_with_indicators_main, rsi_strat.name, rsi_signals, rsi_tradelog)

    print("\n--- Optimizing RSI Strategy (Thresholds and RSI_period) ---")
    rsi_param_config_to_optimize = [
        ('RSI_period', 7, 25),      
        ('RSI_oversold', 15, 45),   
        ('RSI_overbought', 55, 85)  
    ]
    rsi_initial_logic_params_for_opt = {} 
    
    optimized_rsi_full_params = optimize_strategy_params(
        raw_data_df=raw_df_main.copy(), 
        strategy_class=RsiStrategy, 
        param_config=rsi_param_config_to_optimize,
        initial_indicator_params=base_indicator_params.copy(), 
        initial_strategy_logic_params=rsi_initial_logic_params_for_opt,
        metric_to_optimize="Sharpe Ratio"
    )
    
    print("\n--- Running RSI Strategy with Fully Optimized Parameters ---")
    data_for_optimized_rsi_run = calculate_indicators(raw_df_main.copy(), optimized_rsi_full_params) 
    if not data_for_optimized_rsi_run.empty:
        optimized_rsi_strat = RsiStrategy(data_for_optimized_rsi_run, optimized_rsi_full_params)
        optimized_rsi_signals = optimized_rsi_strat.generate_signals()
        optimized_rsi_portfolio, optimized_rsi_tradelog = backtest_strategy(data_for_optimized_rsi_run, optimized_rsi_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
        if not optimized_rsi_portfolio.empty:
            calculate_performance_metrics(optimized_rsi_portfolio, RISK_FREE_RATE)
            plot_results(optimized_rsi_portfolio, data_for_optimized_rsi_run, "Fully Optimized " + optimized_rsi_strat.name, optimized_rsi_signals, optimized_rsi_tradelog)
    else:
        print("Could not run optimized RSI strategy as data was empty after indicator recalculation.")


    # --- Optimizing MACD Strategy --- 
    print("\n--- Optimizing MACD Strategy (Fast, Slow, Signal periods) ---")
    macd_param_config_to_optimize = [
        ('MACD_fast', 5, 20),      
        ('MACD_slow', 21, 50),   
        ('MACD_signal', 5, 15)  
    ]
    # Ensure MACD_fast < MACD_slow, objective_function handles errors if constraints violated by returning np.inf
    macd_initial_logic_params_for_opt = {} # MACD params are indicator params
    
    optimized_macd_full_params = optimize_strategy_params(
        raw_data_df=raw_df_main.copy(), 
        strategy_class=MacdStrategy, 
        param_config=macd_param_config_to_optimize,
        initial_indicator_params=base_indicator_params.copy(), 
        initial_strategy_logic_params=macd_initial_logic_params_for_opt,
        metric_to_optimize="Sharpe Ratio"
    )
    
    print("\n--- Running MACD Strategy with Fully Optimized Parameters ---")
    data_for_optimized_macd_run = calculate_indicators(raw_df_main.copy(), optimized_macd_full_params) 
    if not data_for_optimized_macd_run.empty:
        optimized_macd_strat = MacdStrategy(data_for_optimized_macd_run, optimized_macd_full_params)
        optimized_macd_signals = optimized_macd_strat.generate_signals()
        optimized_macd_portfolio, optimized_macd_tradelog = backtest_strategy(data_for_optimized_macd_run, optimized_macd_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
        if not optimized_macd_portfolio.empty:
            calculate_performance_metrics(optimized_macd_portfolio, RISK_FREE_RATE)
            plot_results(optimized_macd_portfolio, data_for_optimized_macd_run, "Fully Optimized " + optimized_macd_strat.name, optimized_macd_signals, optimized_macd_tradelog)
    else:
        print("Could not run optimized MACD strategy as data was empty after indicator recalculation.")

    # --- Optimizing MA Cross Strategy --- 
    print("\n--- Optimizing MA Cross Strategy (Short and Long SMA periods) ---")
    ma_param_config_to_optimize = [
        ('SMA_short_period', 5, 40),      
        ('SMA_long_period', 21, 100) # Start long period higher to ensure short < long
    ]
    # Ensure SMA_short_period < SMA_long_period, objective_function handles errors
    ma_initial_logic_params_for_opt = {} # MA Cross params are indicator params

    optimized_ma_cross_full_params = optimize_strategy_params(
        raw_data_df=raw_df_main.copy(), 
        strategy_class=MaCrossStrategy, 
        param_config=ma_param_config_to_optimize,
        initial_indicator_params=base_indicator_params.copy(),
        initial_strategy_logic_params=ma_initial_logic_params_for_opt,
        metric_to_optimize="Sharpe Ratio"
    )

    print("\n--- Running MA Cross Strategy with Fully Optimized Parameters ---")
    data_for_optimized_ma_run = calculate_indicators(raw_df_main.copy(), optimized_ma_cross_full_params)
    if not data_for_optimized_ma_run.empty:
        optimized_ma_strat = MaCrossStrategy(data_for_optimized_ma_run, optimized_ma_cross_full_params)
        optimized_ma_signals = optimized_ma_strat.generate_signals()
        optimized_ma_portfolio, optimized_ma_tradelog = backtest_strategy(data_for_optimized_ma_run, optimized_ma_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
        if not optimized_ma_portfolio.empty:
            calculate_performance_metrics(optimized_ma_portfolio, RISK_FREE_RATE)
            plot_results(optimized_ma_portfolio, data_for_optimized_ma_run, "Fully Optimized " + optimized_ma_strat.name, optimized_ma_signals, optimized_ma_tradelog)
    else:
        print("Could not run optimized MA Cross strategy as data was empty after indicator recalculation.")

    # --- Running Converging Signals Strategy (ALL components optimized) ---
    print("\n--- Running Converging Signals Strategy (ALL components optimized) ---")
    # Ensure all underlying data for optimization was successfully created
    if not data_for_optimized_rsi_run.empty and not data_for_optimized_macd_run.empty and not data_for_optimized_ma_run.empty:
        # 1. RSI Strategy with its optimized params and data
        # data_for_optimized_rsi_run already calculated earlier
        rsi_for_converge_fully_optimized = RsiStrategy(data_for_optimized_rsi_run, optimized_rsi_full_params)

        # 2. MACD Strategy with its optimized params and data
        # data_for_optimized_macd_run already calculated
        macd_for_converge_fully_optimized = MacdStrategy(data_for_optimized_macd_run, optimized_macd_full_params)

        # 3. MA Cross Strategy with its optimized params and data
        # data_for_optimized_ma_run already calculated
        ma_for_converge_fully_optimized = MaCrossStrategy(data_for_optimized_ma_run, optimized_ma_cross_full_params)

        converging_strat_list_all_optimized = [
            rsi_for_converge_fully_optimized, 
            macd_for_converge_fully_optimized, 
            ma_for_converge_fully_optimized
        ]
        converging_logic_params = {'min_signals_to_converge': 2} 
        
        # The ConvergingSignalsStrategy itself uses the raw_df for index, sub-strategies use their own data
        converging_strategy_all_optimized = ConvergingSignalsStrategy(
            raw_df_main.copy(), # Base data for index alignment
            converging_strat_list_all_optimized, 
            converging_logic_params
        )
        
        print(f"Attempting to run: {converging_strategy_all_optimized.name} with all components optimized.")
        converging_signals_all_optimized = converging_strategy_all_optimized.generate_signals()

        # For backtesting and plotting the converging strategy, create a master data set
        # This master set will use indicator parameters from each optimized strategy
        master_indicator_params_for_convergence = base_indicator_params.copy()
        master_indicator_params_for_convergence.update({
            'RSI_period': optimized_rsi_full_params.get('RSI_period'),
            'MACD_fast': optimized_macd_full_params.get('MACD_fast'),
            'MACD_slow': optimized_macd_full_params.get('MACD_slow'),
            'MACD_signal': optimized_macd_full_params.get('MACD_signal'),
            'SMA_short_period': optimized_ma_cross_full_params.get('SMA_short_period'),
            'SMA_long_period': optimized_ma_cross_full_params.get('SMA_long_period'),
        })
        # Remove None values if any param wasn't found (e.g. if optimization failed for a strategy)
        master_indicator_params_for_convergence = {k: v for k, v in master_indicator_params_for_convergence.items() if v is not None}

        data_for_converging_backtest_and_plot = calculate_indicators(raw_df_main.copy(), master_indicator_params_for_convergence)

        if not data_for_converging_backtest_and_plot.empty and not converging_signals_all_optimized.empty:
            converging_portfolio_all_optimized, converging_tradelog_all_optimized = backtest_strategy(
                data_for_converging_backtest_and_plot, 
                converging_signals_all_optimized, 
                INITIAL_CAPITAL, 
                TRANSACTION_FEE_PERCENT
            )
            
            if not converging_portfolio_all_optimized.empty:
                print(f"Performance for {converging_strategy_all_optimized.name} (all components optimized):")
                calculate_performance_metrics(converging_portfolio_all_optimized, RISK_FREE_RATE)
                plot_results(
                    converging_portfolio_all_optimized, 
                    data_for_converging_backtest_and_plot, 
                    converging_strategy_all_optimized.name + " (All Opt)", 
                    converging_signals_all_optimized, 
                    converging_tradelog_all_optimized
                )
            else:
                print(f"{converging_strategy_all_optimized.name} (all components optimized) did not generate any trades or the portfolio was empty.")
        else:
            print(f"Could not run backtest for {converging_strategy_all_optimized.name} due to empty data or signals.")
    else:
        print("Skipping fully optimized Converging Signals Strategy run because one or more component strategy data was empty after optimization.")

    print("\n--- Full Bot Script Execution Finished ---")

