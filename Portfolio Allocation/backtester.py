import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Import optimizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px # Import plotly express for color sequence
from joblib import Parallel, delayed # Import joblib for parallel execution
from pandas.tseries.offsets import DateOffset # For rolling dates

# Note: Global constants like ticker_map, baseline_allocation etc., 
# are now expected to be passed into the relevant functions (primarily run_backtest_for_period)

# -------------------------------
# Data Download Functions
# -------------------------------
# Add this new function
def get_indicator_ohlc_data(ticker, start, end, interval='1d'):
    """Downloads OHLC data for a single ticker at a specific interval."""
    print(f"Downloading {interval} OHLC data for indicator ({ticker})...")
    try:
        data_raw = yf.download(ticker, start=start, end=end, interval=interval)
        if data_raw.empty:
            print(f"Error: yfinance download returned empty OHLC data for {ticker} at {interval}.")
            return None
        # Select only OHLC columns, rename for consistency
        ohlc_data = data_raw[['Open', 'High', 'Low', 'Close']].copy()
        # Drop rows with any missing OHLC values for robustness
        ohlc_data = ohlc_data.dropna()
        if ohlc_data.empty:
             print(f"Error: OHLC data for {ticker} empty after dropping NaNs.")
             return None
        print(f"Indicator OHLC data for {ticker} downloaded.")
        return ohlc_data
    except Exception as e:
        print(f"Error during indicator OHLC data download for {ticker}: {e}")
        return None

# Modify existing download function
def download_prepare_data(ticker_mapping, start, end):
    """Downloads daily close data for tickers, renames columns, and handles missing data."""
    print("Downloading Daily Close data for backtesting...")
    
    # Separate CASH placeholder from actual tickers
    actual_tickers = {symbol: ticker for symbol, ticker in ticker_mapping.items() if symbol != "CASH"}
    tickers_to_download = list(actual_tickers.values())
    
    if not tickers_to_download:
        print("Error: No actual tickers specified for download (excluding CASH).")
        return None
        
    try:
        # Download data only for actual tickers
        data = yf.download(tickers_to_download, start=start, end=end, progress=True)
        if data.empty:
            print("Error: No data downloaded. Check tickers and date range.")
            return None
            
        # Select 'Close' prices and handle potential multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data_close = data['Close']
        else: # Handle case where only one ticker is downloaded
            data_close = data[['Close']] if 'Close' in data else data # Assume data is Close if 'Close' not present
            if len(tickers_to_download) == 1:
                 data_close.columns = [tickers_to_download[0]] # Ensure column name matches ticker
        
        # --- Create an explicit copy to avoid SettingWithCopyWarning ---
        data_close = data_close.copy()
                 
        # Rename columns based on the mapping (for actual tickers)
        reverse_ticker_map = {v: k for k, v in actual_tickers.items()}
        data_close.rename(columns=reverse_ticker_map, inplace=True)
        
        # Add CASH column with constant value 1.0
        if "CASH" in ticker_mapping:
            data_close["CASH"] = 1.0
            print("Added CASH placeholder column.")
            
        # Forward fill and then backfill missing data
        data_close.ffill(inplace=True)
        data_close.bfill(inplace=True)
        
        # Check if any NaNs remain after filling
        if data_close.isnull().values.any():
            print("Warning: Remaining NaNs found in data after fill. Columns:")
            print(data_close.isnull().sum())
            # Optional: Drop rows/cols with NaNs or handle differently
            # data_close.dropna(inplace=True)
            
        print("Daily Close data download and preparation complete.")
        return data_close
        
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        print(f"Tickers attempted: {tickers_to_download}")
        return None

# -------------------------------
# Regime Classification & ATR
# -------------------------------
# Function to calculate Average True Range (ATR) - needed within classify_regime
def calculate_atr(high, low, close, length=14):
    """Calculates ATR using Wilder's smoothing."""
    # Simplified type checking
    if not isinstance(high, pd.Series):
        raise TypeError(f"Input 'high' must be a pandas Series, got {type(high)}")
    if not isinstance(low, pd.Series):
        raise TypeError(f"Input 'low' must be a pandas Series, got {type(low)}")
    if not isinstance(close, pd.Series):
        raise TypeError(f"Input 'close' must be a pandas Series, got {type(close)}")
        
    if not (len(high) == len(low) == len(close)):
        raise ValueError("Input Series must have the same length")
    if not high.index.equals(low.index) or not low.index.equals(close.index):
        # Add explicit index check
        raise ValueError("Input Series must have the same index")

    # Calculate True Range (TR)
    high_low = high - low
    # Ensure previous close is aligned; handle first NaN from shift()
    close_prev = close.shift(1)
    high_close_prev = abs(high - close_prev).fillna(0) # Fill NaN for first row calculation
    low_close_prev = abs(low - close_prev).fillna(0)  # Fill NaN for first row calculation

    # Combine TR components into a DataFrame before taking the max
    tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev})
    tr = tr_df.max(axis=1)

    # Use EMA with alpha = 1 / length for Wilder's smoothing
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def classify_regime(data_ohlc_proxy, emaFastLen=30, emaSlowLen=60, emaMarginATRLen=60, emaMarginATRMult=0.30):
    """
    Classify market regime based on the Bull/Bear Market Support Band logic.
    Uses EMA difference compared to an ATR-based margin.
    Expects a DataFrame with columns 'High', 'Low', 'Close'.
    """
    if not all(col in data_ohlc_proxy.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("Input DataFrame must contain 'High', 'Low', 'Close' columns.")

    # Explicitly squeeze to ensure Series are passed
    close = data_ohlc_proxy['Close'].squeeze()
    high = data_ohlc_proxy['High'].squeeze()
    low = data_ohlc_proxy['Low'].squeeze()

    # Calculate Moving Averages and ATR
    emaFast = close.ewm(span=emaFastLen, adjust=False).mean()
    emaSlow = close.ewm(span=emaSlowLen, adjust=False).mean()
    atr_val = calculate_atr(high, low, close, length=emaMarginATRLen)

    # Calculate EMA Difference and Margin
    emaDiff = emaFast - emaSlow
    margin = emaMarginATRMult * atr_val

    # Determine Regimes
    regimes = pd.Series(index=data_ohlc_proxy.index, dtype=str)
    regimes[emaDiff > margin] = "Bull"
    regimes[emaDiff < -margin] = "Bear"
    regimes[(emaDiff <= margin) & (emaDiff >= -margin)] = "Stagnation"
    regimes[pd.isna(margin)] = "Stagnation" # Handle potential NaNs in margin at start
    return regimes

# -------------------------------
# Allocation and Performance
# -------------------------------
def get_target_allocation(regime, current_regime_adjustments, base_alloc):
    """Combine baseline allocation with regime adjustments and normalize to sum to 1."""
    adj = current_regime_adjustments.get(regime, {})
    target = {asset: base_alloc[asset] + adj.get(asset, 0) for asset in base_alloc}
    # Ensure weights are non-negative before normalization
    target = {asset: max(0, weight) for asset, weight in target.items()}
    total = sum(target.values())
    if total <= 1e-9: # Avoid division by zero if total is near zero
        # Default to equal weight if total is zero (handle potential case with all negative adjustments)
        num_assets = len(base_alloc)
        normalized = {asset: 1.0 / num_assets if num_assets > 0 else 0 for asset in base_alloc}
    else:
        normalized = {asset: weight / total for asset, weight in target.items()}
    return normalized

def calculate_performance(regime_adjustments, data_df, base_alloc, initial_capital, rebalance_frequency, trading_cost_bps=5):
    """Runs the backtest with periodic rebalancing, calculates performance, and includes trading costs."""
    # Input validation
    if data_df is None or data_df.empty or len(data_df) < 2:
        print("Error: Invalid or insufficient data provided (need >= 2 days).")
        return None
    if 'Regime' not in data_df.columns:
        print("Error: 'Regime' column missing from data for backtest.")
        return None
    if data_df["Regime"].empty:
        print("Error: 'Regime' column is empty.")
        return None

    N = len(data_df)
    portfolio_eod = np.zeros(N)       # Value at End of Day i
    weights_sod = [{} for _ in range(N)] # Weights held during Day i (Set EOD i-1)
    target_eod = [{} for _ in range(N)]  # Target determined at EOD i
    assets_order = sorted(base_alloc.keys())
    total_trading_cost = 0.0
    cost_factor = trading_cost_bps / 10000.0 # Convert bps to decimal

    # --- Initialization for Day 0 EOD / Day 1 SOD --- 
    portfolio_eod[0] = initial_capital
    regime_0 = data_df["Regime"].iloc[0]
    target_eod[0] = get_target_allocation(regime_0, regime_adjustments, base_alloc)
    # Weights for SOD 1 are determined by target at EOD 0 (first rebalance)
    if N > 1: 
        weights_sod[1] = target_eod[0]
    last_rebalance_date = data_df.index[0]
    # Define weights for Day 0 itself (less critical as loop starts at 1)
    weights_sod[0] = target_eod[0]

    # --- Loop for Day i (from i=1 to N-1) --- 
    for i in range(1, N):
        date = data_df.index[i]
        prev_date = data_df.index[i-1]
        row = data_df.loc[date]
        prev_row = data_df.loc[prev_date]

        # Value at SOD i is EOD i-1
        sod_value = portfolio_eod[i-1]

        # Calculate EOD i Value based on weights held *during* day i (weights_sod[i])
        current_day_weights = weights_sod[i] 
        returns = {}
        for asset in assets_order:
            current_price = row.get(asset, np.nan)
            prev_price = prev_row.get(asset, np.nan)
            if pd.isna(current_price) or pd.isna(prev_price) or prev_price == 0:
                returns[asset] = 0.0
            else:
                returns[asset] = (current_price / prev_price) - 1

        daily_return = sum(current_day_weights.get(asset, 0) * returns.get(asset, 0) for asset in assets_order)
        portfolio_eod[i] = sod_value * (1 + daily_return)
        portfolio_eod[i] = max(portfolio_eod[i], 1e-9) # Floor value

        # --- Determine Weights for SOD i+1 --- 
        # Determine Target Allocation at EOD i
        regime_i = row["Regime"]
        target_eod[i] = get_target_allocation(regime_i, regime_adjustments, base_alloc)

        # --- Calculate portfolio value *before* potential rebalance for cost calculation ---
        value_before_rebalance = portfolio_eod[i]
        holdings_value_before_rebalance = {}
        for asset in assets_order:
            asset_weight_sod = weights_sod[i].get(asset, 0)
            asset_return = returns.get(asset, 0)
            holdings_value_before_rebalance[asset] = sod_value * asset_weight_sod * (1 + asset_return)

        effective_value_before_rebalance = portfolio_eod[i]

        # Decide if rebalance happens at EOD i (affects weights for SOD i+1)
        rebalance_today = False
        if not isinstance(date, pd.Timestamp) or not isinstance(last_rebalance_date, pd.Timestamp):
             print(f"Warning: Non-Timestamp dates found ({type(date)}, {type(last_rebalance_date)}). Rebalance check might fail.")
        else:
             if rebalance_frequency == 'daily':
                 rebalance_today = True
             elif rebalance_frequency == 'monthly':
                 if date.month != last_rebalance_date.month or date.year != last_rebalance_date.year:
                     rebalance_today = True
             elif rebalance_frequency == 'quarterly':
                 current_quarter = (date.month - 1) // 3 + 1
                 last_rebalance_quarter = (last_rebalance_date.month - 1) // 3 + 1
                 if current_quarter != last_rebalance_quarter or date.year != last_rebalance_date.year:
                     rebalance_today = True
             elif rebalance_frequency == 'yearly':
                 if date.year != last_rebalance_date.year:
                     rebalance_today = True

        # Set weights for SOD i+1
        if i + 1 < N: # Avoid index out of bounds on last day
            if rebalance_today:
                new_weights = target_eod[i] 
                weights_sod[i+1] = new_weights
                last_rebalance_date = date        

                # --- Calculate Trading Costs --- 
                turnover = 0.0
                for asset in assets_order:
                    current_value_asset = holdings_value_before_rebalance.get(asset, 0)
                    target_value_asset = effective_value_before_rebalance * new_weights.get(asset, 0)
                    turnover += abs(target_value_asset - current_value_asset)

                trade_volume = turnover / 2.0
                cost = trade_volume * cost_factor
                total_trading_cost += cost
                portfolio_eod[i] -= cost 
                portfolio_eod[i] = max(portfolio_eod[i], 1e-9) 

            else:
                # Carry forward weights from SOD i, accounting for market drift
                drifted_weights = {}
                total_value_eod = portfolio_eod[i] 

                if total_value_eod > 1e-9: 
                    for asset in assets_order:
                        eod_asset_value = holdings_value_before_rebalance.get(asset, 0) # Use already calculated EOD value
                        drifted_weights[asset] = eod_asset_value / total_value_eod
                    
                    norm_factor = sum(drifted_weights.values())
                    if norm_factor > 1e-9:
                       drifted_weights = {asset: weight / norm_factor for asset, weight in drifted_weights.items()}
                    else: 
                       drifted_weights = weights_sod[i] 

                else: 
                    drifted_weights = weights_sod[i] 

                weights_sod[i+1] = drifted_weights
                
    # --- Post Loop --- 
    backtest_results = pd.DataFrame({"PortfolioValue": portfolio_eod}, index=data_df.index)
    actual_weights_df = pd.DataFrame(weights_sod, index=data_df.index) 

    # --- Calculate Performance Metrics --- 
    backtest_results['DailyReturn'] = backtest_results['PortfolioValue'].pct_change().fillna(0)
    
    if backtest_results.empty or len(backtest_results.index) < 2:
        print("Warning: Not enough data points for performance calculation after pct_change.")
        return {"CAGR": 0, "MaxDrawdown": 0, "Volatility": 0, "Sharpe": np.nan, "Sortino": np.nan, "FinalValue": initial_capital, "TotalTradingCost": total_trading_cost, "Allocations": actual_weights_df, "ReturnsData": backtest_results}

    years = (backtest_results.index[-1] - backtest_results.index[0]).days / 365.25
    final_value = portfolio_eod[-1] 
    
    if years <= 0 or initial_capital <= 0 or final_value <= 0:
        cagr = 0
    else:
        cagr = (final_value / initial_capital) ** (1 / years) - 1
        
    annualized_volatility = backtest_results['DailyReturn'][1:].std() * np.sqrt(252) 

    risk_free_rate = 0.0
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    daily_returns_series = backtest_results['DailyReturn'][1:] 
    downside_returns = daily_returns_series[daily_returns_series < risk_free_rate]
    if not downside_returns.empty:
        downside_variance = (downside_returns**2).mean()
        if pd.notna(downside_variance) and downside_variance >= 0:
            downside_deviation = np.sqrt(downside_variance) * np.sqrt(252)
            sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
        else:
            sortino_ratio = np.nan 
    else:
        sortino_ratio = np.inf if cagr > risk_free_rate else 0.0 

    rolling_max = backtest_results['PortfolioValue'].cummax()
    daily_drawdown = backtest_results['PortfolioValue'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()

    return {
        "CAGR": cagr,
        "MaxDrawdown": max_drawdown if pd.notna(max_drawdown) else 0,
        "Volatility": annualized_volatility if pd.notna(annualized_volatility) else 0,
        "Sharpe": sharpe_ratio if pd.notna(sharpe_ratio) else np.nan,
        "Sortino": sortino_ratio if pd.notna(sortino_ratio) else np.nan,
        "FinalValue": final_value,
        "TotalTradingCost": total_trading_cost, 
        "Allocations": actual_weights_df, 
        "ReturnsData": backtest_results
    }

# ----------------------------------
# Optimization Helper Functions
# ----------------------------------
def flatten_adjustments(adj_dict, regimes_order, assets_order):
    """Flattens the regime adjustments dictionary into a list for the optimizer."""
    flat_list = []
    for regime in regimes_order:
        regime_data = adj_dict.get(regime, {})
        for asset in assets_order:
            flat_list.append(regime_data.get(asset, 0.0))
    return np.array(flat_list)

def reconstruct_adjustments(flat_params, regimes_order, assets_order):
    """Reconstructs the regime adjustments dictionary from a flat list."""
    adj_dict = {}
    num_assets = len(assets_order)
    for i, regime in enumerate(regimes_order):
        adj_dict[regime] = {}
        start_idx = i * num_assets
        for j, asset in enumerate(assets_order):
            adj_dict[regime][asset] = flat_params[start_idx + j]
    return adj_dict

# Objective function for maximizing CAGR (minimizing negative CAGR)
def objective_cagr(flat_params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps):
    adjustments = reconstruct_adjustments(flat_params, regimes_order, assets_order)
    performance = calculate_performance(adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
    if performance is None or pd.isna(performance["CAGR"]):
        return np.inf 
    return -performance["CAGR"] 

# Objective function for maximizing protection (minimizing Max Drawdown magnitude)
def objective_max_drawdown(flat_params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps):
    adjustments = reconstruct_adjustments(flat_params, regimes_order, assets_order)
    performance = calculate_performance(adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
    if performance is None or pd.isna(performance["MaxDrawdown"]):
        return np.inf 
    return abs(performance["MaxDrawdown"]) 

# Objective function for maximizing Sharpe Ratio (minimizing negative Sharpe)
def objective_sharpe(flat_params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps):
    adjustments = reconstruct_adjustments(flat_params, regimes_order, assets_order)
    performance = calculate_performance(adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
    if performance is None or pd.isna(performance["Sharpe"]):
        return np.inf 
    return -performance["Sharpe"] 

def find_optimal_allocation(objective_func, data_df, base_alloc, initial_cap, initial_adjustments, regimes_order, assets_order, freq, cost_bps):
    """Finds the optimal regime adjustments using the specified objective function."""
    print(f"\n--- Starting Optimization: {objective_func.__name__} ---")
    
    # Flatten initial adjustments to use as starting point
    initial_flat_params = flatten_adjustments(initial_adjustments, regimes_order, assets_order)
    num_params = len(initial_flat_params)

    # Define bounds for each adjustment parameter (e.g., -0.5 to +0.5)
    bounds = [(-0.5, 0.5)] * num_params
    
    # Define the objective wrapper to pass extra arguments
    objective_wrapper = lambda params: objective_func(
        params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps
    )

    # Optimization using L-BFGS-B (supports bounds)
    result = minimize(
        objective_wrapper,
        initial_flat_params,
        # args=(), # Removed args, objective_wrapper captures them
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-5}
    )

    if result.success:
        print(f"Optimization successful after {result.nit} iterations.")
        optimal_flat_params = result.x
        # --- Round the optimal parameters to 2 decimal places --- 
        rounded_flat_params = np.round(optimal_flat_params, 2)
        print(f"Rounded optimal flat params: {rounded_flat_params}") # Optional: Log rounded params

        # Reconstruct adjustments using ROUNDED parameters
        optimal_adjustments = reconstruct_adjustments(rounded_flat_params, regimes_order, assets_order)
        
        # Recalculate final performance with ROUNDED optimal params to get all metrics
        final_performance = calculate_performance(optimal_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        if final_performance is None:
            print("Error: Could not calculate performance with optimal parameters.")
            # Return initial if optimal fails
            return initial_adjustments, calculate_performance(initial_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        return optimal_adjustments, final_performance
    else:
        print(f"Optimization failed: {result.message}")
        # Return initial if failed, along with its performance
        initial_performance = calculate_performance(initial_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        return initial_adjustments, initial_performance

# ----------------------------------
# Plotting and Summary Functions
# ----------------------------------
def plot_results_interactive(performance_data, title_prefix="Backtest"):
    """Plots portfolio value and allocation using Plotly."""
    if performance_data is None or 'ReturnsData' not in performance_data or 'Allocations' not in performance_data:
        print("Plotting skipped: Missing ReturnsData or Allocations in performance data.")
        return

    returns_df = performance_data['ReturnsData']
    allocations_df = performance_data['Allocations']
    assets_order = allocations_df.columns.tolist() # Get asset order from actual data

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=('Portfolio Value', 'Allocation (%)'))

    # Plot Portfolio Value
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['PortfolioValue'], 
                           mode='lines', name='Portfolio Value'), 
                  row=1, col=1)

    # Plot Allocation Area Chart
    plot_cols = [col for col in assets_order if col in allocations_df.columns]
    color_sequence = px.colors.qualitative.Plotly 
    for i, asset in enumerate(plot_cols):
        fig.add_trace(go.Scatter(x=allocations_df.index, y=allocations_df[asset],
                                 mode='lines', name=asset,
                                 stackgroup='allocation', 
                                 line=dict(width=0), 
                                 fillcolor=color_sequence[i % len(color_sequence)], 
                                 hoverinfo='x+y+name'),
                      row=2, col=1)
    fig.update_layout(
        title_text=f"{title_prefix} Backtest Results",
        hovermode='x unified',
        height=800, 
        yaxis1_title="Portfolio Value ($)",
        yaxis2_title="Held Allocation (%)",
        yaxis2_tickformat='.0%'
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.show()

def display_summary(performance_data, title, initial_capital, adjustments=None):
    """Prints a summary of performance metrics and adjustments."""
    print(f"\n--- {title} ---   ")
    if performance_data is None:
        print("No performance data available.")
        return

    if adjustments:
        print("Adjustments:")
        for regime, adj_values in adjustments.items():
            # Ensure adjustment values are printed reasonably (e.g., 2-4 decimal places)
            formatted_adj = {asset: f"{val:.4f}" for asset, val in adj_values.items()}
            print(f"  {regime}: {formatted_adj}")

    print("Performance Metrics:")
    print(f"  Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"  Final Portfolio Value: ${performance_data.get('FinalValue', 0):,.2f}")
    print(f"  CAGR: {performance_data.get('CAGR', 0):.2%}")
    print(f"  Annualized Volatility: {performance_data.get('Volatility', 0):.2%}")
    print(f"  Sharpe Ratio (Rf=0%): {performance_data.get('Sharpe', float('nan')):.2f}")
    print(f"  Sortino Ratio (Rf=0%): {performance_data.get('Sortino', float('nan')):.2f}")
    print(f"  Maximum Drawdown: {performance_data.get('MaxDrawdown', 0):.2%}")
    print(f"  Total Trading Costs: ${performance_data.get('TotalTradingCost', 0):,.0f}") # Added Trading Costs (rounded)

# -------------------------------
# Core Backtest Orchestration (for one period)
# -------------------------------
def run_backtest_for_period(period_start_date, period_end_date, args, 
                            ticker_map, base_alloc, initial_adjustments, assets_order, regimes_order, 
                            cost_bps, # Backtest Execution Params
                            initial_capital=100000, 
                            rebalance_frequency='monthly',
                            # Regime Params
                            indicator_timeframe='daily',
                            regime_proxy_ticker="VOO",
                            emaFastLen=30,
                            emaSlowLen=60,
                            emaMarginATRLen=60,
                            emaMarginATRMult=0.30
                            ):
    """Refactored logic to run data download, regime classification, and selected backtests for a specific period."""
    print(f"\n{'='*20} RUNNING BACKTEST FOR PERIOD: {period_start_date.date()} to {period_end_date.date()} {'='*20}")

    # --- Parameters Now Passed In --- 
    # initial_capital = 100000 # Removed hardcoding
    # REBALANCE_FREQUENCY = 'monthly' # Removed hardcoding
    # INDICATOR_TIMEFRAME = 'daily' # Removed hardcoding
    # regime_proxy_ticker = "VOO" # Removed hardcoding
    # emaFastLen = 30 # Removed hardcoding
    # emaSlowLen = 60 # Removed hardcoding
    # emaMarginATRLen = 60 # Removed hardcoding
    # emaMarginATRMult = 0.30 # Removed hardcoding
    interval_map = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
    indicator_interval = interval_map.get(indicator_timeframe, '1d') # Use passed indicator_timeframe
    
    # --- 1. Data Download for Period --- 
    print(f"\n--- Downloading Data for {period_start_date.date()} to {period_end_date.date()} ---")
    data_close_daily = download_prepare_data(ticker_map, period_start_date, period_end_date)
    if data_close_daily is None or data_close_daily.empty:
        print(f"Skipping period {period_start_date.date()} - {period_end_date.date()} due to missing daily close data.")
        return None # Return None if period fails
        
    proxy_ohlc_indicator_tf = get_indicator_ohlc_data(
        regime_proxy_ticker, # Use passed regime_proxy_ticker
        period_start_date, 
        period_end_date, 
        interval=indicator_interval
    )
    if proxy_ohlc_indicator_tf is None or proxy_ohlc_indicator_tf.empty:
        print(f"Skipping period {period_start_date.date()} - {period_end_date.date()} due to missing indicator OHLC data.")
        return None # Return None if period fails

    if proxy_ohlc_indicator_tf.index.has_duplicates:
        proxy_ohlc_indicator_tf = proxy_ohlc_indicator_tf[~proxy_ohlc_indicator_tf.index.duplicated(keep='first')]

    # --- 2. Regime Classification for Period --- 
    print(f"\n--- Classifying Regimes for {period_start_date.date()} to {period_end_date.date()} ---")
    # Use passed regime parameters
    regimes_indicator_tf = classify_regime(
        proxy_ohlc_indicator_tf, 
        emaFastLen=emaFastLen, emaSlowLen=emaSlowLen, 
        emaMarginATRLen=emaMarginATRLen, emaMarginATRMult=emaMarginATRMult 
    )
    # Ensure data_close_daily has enough length after potential drops in download_prepare_data
    if len(data_close_daily) < 2:
        print(f"Skipping period {period_start_date.date()} - {period_end_date.date()} due to insufficient data length after download.")
        return None

    data_close_daily["Regime"] = regimes_indicator_tf.reindex(data_close_daily.index).ffill().bfill()
    if data_close_daily["Regime"].isnull().any(): data_close_daily["Regime"].fillna("Stagnation", inplace=True)
    print("Daily-Aligned Market Regime Distribution for Period:")
    print(data_close_daily["Regime"].value_counts())

    # --- Run Selected Backtests for Period --- 
    primary_result = None # Store the result of the main strategy run

    # 1. Baseline SAA Backtest
    if args.run_baseline:
        print("\n--- Running Baseline SAA Backtest (Period) ---")
        baseline_adjustments = {regime: {asset: 0.0 for asset in assets_order} for regime in regimes_order}
        baseline_performance = calculate_performance(
            baseline_adjustments, 
            data_close_daily, 
            base_alloc, 
            initial_capital, # Use passed initial_capital
            rebalance_frequency, # Use passed rebalance_frequency
            cost_bps
        )
        display_summary(baseline_performance, f"Baseline SAA Perf ({rebalance_frequency})", initial_capital, baseline_adjustments)
        if primary_result is None: 
            primary_result = baseline_performance
        # Optional: plot_results_interactive(baseline_performance, f"Baseline SAA ({period_start_date.date()} - {period_end_date.date()})")

    # 2. Initial TAA Backtest
    if args.run_initial_taa:
        print("\n--- Running Initial TAA Backtest (Period) ---")
        initial_performance = calculate_performance(
            initial_adjustments, 
            data_close_daily, 
            base_alloc, 
            initial_capital, # Use passed initial_capital
            rebalance_frequency, # Use passed rebalance_frequency
            cost_bps
        )
        display_summary(initial_performance, f"Initial TAA Perf ({rebalance_frequency}, {indicator_timeframe})", initial_capital, initial_adjustments)
        if primary_result is None or args.run_baseline: 
            primary_result = initial_performance
        # Optional: plot_results_interactive(initial_performance, f"Initial TAA ({period_start_date.date()} - {period_end_date.date()})")

    # 3. Optimization Runs
    if args.run_optimization:
        print(f"\n--- Starting Optimizations (Period, Objective: {args.opt_objective}) ---")
        
        objectives_to_run = []
        if args.opt_objective == 'all':
            objectives_to_run = [('cagr', objective_cagr), ('mdd', objective_max_drawdown), ('sharpe', objective_sharpe)]
        elif args.opt_objective == 'cagr':
            objectives_to_run = [('cagr', objective_cagr)]
        elif args.opt_objective == 'mdd':
            objectives_to_run = [('mdd', objective_max_drawdown)]
        elif args.opt_objective == 'sharpe':
            objectives_to_run = [('sharpe', objective_sharpe)]

        # --- Optimization logic within rolling window --- 
        # Function to wrap the optimization call for a single objective for this period
        def run_single_optimization_period(name, objective_func):
            print(f"\nDispatching optimization task for: {name} ({period_start_date.date()} - {period_end_date.date()})")
            # Pass parameters down to find_optimal_allocation
            optimal_adj, performance = find_optimal_allocation(
                objective_func, data_close_daily, base_alloc, initial_capital, 
                initial_adjustments, regimes_order, assets_order, 
                rebalance_frequency, cost_bps # Use passed rebalance_frequency & cost_bps
            )
            return name, optimal_adj, performance

        # Run optimizations in parallel or sequentially for the current period
        if len(objectives_to_run) > 1:
            print(f"Running {len(objectives_to_run)} optimizations in parallel for period...")
            # Use a specific backend like 'loky' for better robustness if needed
            results = Parallel(n_jobs=-1)(delayed(run_single_optimization_period)(name, func) for name, func in objectives_to_run)
        elif objectives_to_run: # Check if list is not empty
            print(f"Running 1 optimization sequentially for period...")
            results = [run_single_optimization_period(name, func) for name, func in objectives_to_run]
        else:
            results = [] # No objectives selected for optimization

        # Process and display results for this period
        print(f"\n--- Optimization Results ({period_start_date.date()} - {period_end_date.date()}) ---")
        optimization_results_dict = {} # Store results by objective name
        sharpe_optimal_adj_period = None
        for name, optimal_adj, performance in results:
            title_map = {
                'cagr': "Max Returns (CAGR)",
                'mdd': "Max Protection",
                'sharpe': "Max Sharpe"
            }
            # Modify title to include period indication
            display_title = f"{title_map[name]} Optimized ({rebalance_frequency}, {indicator_timeframe}) - Period: {period_start_date.date()} to {period_end_date.date()}"
            display_summary(performance, display_title, initial_capital, optimal_adj)
            
            optimization_results_dict[name] = performance # Store performance
            if name == 'sharpe':
                sharpe_optimal_adj_period = optimal_adj
        
        # --- Determine primary result from optimization --- 
        if args.opt_objective == 'all' and 'sharpe' in optimization_results_dict: 
             primary_result = optimization_results_dict['sharpe'] # Default to Sharpe if 'all' objectives run
        elif args.opt_objective in optimization_results_dict:
             primary_result = optimization_results_dict[args.opt_objective] # Use the specific objective run
        elif results: # Fallback if specific objective failed but others ran
             primary_result = results[0][2] # Return the first successful one

        # Handle Sharpe update prompt - only prompt if NOT in rolling mode
        if not args.run_rolling and sharpe_optimal_adj_period is not None:
            print("-"*50)
            try:
                update_choice = input("Do you want to update the initial_regime_adjustments with these Max Sharpe results? (y/n): ").lower()
                if update_choice in ['y', 'yes']:
                    print("\nPlease copy the following dictionary and paste it to replace the initial_regime_adjustments definition in your script:\n")
                    formatted_adjustments = "initial_regime_adjustments = {\n"
                    for regime, adjustments in sharpe_optimal_adj_period.items():
                        # Ensure inner dicts are formatted correctly
                        adj_str = ", ".join([f"'{k}': {v:.2f}" for k, v in adjustments.items()])
                        formatted_adjustments += f"    \"{regime}\": {{{adj_str}}},\n"
                    formatted_adjustments += "}"
                    print(formatted_adjustments)
                    print("\nReminder: Ensure the sum of adjustments within each regime is close to zero if required by your logic.")
                else:
                    print("Initial adjustments will not be updated.")
            except Exception as e:
                print(f"Could not process input: {e}. Initial adjustments will not be updated.")
            print("-"*50)
            
    print(f"\n{'='*20} COMPLETED BACKTEST FOR PERIOD: {period_start_date.date()} to {period_end_date.date()} {'='*20}")
    # Return the performance dict of the primary strategy tested
    return primary_result 