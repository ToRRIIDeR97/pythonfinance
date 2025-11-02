import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # Import optimizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px # Import plotly express for color sequence
import argparse # Add argparse import

# -------------------------------
# Global Constants & Configuration
# -------------------------------
# Define assets and their corresponding tickers
ticker_map = {
    "DEF": "SCHD",    # Defensive Equity (e.g., Consumer Staples, Utilities)
    "WLD": "VT",     # World Equity (e.g., Total World Stock ETF)
    "AGG": "QQQ",    # Aggregate Bonds (e.g., Total Bond Market ETF)
    "CRY": "ETH-USD", # Crypto (e.g., Bitcoin)
    "CASH": "CASH"   # Cash placeholder
}

# Baseline Strategic Asset Allocation (SAA) - Should sum to 1.0
baseline_allocation = {
    "DEF": 0.25,
    "WLD": 0.30,
    "AGG": 0.35,
    "CRY": 0.10, # Baseline 0% Crypto, adjust only tactically
    "CASH": 0.00  # Added 10% baseline cash
}

# Define the regimes
REGIMES_ORDER = ["Bull", "Bear", "Stagnation"]

# Initial Tactical Asset Allocation (TAA) Adjustments per Regime
# Note: The SUM of adjustments within each regime MUST BE 0.
initial_regime_adjustments = {
    "Bull": {
        "DEF": -0.05,  # Slightly reduce defensive
        "WLD": 0.05,   # Slightly increase world equity
        "AGG": -0.05,  # Reduce bonds
        "CRY": 0.05,   # Increase crypto
        "CASH": 0.00   # No change in cash
    },
    "Bear": { # Modified to target ~60% cash
        "DEF": -0.05,  # Target: 30% - 5% = 25%
        "WLD": -0.30,  # Target: 40% - 30% = 10%
        "AGG": -0.05,  # Target: 20% - 5% = 15%
        "CRY": -0.10,  # Target: 0% (floor)
        "CASH": 0.50   # Target: 10% + 50% = 60%
    },
    "Stagnation": { # Neutral adjustments, potentially slight increase in cash/bonds
        "DEF": 0.00,
        "WLD": -0.05,  # Slightly reduce world equity
        "AGG": 0.00,
        "CRY": -0.05,  # Reduce crypto exposure
        "CASH": 0.10   # Increase cash
    }
}

# Get asset order from baseline keys (ensures consistency)
ASSETS_ORDER = sorted(baseline_allocation.keys())
print(f"Assets defined: {ASSETS_ORDER}")

# -------------------------------
# 1. Download Historical Data Using yfinance
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
# 2. Define a Regime Classifier Based on Bull/Bear Support Band
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
    regimes[pd.isna(margin)] = "Stagnation"
    return regimes

# -------------------------------
# 3. Define Baseline Allocation and Regime Adjustments
# -------------------------------
# Your baseline allocation:
baseline_allocation = {
    "DEF": 0.30,  # Defensive ETF
    "WLD": 0.20,  # World Stock Market ETF
    "AGG": 0.30,  # Aggressive Market ETF
    "CRY": 0.20,   # Crypto
    "CASH": 0.00  # Cash placeholder
}

# Initial Regime-based additive adjustments (Updated for new regimes)
initial_regime_adjustments = {
    "Bull": {"DEF": -0.05, "WLD": 0.05, "AGG": -0.05, "CRY": 0.05, "CASH": 0.00},
    "Bear": {"DEF": 0.10, "WLD": -0.10, "AGG": 0.05, "CRY": -0.10, "CASH": 0.05},
    "Stagnation": {"DEF": 0.00, "WLD": -0.05, "AGG": 0.00, "CRY": -0.05, "CASH": 0.10} # Default Stagnation adjustments
}

# Define consistent order for optimization parameters (Updated)
REGIMES_ORDER = ["Bull", "Bear", "Stagnation"]
ASSETS_ORDER = sorted(baseline_allocation.keys())

# Function to get target allocation based on adjustments
def get_target_allocation(regime, current_regime_adjustments, base_alloc):
    """Combine baseline allocation with regime adjustments and normalize to sum to 1."""
    adj = current_regime_adjustments.get(regime, {})
    target = {asset: base_alloc[asset] + adj.get(asset, 0) for asset in base_alloc}
    # Ensure weights are non-negative before normalization
    target = {asset: max(0, weight) for asset, weight in target.items()}
    total = sum(target.values())
    if total == 0: # Avoid division by zero if all weights become zero
        # Default to equal weight if total is zero
        num_assets = len(base_alloc)
        normalized = {asset: 1.0 / num_assets if num_assets > 0 else 0 for asset in base_alloc}
    else:
        normalized = {asset: weight / total for asset, weight in target.items()}
    return normalized

# ------------------------------------
# 4. Backtesting and Performance Calculation Function
# ------------------------------------
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
        # This requires knowing the value of each holding based on SOD i weights and EOD i prices
        value_before_rebalance = portfolio_eod[i]
        holdings_value_before_rebalance = {}
        for asset in assets_order:
            asset_weight_sod = weights_sod[i].get(asset, 0)
            # Use EOD price to value the holdings brought into the day
            asset_price_eod = row.get(asset, np.nan)
            if pd.notna(asset_price_eod) and asset_price_eod > 0 and asset_weight_sod > 0:
                holdings_value_before_rebalance[asset] = sod_value * asset_weight_sod * (row[asset] / prev_row[asset]) if prev_row.get(asset, 0) != 0 else 0
            else:
                holdings_value_before_rebalance[asset] = 0 # Or handle appropriately if price is missing

        # Summing up can slightly differ from portfolio_eod[i] due to returns calc, use portfolio_eod[i] as the total value
        effective_value_before_rebalance = portfolio_eod[i]

        # Decide if rebalance happens at EOD i (affects weights for SOD i+1)
        rebalance_today = False
        # Ensure dates are comparable (e.g., Timestamps)
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
                new_weights = target_eod[i] # Use the target calculated today
                weights_sod[i+1] = new_weights
                last_rebalance_date = date        # Update the last rebalance trigger date

                # --- Calculate Trading Costs --- 
                turnover = 0.0
                # Calculate value of each asset *after* rebalancing (target weight * current total value)
                for asset in assets_order:
                    current_value_asset = holdings_value_before_rebalance.get(asset, 0)
                    target_value_asset = effective_value_before_rebalance * new_weights.get(asset, 0)
                    turnover += abs(target_value_asset - current_value_asset)

                # Cost is applied to the total value traded (sum of absolute differences)
                # We divide turnover by 2 because buying $100 of A and selling $100 of B is $200 turnover but only $100 traded value
                trade_volume = turnover / 2.0
                cost = trade_volume * cost_factor
                total_trading_cost += cost
                portfolio_eod[i] -= cost # Deduct cost from EOD value *after* calculating returns
                portfolio_eod[i] = max(portfolio_eod[i], 1e-9) # Floor value again after cost deduction
            else:
                # Carry forward weights from SOD i
                weights_sod[i+1] = weights_sod[i]
        # else: last day, no need to set weights for tomorrow

    # --- Post Loop --- 
    backtest_results = pd.DataFrame({"PortfolioValue": portfolio_eod}, index=data_df.index)
    # The weights_sod represents the weights held at the START of each day `i`
    actual_weights_df = pd.DataFrame(weights_sod, index=data_df.index) 

    # --- Calculate Performance Metrics --- 
    backtest_results['DailyReturn'] = backtest_results['PortfolioValue'].pct_change().fillna(0)
    
    # Check for sufficient data points after pct_change
    if backtest_results.empty or len(backtest_results.index) < 2:
        print("Warning: Not enough data points for performance calculation after pct_change.")
        return {"CAGR": 0, "MaxDrawdown": 0, "Volatility": 0, "Sharpe": np.nan, "Sortino": np.nan, "FinalValue": initial_capital, "TotalTradingCost": total_trading_cost, "Allocations": actual_weights_df, "ReturnsData": backtest_results}

    years = (backtest_results.index[-1] - backtest_results.index[0]).days / 365.25
    final_value = portfolio_eod[-1] # Use last value from array
    
    # Robust CAGR calculation
    if years <= 0 or initial_capital <= 0 or final_value <= 0:
        cagr = 0
    else:
        cagr = (final_value / initial_capital) ** (1 / years) - 1
        
    annualized_volatility = backtest_results['DailyReturn'][1:].std() * np.sqrt(252) # Exclude first NaN/0 return

    risk_free_rate = 0.0
    sharpe_ratio = (cagr - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    # Robust Sortino calculation using only returns > 0 for calculation
    daily_returns_series = backtest_results['DailyReturn'][1:] # Exclude first NaN/0
    downside_returns = daily_returns_series[daily_returns_series < risk_free_rate]
    if not downside_returns.empty:
        downside_variance = (downside_returns**2).mean()
        if pd.notna(downside_variance) and downside_variance >= 0:
            downside_deviation = np.sqrt(downside_variance) * np.sqrt(252)
            sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
        else:
            sortino_ratio = np.nan # Should not happen if variance is calculated correctly
    else:
        sortino_ratio = np.inf if cagr > risk_free_rate else 0.0 # No downside returns

    # Robust Max Drawdown calculation
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
        "TotalTradingCost": total_trading_cost, # Add total cost to results
        "Allocations": actual_weights_df, # Return the log of ACTUAL weights held SOD
        "ReturnsData": backtest_results
    }

# ----------------------------------
# 5. Optimization Helper Functions
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
    # Handle cases where performance calculation fails
    if performance is None or pd.isna(performance["CAGR"]):
        return np.inf # Penalize invalid results heavily
    return -performance["CAGR"] # Minimize negative CAGR

# Objective function for maximizing protection (minimizing Max Drawdown magnitude)
def objective_max_drawdown(flat_params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps):
    adjustments = reconstruct_adjustments(flat_params, regimes_order, assets_order)
    performance = calculate_performance(adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
    # Handle cases where performance calculation fails
    if performance is None or pd.isna(performance["MaxDrawdown"]):
        return np.inf # Penalize invalid results heavily
    # Minimize the *magnitude* (absolute value) of the drawdown
    return abs(performance["MaxDrawdown"])

# Objective function for maximizing Sharpe Ratio (minimizing negative Sharpe)
def objective_sharpe(flat_params, data_df, base_alloc, initial_cap, regimes_order, assets_order, freq, cost_bps):
    adjustments = reconstruct_adjustments(flat_params, regimes_order, assets_order)
    performance = calculate_performance(adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
    # Handle cases where performance calculation fails or Sharpe is NaN
    if performance is None or pd.isna(performance["Sharpe"]):
        return np.inf # Penalize invalid results heavily
    return -performance["Sharpe"] # Minimize negative Sharpe Ratio

# ----------------------------------
# 6. Optimization Execution Functions
# ----------------------------------

def find_optimal_allocation(objective_func, data_df, base_alloc, initial_cap, initial_adjustments, regimes_order, assets_order, freq, cost_bps):
    """Finds the optimal regime adjustments using the specified objective function."""
    print(f"\n--- Starting Optimization: {objective_func.__name__} ---")
    
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
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-5}
    )
    
    if result.success:
        print(f"Optimization successful after {result.nit} iterations.")
        optimal_flat_params = result.x
        optimal_adjustments = reconstruct_adjustments(optimal_flat_params, regimes_order, assets_order)
        # Recalculate final performance with optimal params to get all metrics
        final_performance = calculate_performance(optimal_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        if final_performance is None:
            print("Error: Could not calculate performance with optimal parameters.")
            return initial_adjustments, calculate_performance(initial_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        return optimal_adjustments, final_performance
    else:
        print(f"Optimization failed: {result.message}")
        # Return initial if failed, along with its performance
        initial_performance = calculate_performance(initial_adjustments, data_df, base_alloc, initial_cap, freq, trading_cost_bps=cost_bps)
        return initial_adjustments, initial_performance

# ----------------------------------
# 7. Plotting Functions (Updated for Plotly)
# ----------------------------------
def plot_results_interactive(performance_data, title_prefix):
    """Plots portfolio value and allocation history using Plotly."""
    if performance_data is None or 'ReturnsData' not in performance_data or 'Allocations' not in performance_data:
        print(f"Skipping plotting for {title_prefix}: Missing performance data.")
        return
    returns_df = performance_data['ReturnsData']
    allocations_df = performance_data['Allocations'] # Actual SOD weights
    if returns_df.empty or allocations_df.empty:
        print(f"Skipping plotting for {title_prefix}: Empty results data.")
        return
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f"{title_prefix}: Portfolio Value Over Time",
                                        f"{title_prefix}: Actual Held Asset Allocation Over Time (Start of Day)"))
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df["PortfolioValue"],
                             mode='lines', name='Portfolio Value',
                             line=dict(color='blue')),
                  row=1, col=1)
    plot_cols = [col for col in ASSETS_ORDER if col in allocations_df.columns]
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

# --- Placeholder for Regime Indicator Plot --- 
# def plot_regime_indicator(index, high, low, close, atr_val, regimes, title, emaFastLen, emaSlowLen, emaMarginATRMult, regime_proxy_asset):
#     """Placeholder for plotting the regime indicator using Plotly."""
#     # Input validation and alignment would go here
#     # EMA calculations based on aligned data
#     # Plotting logic (lines, EMAs, shaded regions)
#     print(f"Plotting for {title} is currently disabled.")
#     pass 
#     # Example using go.Figure:
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=index, y=close, name='Close'))
#     # ... add EMA lines ...
#     # ... add shaded regions for regimes ...
#     # fig.update_layout(title=title)
#     # fig.show()

# ----------------------------------
# 8. Display Summary Function
# ----------------------------------
def display_summary(performance_data, title, initial_capital, adjustments=None):
    """Prints a summary of performance metrics and adjustments."""
    print(f"\n--- {title} ---   ")
    if performance_data is None:
        print("No performance data available.")
        return

    if adjustments:
        print("Adjustments:")
        for regime, adj_values in adjustments.items():
            print(f"  {regime}: {adj_values}")

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
# Main Execution Logic
# -------------------------------
if __name__ == "__main__":
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Run Regime-Based Tactical Asset Allocation Backtests.")
    parser.add_argument('--run-baseline', action='store_true', help='Run the baseline SAA backtest.')
    parser.add_argument('--run-initial-taa', action='store_true', help='Run the initial TAA backtest.')
    parser.add_argument('--run-optimization', action='store_true', help='Run the TAA optimization backtests.')
    parser.add_argument('--opt-objective', type=str, default='all', choices=['cagr', 'mdd', 'sharpe', 'all'], 
                        help='Objective function for optimization (or \'all\'). Only used if --run-optimization is set.')
    args = parser.parse_args()

    # --- Parameters --- 
    start_date = "2017-01-01"
    end_date = "2024-12-31" 
    initial_capital = 100000
    REBALANCE_FREQUENCY = 'weekly' # Options: 'daily', 'monthly', 'quarterly', 'yearly'
    TRADING_COST_BPS = 5 # Define trading cost in basis points (e.g., 5 bps = 0.05%)

    # Regime Indicator Parameters
    INDICATOR_TIMEFRAME = 'daily' # Options: 'daily', 'weekly', 'monthly'
    regime_proxy_ticker = "VOO"  # Original ticker for the proxy asset
    emaFastLen = 30
    emaSlowLen = 60
    emaMarginATRLen = 60
    emaMarginATRMult = 0.30

    # Map friendly timeframe names to yfinance intervals
    interval_map = {'daily': '1d', 'weekly': '1wk', 'monthly': '1mo'}
    indicator_interval = interval_map.get(INDICATOR_TIMEFRAME, '1d')

    # --- 1. Data Download --- 
    # Download Daily Close data for all assets for backtesting
    data_close_daily = download_prepare_data(ticker_map, start_date, end_date)
    if data_close_daily is None:
        print("Exiting due to errors downloading daily close data.")
        exit()
        
    # Download OHLC data for the proxy asset at the indicator timeframe
    proxy_ohlc_indicator_tf = get_indicator_ohlc_data(
        regime_proxy_ticker, 
        start_date, 
        end_date, 
        interval=indicator_interval
    )
    if proxy_ohlc_indicator_tf is None:
        print(f"Exiting due to errors downloading {INDICATOR_TIMEFRAME} OHLC data for proxy {regime_proxy_ticker}.")
        exit()

    # Check for and remove duplicate index entries (can happen with some intervals)
    if proxy_ohlc_indicator_tf.index.has_duplicates:
        print(f"Warning: Duplicate dates found in {INDICATOR_TIMEFRAME} indicator data index. Keeping first occurrence.")
        proxy_ohlc_indicator_tf = proxy_ohlc_indicator_tf[~proxy_ohlc_indicator_tf.index.duplicated(keep='first')]

    # --- 2. Regime Classification --- 
    print(f"Classifying regimes using {INDICATOR_TIMEFRAME} data for {regime_proxy_ticker}...")
    
    # Extract columns and ensure they are Series using squeeze()
    proxy_high = proxy_ohlc_indicator_tf['High'].squeeze()
    proxy_low = proxy_ohlc_indicator_tf['Low'].squeeze()
    proxy_close = proxy_ohlc_indicator_tf['Close'].squeeze()
    
    # Calculate ATR for the indicator timeframe first
    atr_indicator_tf = calculate_atr(
        proxy_high, 
        proxy_low, 
        proxy_close, 
        length=emaMarginATRLen
    )
    if atr_indicator_tf is None: # Check if ATR calculation failed
         print("Error: Failed to calculate ATR for indicator timeframe. Exiting.")
         exit()
         
    # Classify regimes (Note: classify_regime now recalculates ATR internally, ideally refactor later)
    regimes_indicator_tf = classify_regime(
        proxy_ohlc_indicator_tf, 
        emaFastLen=emaFastLen, emaSlowLen=emaSlowLen, 
        emaMarginATRLen=emaMarginATRLen, emaMarginATRMult=emaMarginATRMult
    )
    # Align regimes to daily data - Use ffill().bfill() instead of fillna(method=...)
    data_close_daily["Regime"] = regimes_indicator_tf.reindex(data_close_daily.index).ffill().bfill()
    if data_close_daily["Regime"].isnull().any(): data_close_daily["Regime"].fillna("Stagnation", inplace=True)
    print("Regime classification and alignment complete.")
    print("\nDaily-Aligned Market Regime Distribution:")
    print(data_close_daily["Regime"].value_counts())

    # --- Plot Regime Indicator --- 
    # Plotting call removed as per request

    # --- Run Selected Backtests --- 

    # 1. Baseline SAA Backtest (No Adjustments)
    if args.run_baseline:
        print("\n--- Running Baseline SAA Backtest ---")
        baseline_adjustments = {regime: {asset: 0.0 for asset in ASSETS_ORDER} for regime in REGIMES_ORDER}
        baseline_performance = calculate_performance(
            baseline_adjustments, # Zero adjustments for baseline SAA
            data_close_daily, 
            baseline_allocation, 
            initial_capital, 
            REBALANCE_FREQUENCY,
            TRADING_COST_BPS 
        )
        display_summary(baseline_performance, f"Baseline SAA Performance ({REBALANCE_FREQUENCY} rebalance)", initial_capital, baseline_adjustments)
        plot_results_interactive(baseline_performance, f"Baseline SAA Backtest ({REBALANCE_FREQUENCY} rebalance)")

    # 2. Initial TAA Backtest (Hardcoded Adjustments)
    if args.run_initial_taa:
        # Pass the daily close data with the daily-aligned regimes
        print("\n--- Running Initial TAA Backtest (Using Forced Adjustments) ---")
        
        # --- Force Override of Adjustments for Testing --- # Keep this override for the initial test
        # print("!!! Overriding initial_regime_adjustments for this run !!!") # Optional: Keep if you want the print message
        forced_adjustments = {
            "Bull": {
                "DEF": -0.05, "WLD": 0.05, "AGG": -0.05, "CRY": 0.05, "CASH": 0.00
            },
            "Bear": { # Target ~60% cash
                "DEF": -0.05, "WLD": -0.30, "AGG": -0.05, "CRY": -0.10, "CASH": 0.50
            },
            "Stagnation": { 
                "DEF": 0.00, "WLD": -0.05, "AGG": 0.00, "CRY": -0.05, "CASH": 0.10
            }
        }
        # --- End Override ---
        
        initial_performance = calculate_performance(
            forced_adjustments, # Use the overridden adjustments
            data_close_daily, # Use daily data with aligned regimes
            baseline_allocation, 
            initial_capital, 
            REBALANCE_FREQUENCY,
            TRADING_COST_BPS # Pass cost parameter
        )
        display_summary(initial_performance, f"Initial TAA Strategy Performance ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal)", initial_capital, forced_adjustments) # Display the forced adjustments
        plot_results_interactive(initial_performance, f"Initial TAA Strategy Backtest ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal)")

    # 3. Optimization Runs
    if args.run_optimization:
        # Pass daily data with aligned regimes to the optimizer
        print(f"\n--- Starting Optimizations ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal, Objective: {args.opt_objective}) ---")
        
        # Optimize for Max Returns (Max CAGR)
        if args.opt_objective in ['cagr', 'all']:
            optimal_cagr_adj, cagr_performance = find_optimal_allocation(
                objective_cagr, 
                data_close_daily, 
                baseline_allocation, 
                initial_capital, 
                initial_regime_adjustments, 
                REGIMES_ORDER, 
                ASSETS_ORDER, 
                REBALANCE_FREQUENCY,
                TRADING_COST_BPS # Pass cost parameter
            )
            display_summary(cagr_performance, f"Max Returns (CAGR) Optimized Strategy ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal)", initial_capital, optimal_cagr_adj)
            # plot_results_interactive(cagr_performance, "Max Returns Optimized Backtest") # Optional plotting

        # Optimize for Max Protection (Min Max Drawdown)
        if args.opt_objective in ['mdd', 'all']:
            optimal_mdd_adj, mdd_performance = find_optimal_allocation(
                objective_max_drawdown, 
                data_close_daily,
                baseline_allocation, 
                initial_capital, 
                initial_regime_adjustments, 
                REGIMES_ORDER, 
                ASSETS_ORDER, 
                REBALANCE_FREQUENCY,
                TRADING_COST_BPS # Pass cost parameter
            )
            display_summary(mdd_performance, f"Max Protection Optimized Strategy ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal)", initial_capital, optimal_mdd_adj)
            # plot_results_interactive(mdd_performance, "Max Protection Optimized Backtest") # Optional plotting

        # Optimize for Max Sharpe Ratio (Min Negative Sharpe)
        if args.opt_objective in ['sharpe', 'all']:
            optimal_sharpe_adj, sharpe_performance = find_optimal_allocation(
                objective_sharpe, 
                data_close_daily, 
                baseline_allocation, 
                initial_capital, 
                initial_regime_adjustments, 
                REGIMES_ORDER, 
                ASSETS_ORDER, 
                REBALANCE_FREQUENCY,
                TRADING_COST_BPS # Pass cost parameter
            )
            display_summary(sharpe_performance, f"Max Sharpe Optimized Strategy ({REBALANCE_FREQUENCY} rebalance, {INDICATOR_TIMEFRAME} signal)", initial_capital, optimal_sharpe_adj)
            # plot_results_interactive(sharpe_performance, "Max Sharpe Ratio Optimized Backtest") # Optional plotting

    print("\nScript finished.")
