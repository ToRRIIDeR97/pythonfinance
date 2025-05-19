import yfinance as yf
import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
from pandas.tseries.offsets import DateOffset

# Import the backtester functions
import backtester

# -------------------------------
# Global Configuration (Keep or move to separate config file?)
# -------------------------------
# Define assets and their corresponding tickers
ticker_map = {
    "DEF": "SCHD",    # Defensive Equity (e.g., Consumer Staples, Utilities)
    "WLD": "VOO",     # World Equity (e.g., Total World Stock ETF)
    "AGG": "QQQ",    # Aggregate Bonds (e.g., Total Bond Market ETF)
    "CRY": "ETH-USD", # Crypto (e.g., Bitcoin)
    "CASH": "CASH"   # Cash placeholder
}

# Baseline Strategic Asset Allocation (SAA) - Should sum to 1.0
baseline_allocation = {
    "DEF": 0.25,
    "WLD": 0.30,
    "AGG": 0.35,
    "CRY": 0.10, 
    "CASH": 0.00
}

# Define the regimes order
REGIMES_ORDER = ["Bull", "Bear", "Stagnation"]

# Initial Tactical Asset Allocation (TAA) Adjustments per Regime
# Note: The SUM of adjustments within each regime MUST BE 0 (or handled by normalization).
initial_regime_adjustments = {
    "Bull": {'AGG': 0.18, 'CASH': 0.5, 'CRY': -0.11, 'DEF': -0.5, 'WLD': -0.5},
    "Bear": {'AGG': -0.44, 'CASH': -0.38, 'CRY': -0.5, 'DEF': 0.5, 'WLD': -0.22},
    "Stagnation": {'AGG': -0.46, 'CASH': 0.49, 'CRY': -0.1, 'DEF': -0.49, 'WLD': -0.43}
}

# Get asset order from baseline keys (ensures consistency)
ASSETS_ORDER = sorted(baseline_allocation.keys())
print(f"Assets defined: {ASSETS_ORDER}")

# -------------------------------
# Main Execution Logic
# -------------------------------
if __name__ == "__main__":
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Run Regime-Based Tactical Asset Allocation Backtests.")
    # Backtest Selection
    parser.add_argument('--run-baseline', action='store_true', help='Run the baseline SAA backtest.')
    parser.add_argument('--run-initial-taa', action='store_true', help='Run the initial TAA backtest.')
    parser.add_argument('--run-optimization', action='store_true', help='Run the TAA optimization backtests.')
    parser.add_argument('--opt-objective', type=str, default='all', choices=['cagr', 'mdd', 'sharpe', 'all'], 
                        help='Objective function for optimization (or \'all\'). Only used if --run-optimization is set.')
    # Rolling Window Parameters
    parser.add_argument('--run-rolling', action='store_true', help='Enable rolling window backtesting.')
    parser.add_argument('--rolling-window-years', type=int, default=5, help='Duration of the rolling window in years.')
    parser.add_argument('--rolling-step-months', type=int, default=1, help='Number of months to step the window forward.')
    parser.add_argument('--overall-start-date', type=str, default="2020-01-01", help='Overall start date for rolling analysis.') # Example default
    parser.add_argument('--overall-end-date', type=str, default="2024-12-31", help='Overall end date for rolling analysis.')   # Example default
    
    args = parser.parse_args()

    # --- Global Parameters --- 
    # These could also be moved to a config file or GUI settings
    INITIAL_CAPITAL = 100000
    REBALANCE_FREQUENCY = 'monthly' 
    TRADING_COST_BPS = 5 
    # Regime Parameters (Make these configurable?)
    INDICATOR_TIMEFRAME = 'daily' 
    REGIME_PROXY_TICKER = "VOO"
    EMA_FAST_LEN = 30
    EMA_SLOW_LEN = 60
    EMA_MARGIN_ATR_LEN = 60
    EMA_MARGIN_ATR_MULT = 0.30

    # Use global config dicts/lists
    base_alloc = baseline_allocation
    initial_adj = initial_regime_adjustments
    ticker_map_config = ticker_map
    assets_order_config = ASSETS_ORDER
    regimes_order_config = REGIMES_ORDER

    # --- Execution Flow --- 
    all_periods_results = [] # List to store results from each period

    if args.run_rolling:
        # --- Rolling Window Execution --- 
        print("--- Generating Rolling Window Periods --- ")
        overall_start = pd.to_datetime(args.overall_start_date)
        overall_end = pd.to_datetime(args.overall_end_date)
        window_years = args.rolling_window_years
        step_months = args.rolling_step_months
        
        periods = [] # List to store (start_date, end_date) tuples
        current_start = overall_start
        while True:
            current_end = current_start + DateOffset(years=window_years)
            if current_end > overall_end:
                current_end = overall_end # Adjust final window end date
                # Ensure final window is reasonably long (e.g., > 1 month) before adding
                if current_start < current_end - DateOffset(months=1): 
                    periods.append((current_start, current_end))
                break # Exit loop after adding the potentially adjusted final window
            
            periods.append((current_start, current_end))
            
            # Move to the next window start date
            next_start = current_start + DateOffset(months=step_months)
            # Stop if next start date is beyond overall end date
            if next_start >= overall_end: 
                 break 
            current_start = next_start
        
        print(f"Generated {len(periods)} rolling periods.")
        print("--- Starting Parallel Rolling Window Backtests --- ")

        # Run backtests for all periods in parallel
        # Pass necessary config lists/dicts AND parameters to the function
        all_periods_results = Parallel(n_jobs=-1)(delayed(backtester.run_backtest_for_period)(
            p_start, p_end, args, 
            ticker_map_config, base_alloc, initial_adj, assets_order_config, regimes_order_config, 
            TRADING_COST_BPS,
            # Pass parameters
            initial_capital=INITIAL_CAPITAL,
            rebalance_frequency=REBALANCE_FREQUENCY,
            indicator_timeframe=INDICATOR_TIMEFRAME,
            regime_proxy_ticker=REGIME_PROXY_TICKER,
            emaFastLen=EMA_FAST_LEN,
            emaSlowLen=EMA_SLOW_LEN,
            emaMarginATRLen=EMA_MARGIN_ATR_LEN,
            emaMarginATRMult=EMA_MARGIN_ATR_MULT
            ) for p_start, p_end in periods)

        # --- Aggregate and Display Average Results --- 
        valid_results = [res for res in all_periods_results if res is not None] # Filter out None results from skipped periods
        num_successful_periods = len(valid_results)
        print(f"\n{'='*20} ROLLING BACKTEST SUMMARY ({num_successful_periods} successful periods) {'='*20}")
        
        if num_successful_periods > 0:
            avg_metrics = {}
            # Use keys present in the first valid result dict as reference
            metric_keys = [k for k in valid_results[0].keys() if isinstance(valid_results[0][k], (int, float))]
            # metric_keys = ['CAGR', 'Volatility', 'Sharpe', 'Sortino', 'MaxDrawdown', 'FinalValue', 'TotalTradingCost']
            for key in metric_keys:
                # Calculate average, handling potential NaNs
                metric_values = [res.get(key, np.nan) for res in valid_results]
                try: # Add try-except for potential issues with nanmean
                    avg_metrics[key] = np.nanmean(metric_values)
                except TypeError: # Handle cases like averaging complex numbers or other non-numerics
                    print(f"Warning: Could not calculate average for metric '{key}'")
                    avg_metrics[key] = None 

            print("Average Performance Metrics Across Rolling Windows:")
            # Display only if average was calculated successfully
            if avg_metrics.get('CAGR') is not None: print(f"  Average CAGR: {avg_metrics['CAGR']:.2%}")
            if avg_metrics.get('Volatility') is not None: print(f"  Average Annualized Volatility: {avg_metrics['Volatility']:.2%}")
            if avg_metrics.get('Sharpe') is not None: print(f"  Average Sharpe Ratio (Rf=0%): {avg_metrics['Sharpe']:.2f}")
            if avg_metrics.get('Sortino') is not None: print(f"  Average Sortino Ratio (Rf=0%): {avg_metrics['Sortino']:.2f}")
            if avg_metrics.get('MaxDrawdown') is not None: print(f"  Average Maximum Drawdown: {avg_metrics['MaxDrawdown']:.2%}")
            if avg_metrics.get('FinalValue') is not None: print(f"  Average Final Portfolio Value: ${avg_metrics['FinalValue']:,.2f}")
            if avg_metrics.get('TotalTradingCost') is not None: print(f"  Average Total Trading Costs: ${avg_metrics['TotalTradingCost']:,.0f}")
        else:
            print("No successful periods completed to calculate average results.")

    else:
        # --- Single Period Execution (Original Logic) --- 
        print("--- Starting Single Period Backtest --- ")
        single_start = pd.to_datetime(args.overall_start_date)
        single_end = pd.to_datetime(args.overall_end_date)
        # Pass necessary config lists/dicts AND parameters to the function
        backtester.run_backtest_for_period(single_start, single_end, args, 
                                          ticker_map_config, base_alloc, initial_adj, assets_order_config, regimes_order_config,
                                          TRADING_COST_BPS,
                                          # Pass parameters
                                          initial_capital=INITIAL_CAPITAL,
                                          rebalance_frequency=REBALANCE_FREQUENCY,
                                          indicator_timeframe=INDICATOR_TIMEFRAME,
                                          regime_proxy_ticker=REGIME_PROXY_TICKER,
                                          emaFastLen=EMA_FAST_LEN,
                                          emaSlowLen=EMA_SLOW_LEN,
                                          emaMarginATRLen=EMA_MARGIN_ATR_LEN,
                                          emaMarginATRMult=EMA_MARGIN_ATR_MULT
                                          )

    print("\nScript finished.")
