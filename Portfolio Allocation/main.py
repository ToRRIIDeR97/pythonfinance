import argparse
import os
import sys
import tempfile
import subprocess
import pickle
from datetime import datetime
from joblib import Parallel, delayed

import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

import matplotlib
# Set the backend before importing pyplot
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt

# Import the backtester functions
import backtester
from report_generator import generate_report, create_portfolio_plot, create_drawdown_plot

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
    "Bull": {'AGG': 0.50, 'CASH': -0.13, 'CRY': -0.11, 'DEF': -0.50, 'WLD': -0.50},
    "Bear": {'AGG': -0.44, 'CASH': -0.38, 'CRY': -0.50, 'DEF': 0.50, 'WLD': -0.32},
    "Stagnation": {'AGG': -0.46, 'CASH': 0.50, 'CRY': 0.25, 'DEF': -0.49, 'WLD': -0.43},
}

# Get asset order from baseline keys (ensures consistency)
ASSETS_ORDER = sorted(baseline_allocation.keys())
print(f"Assets defined: {ASSETS_ORDER}")

# -------------------------------
# Main Execution Logic
# -------------------------------
def main():
    # Ensure we have the os module available
    import os
    
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
    parser.add_argument('--show-chart', action='store_true', default=True,
                       help='Show allocation chart after backtest (default: True)')   # Example default
    
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

    # --- Run Selected Backtests for Period --- 
    primary_result = None # Store the result of the main strategy run
    all_periods_results = [] # Initialize the results list

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
        result = backtester.run_backtest_for_period(single_start, single_end, args, 
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
        if result is not None:
            all_periods_results.append(result)
            print(f"[DEBUG] Added result to all_periods_results. Total results: {len(all_periods_results)}")
        else:
            print("[WARNING] Backtest returned None result")

    print("\n=== Starting PDF Report Generation ===")
    print(f"Current working directory: {os.getcwd()}")
    
    # Create reports directory if it doesn't exist
    reports_dir = 'reports'
    print(f"Ensuring reports directory exists: {os.path.abspath(reports_dir)}")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Check if directory is writable
    if not os.access(reports_dir, os.W_OK):
        print(f"ERROR: Directory is not writable: {os.path.abspath(reports_dir)}")
    else:
        print(f"Reports directory is writable")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(reports_dir, f"backtest_report_{timestamp}.pdf")
    print(f"Report will be saved to: {os.path.abspath(report_filename)}")
    
    # For rolling window, use the last period's results for the report
    # For single period, use the only result
    if args.run_rolling and all_periods_results:
        # Use the most recent period's results
        performance_data = all_periods_results[-1]
        title = f"Rolling Backtest Report - {args.rolling_window_years}Y Window"
        print(f"Using rolling window results (most recent period)")
    elif not args.run_rolling and all_periods_results:
        performance_data = all_periods_results[0]
        title = f"Single Period Backtest Report - {args.overall_start_date} to {args.overall_end_date}"
        print(f"Using single period results")
    else:
        performance_data = None
        print("WARNING: No performance data available for report generation")
    
    if performance_data is not None:
        print("\nPerformance data structure:")
        if isinstance(performance_data, dict):
            print(f"  Type: dict with {len(performance_data)} keys")
            print(f"  Keys: {list(performance_data.keys())}")
            
            # Check for required keys in performance_data
            required_keys = ['FinalValue', 'TotalReturn', 'CAGR', 'Volatility', 'SharpeRatio', 'SortinoRatio', 'MaxDrawdown', 'TotalTradingCost']
            missing_keys = [k for k in required_keys if k not in performance_data]
            if missing_keys:
                print(f"WARNING: Missing required keys in performance_data: {missing_keys}")
            
            # Print sample of portfolio values if available
            if 'PortfolioValues' in performance_data:
                print(f"  PortfolioValues type: {type(performance_data['PortfolioValues'])}")
                if hasattr(performance_data['PortfolioValues'], 'head'):
                    print("  First 5 PortfolioValues:")
                    print(performance_data['PortfolioValues'].head())
        else:
            print(f"  Type: {type(performance_data)}")
            print(f"  Value: {performance_data}")
        
        print("\nAttempting to generate PDF report...")
        try:
            # Prepare backtest parameters
            backtest_params = {
                'start_date': args.overall_start_date,
                'end_date': args.overall_end_date,
                'assets': list(ticker_map.keys())[:-1],  # All assets except CASH
                'initial_capital': INITIAL_CAPITAL,
                'trading_cost_percent': TRADING_COST_BPS / 100,  # Convert from bps to percentage
                'slippage_percent': 0.0,  # Not currently used in the report
                'backtest_type': []
            }
            
            # Only include optimization parameters if optimization was actually run
            if args.run_optimization:
                backtest_params['opt_objective'] = args.opt_objective if hasattr(args, 'opt_objective') else 'sharpe'
                backtest_params['rebalance_freq'] = args.rebalance_freq if hasattr(args, 'rebalance_freq') else 'monthly'
                backtest_params['backtest_type'].append('optimization')
                
            # Add backtest type information
            if args.run_baseline:
                backtest_params['backtest_type'].append('baseline_saa')
            if args.run_initial_taa:
                backtest_params['backtest_type'].append('initial_taa')
                
            # Convert list to comma-separated string for display
            backtest_params['backtest_type'] = ', '.join(backtest_params['backtest_type'])
            
            report_path = generate_report(
                performance_data=performance_data,
                initial_capital=INITIAL_CAPITAL,
                output_path=report_filename,
                title=title,
                backtest_params=backtest_params
            )
            if os.path.exists(report_path):
                print(f"SUCCESS: Report generated at: {os.path.abspath(report_path)}")
                print(f"File size: {os.path.getsize(report_path)} bytes")
            else:
                print(f"ERROR: Report file was not created at: {os.path.abspath(report_path)}")
        except Exception as e:
            print(f"ERROR generating report: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save performance data to pickle file for visualization
        pickle_filename = os.path.join('reports', f'performance_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(performance_data, f)
        print(f"Performance data saved to: {pickle_filename}")
        
        # Show allocation chart if requested
        if args.show_chart and 'Allocations' in performance_data:
            try:
                print("\nDisplaying allocation chart...")
                # Create the plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(14, 8))
                
                # Plot each asset's allocation over time
                allocations = performance_data['Allocations']
                for column in allocations.columns:
                    plt.plot(allocations.index, allocations[column] * 100, label=column, linewidth=2)
                
                plt.title('Asset Allocation Over Time', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Allocation (%)', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # Save the plot to a file in the reports directory
                reports_dir = 'reports'
                if not os.path.exists(reports_dir):
                    os.makedirs(reports_dir)
                
                # Create a timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chart_filename = os.path.join(reports_dir, f'allocation_chart_{timestamp}.png')
                
                # Save the plot
                plt.savefig(chart_filename, bbox_inches='tight', dpi=100)
                print(f"\nAllocation chart saved to: {os.path.abspath(chart_filename)}")
                
                # Try multiple methods to open the chart
                try:
                    if sys.platform == 'darwin':  # macOS
                        # Try using 'open' command
                        subprocess.run(['open', chart_filename], check=True)
                        print("Attempting to open the chart with the default viewer...")
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(chart_filename)
                        print("Attempting to open the chart with the default viewer...")
                    else:  # Linux and others
                        subprocess.run(['xdg-open', chart_filename], check=True)
                        print("Attempting to open the chart with xdg-open...")
                except Exception as e:
                    print(f"\nCould not open the chart automatically. Please open it manually from:")
                    print(f"{os.path.abspath(chart_filename)}")
                    print(f"Error details: {e}")
                
                # Close the plot to free memory
                plt.close()
                print("\nYou can also find the chart in the 'reports' directory.")
            except Exception as e:
                print(f"Error displaying allocation chart: {e}")
    else:
        print("Skipping report generation - no performance data available")
    
    print("\nScript finished.")
    return performance_data

if __name__ == "__main__":
    main()
