import pandas as pd
import itertools
import time

from utils import logger
import config
from data_fetcher import fetch_historical_data
from strategy import MovingAverageCrossoverStrategy
from backtester import Backtester

def optimize_strategy(
    short_window_range=range(5, 31, 5),  # e.g., 5, 10, 15, 20, 25, 30
    long_window_range=range(20, 71, 10), # e.g., 20, 30, 40, 50, 60, 70
    optimization_metric='sharpe_ratio' # 'sharpe_ratio', 'final_portfolio_value', 'total_return_pct'
):
    """
    Optimizes the MovingAverageCrossoverStrategy by testing different short and long window parameters.
    """
    logger.info("Starting strategy optimization...")
    logger.info(f"Short window range: {list(short_window_range)}")
    logger.info(f"Long window range: {list(long_window_range)}")
    logger.info(f"Optimizing for: {optimization_metric}")

    # 1. Fetch Data (once)
    historical_data = fetch_historical_data(
        ticker=config.MAC_TICKER,
        start_date=config.MAC_START_DATE,
        end_date=config.MAC_END_DATE,
        interval=config.MAC_INTERVAL
    )

    if historical_data is None or historical_data.empty:
        logger.error("Could not fetch historical data. Aborting optimization.")
        return

    all_results = []
    param_combinations = list(itertools.product(short_window_range, long_window_range))
    total_combinations = len(param_combinations)
    logger.info(f"Total parameter combinations to test: {total_combinations}")
    
    # Temporarily reduce logger level for backtester to avoid excessive output during optimization
    # This requires access to the logger instance used by backtester or a global config for log level
    # For simplicity, we'll let it log, but be aware it can be verbose.

    for i, (short_window, long_window) in enumerate(param_combinations):
        if short_window >= long_window:
            logger.debug(f"Skipping: Short window ({short_window}) >= Long window ({long_window})")
            continue

        logger.info(f"Testing combination {i+1}/{total_combinations}: Short={short_window}, Long={long_window}")
        
        # 2. Initialize Strategy with current parameters
        current_strategy = MovingAverageCrossoverStrategy(
            symbol=config.MAC_TICKER,
            short_window=short_window,
            long_window=long_window,
            order_size=config.MAC_ORDER_SIZE
        )

        # 3. Initialize and Run Backtester
        # Note: Backtester logs its own summary. Optimizer will provide a final ranked list.
        backtester = Backtester(
            strategy=current_strategy, 
            data=historical_data.copy(), # Use a copy of data if strategy modifies it (unlikely here)
            initial_capital=config.INITIAL_CAPITAL # Assuming this is defined in config
        )
        
        start_time = time.time()
        try:
            metrics = backtester.run() # This now returns a dict of metrics
            run_time = time.time() - start_time
            logger.info(f"    Combination {short_window}-{long_window} | Time: {run_time:.2f}s | Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Return: {metrics.get('total_return_pct', 0):.2f}%")
            
            result_entry = {
                'short_window': short_window,
                'long_window': long_window,
                **metrics # Unpack all metrics from the backtester run
            }
            all_results.append(result_entry)
        except Exception as e:
            logger.error(f"Error during backtest for SW={short_window}, LW={long_window}: {e}")
            # Optionally append a failure result or skip
            all_results.append({
                'short_window': short_window,
                'long_window': long_window,
                'error': str(e)
            })

    if not all_results:
        logger.warning("No results generated from optimization runs.")
        return

    # 4. Analyze Results
    results_df = pd.DataFrame(all_results)
    
    # Filter out rows with errors if they exist and optimization_metric is a column
    if 'error' in results_df.columns and optimization_metric in results_df.columns:
        results_df = results_df[results_df['error'].isna()]

    if results_df.empty or optimization_metric not in results_df.columns:
        logger.error(f"No valid results to sort by '{optimization_metric}'. Check for errors in backtests.")
        if not results_df.empty:
            print("\n--- All Attempted Optimization Results ---")
            print(results_df.to_string())
        return

    # Sort by the chosen metric
    # For metrics where higher is better (like Sharpe, return, profit factor), ascending=False
    # For metrics where lower is better (like max_drawdown), ascending=True
    ascending_order = False
    if optimization_metric == 'max_drawdown': # Max drawdown is negative, so higher (closer to 0) is better
        ascending_order = False # So we still sort descending to get the least negative
    
    sorted_results_df = results_df.sort_values(by=optimization_metric, ascending=ascending_order)

    logger.info(f"\n--- Optimization Results (Top 10 sorted by {optimization_metric}) ---")
    print(sorted_results_df.head(10).to_string())

    if not sorted_results_df.empty:
        best_params = sorted_results_df.iloc[0]
        logger.info(f"\n--- Best Parameters (based on {optimization_metric}) ---")
        logger.info(f"Short Window: {best_params['short_window']}")
        logger.info(f"Long Window: {best_params['long_window']}")
        logger.info(f"{optimization_metric.replace('_', ' ').title()}: {best_params[optimization_metric]:.2f}")
        logger.info(f"Final Portfolio Value: ${best_params['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return: {best_params['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {best_params['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {best_params['win_rate']:.2%}")
        logger.info(f"Profit Factor: {best_params['profit_factor']:.2f}")
    else:
        logger.info("No best parameters found.")

if __name__ == '__main__':
    # Ensure config has INITIAL_CAPITAL or define it here
    if not hasattr(config, 'INITIAL_CAPITAL'):
        logger.warning("INITIAL_CAPITAL not found in config.py, using default 100000.0 for optimizer.")
        config.INITIAL_CAPITAL = 100000.0 # Define a default if not in config
    
    optimize_strategy(
        short_window_range=range(10, 51, 10), # Example: 10, 20, 30, 40, 50
        long_window_range=range(30, 101, 20), # Example: 30, 50, 70, 90
        optimization_metric='sharpe_ratio' # or 'final_portfolio_value'
    )
