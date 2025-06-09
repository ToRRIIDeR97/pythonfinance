import os
import pandas as pd
import numpy as np
from main import (
    download_data, calculate_indicators, backtest_strategy, 
    calculate_performance_metrics, RsiStrategy, MacdStrategy, 
    MaCrossStrategy, ConvergingSignalsStrategy, optimize_strategy_params
)
from generate_report import generate_pdf_report

# Risk-free rate (annualized, can be set to 0 or actual risk-free rate)
RISK_FREE_RATE = 0.0  # 0% for simplicity, adjust if needed

def calculate_buy_and_hold(data, initial_capital=100000):
    """Calculate buy and hold strategy performance."""
    portfolio = pd.DataFrame(index=data.index)
    portfolio['holdings'] = (initial_capital / data['Close'].iloc[0]) * data['Close']
    portfolio['cash'] = 0
    portfolio['total'] = portfolio['holdings']
    portfolio['position'] = 1  # Always in the market
    portfolio['returns'] = data['Close'].pct_change().fillna(0)
    return portfolio

def run_analysis():
    # Configuration
    TICKER = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_FEE_PERCENT = 0.001
    RISK_FREE_RATE = 0.02
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Download and prepare data
    print("Downloading data...")
    raw_df = download_data(TICKER, START_DATE, END_DATE)
    
    # Base indicator parameters
    base_indicator_params = {
        'SMA_short_period': 20, 'SMA_long_period': 50,
        'EMA_short_period': 12, 'EMA_long_period': 26,
        'RSI_period': 14, 'RSI_oversold': 30, 'RSI_overbought': 70,
        'ROC_period': 10,
        'MACD_fast': 12, 'MACD_slow': 26, 'MACD_signal': 9,
        'ATR_period': 14,
        'BB_period': 20, 'BB_std_dev': 2.0, 'OBV_on': True,
        'Volatility_period': 20,
        'Trend_Strength_short_sma': 20, 'Trend_Strength_long_sma': 40
    }
    
    # Calculate indicators
    data_with_indicators = calculate_indicators(raw_df.copy(), base_indicator_params)
    
    # Dictionary to store results
    results = {}
    
    # Add buy and hold benchmark
    print("\n--- Testing Buy and Hold Strategy ---")
    bh_portfolio = calculate_buy_and_hold(data_with_indicators)
    bh_metrics = calculate_performance_metrics(bh_portfolio, RISK_FREE_RATE)
    print("\nBuy and Hold Performance Metrics:")
    for metric, value in bh_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  Final Portfolio Value: {bh_portfolio['total'].iloc[-1]:.4f}")
    
    results['buy_hold'] = {
        'total_return': bh_metrics['Total Return'],
        'sharpe': bh_metrics['Sharpe Ratio'],
        'max_dd': bh_metrics['Max Drawdown'],
        'win_rate': bh_metrics['Daily Win Rate'],
        'final_value': bh_portfolio['total'].iloc[-1],
        'equity_curve': bh_portfolio['total'],
        'drawdown': (bh_portfolio['total'] / bh_portfolio['total'].cummax() - 1),
        'is_optimized': False,
        'is_benchmark': True
    }
    
    # Strategy configurations
    strategies = [
        {
            'class': RsiStrategy,
            'name': 'RSI',
            'base_params': {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30},
            'param_config': {
                'rsi_period': {'type': 'int', 'bounds': (5, 30)},
                'rsi_overbought': {'type': 'int', 'bounds': (60, 90)},
                'rsi_oversold': {'type': 'int', 'bounds': (10, 40)}
            }
        },
        {
            'class': MacdStrategy,
            'name': 'MACD',
            'base_params': {'fast': 12, 'slow': 26, 'signal': 9},
            'param_config': {
                'fast': {'type': 'int', 'bounds': (5, 20)},
                'slow': {'type': 'int', 'bounds': (15, 40)},
                'signal': {'type': 'int', 'bounds': (5, 20)}
            }
        },
        {
            'class': MaCrossStrategy,
            'name': 'MA Crossover',
            'base_params': {'short_window': 20, 'long_window': 50},
            'param_config': {
                'short_window': {'type': 'int', 'bounds': (5, 40)},
                'long_window': {'type': 'int', 'bounds': (30, 100)}
            }
        }
    macd_portfolio, macd_trades = backtest_strategy(data_with_indicators, macd_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
    
    # Calculate performance metrics
    macd_metrics = calculate_performance_metrics(macd_portfolio, RISK_FREE_RATE)
    
    # Store MACD results
    results['macd'] = {
        'total_return': macd_metrics['Total Return'],
        'sharpe': macd_metrics['Sharpe Ratio'],
        'max_dd': macd_metrics['Max Drawdown'],
        'win_rate': macd_metrics['Daily Win Rate'],
        'final_value': macd_portfolio['total'].iloc[-1],
        'equity_curve': macd_portfolio['total'],
        'drawdown': (macd_portfolio['total'] / macd_portfolio['total'].cummax() - 1),
        'is_optimized': False,
        'is_benchmark': False
    }
    
    # Test MA Crossover Strategy
    print("\n--- Testing MA Crossover Strategy ---")
    ma_params = base_indicator_params.copy()
    ma_strategy = MaCrossStrategy(data_with_indicators, ma_params)
    ma_signals = ma_strategy.generate_signals()
    ma_portfolio, ma_trades = backtest_strategy(data_with_indicators, ma_signals, INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT)
    
    # Calculate performance metrics
    ma_metrics = calculate_performance_metrics(ma_portfolio, RISK_FREE_RATE)
    
    # Store MA results
    results['ma'] = {
        'total_return': ma_metrics['Total Return'],
        'sharpe': ma_metrics['Sharpe Ratio'],
        'max_dd': ma_metrics['Max Drawdown'],
        'win_rate': ma_metrics['Daily Win Rate'],
        'final_value': ma_portfolio['total'].iloc[-1],
        'equity_curve': ma_portfolio['total'],
        'drawdown': (ma_portfolio['total'] / ma_portfolio['total'].cummax() - 1),
        'is_optimized': False,
        'is_benchmark': False
    }
    
    # Test Converging Signals Strategy
    print("\n--- Testing Converging Signals Strategy ---")
    converging_strategies = [
        RsiStrategy(data_with_indicators, rsi_params),
        MacdStrategy(data_with_indicators, macd_params),
        MaCrossStrategy(data_with_indicators, ma_params)
    ]
    
    converging_params = {'min_signals_to_converge': 2}
    converging_strategy = ConvergingSignalsStrategy(data_with_indicators, converging_strategies, converging_params)
    converging_signals = converging_strategy.generate_signals()
    
    # Create a copy of the data with indicators for the converging strategy
    converging_data = data_with_indicators.copy()
    
    # Backtest the converging strategy
    converging_portfolio, converging_trades = backtest_strategy(
        converging_data, 
        converging_signals, 
        INITIAL_CAPITAL, 
        TRANSACTION_FEE_PERCENT
    )
    
    # Calculate performance metrics
    converging_metrics = calculate_performance_metrics(converging_portfolio, RISK_FREE_RATE)
    
    # Store Converging results
    results['conv'] = {
        'total_return': converging_metrics['Total Return'],
        'sharpe': converging_metrics['Sharpe Ratio'],
        'max_dd': converging_metrics['Max Drawdown'],
        'win_rate': converging_metrics['Daily Win Rate'],
        'final_value': converging_portfolio['total'].iloc[-1],
        'equity_curve': converging_portfolio['total'],
        'drawdown': (converging_portfolio['total'] / converging_portfolio['total'].cummax() - 1),
        'is_optimized': False,
        'is_benchmark': False
    }
    
    # Generate PDF report
    print("\nGenerating PDF report...")
    report_path = generate_pdf_report('reports', results)
    print(f"\nReport generated at: {report_path}")

if __name__ == "__main__":
    run_analysis()
