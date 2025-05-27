import pandas as pd
import numpy as np
import config
from utils import logger
from data_fetcher import fetch_historical_data
from strategy import MovingAverageCrossoverStrategy # Assuming strategy.py contains this

class Backtester:
    """
    A simple event-driven backtester for strategies using OHLCV data.
    """
    def __init__(self, strategy, data, initial_capital=10000.0):
        """
        Args:
            strategy: An instance of a trading strategy (e.g., MovingAverageCrossoverStrategy).
            data (pd.DataFrame): Historical market data with 'Open', 'High', 'Low', 'Close', 'Volume'.
                                 The DataFrame index should be a DatetimeIndex.
            initial_capital (float): Starting capital for the backtest.
        """
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity (positive for long, negative for short)
        self.trades = []       # List to store details of each trade
        self.portfolio_history = [] # List to store portfolio value over time
        self.periodic_returns = [] # For Sharpe Ratio calculation

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")

    def _simulate_execution(self, order, current_bar_data):
        """
        Simulates order execution. For simplicity, assumes market orders fill at current bar's close price.
        Updates positions and capital.
        Args:
            order (dict): The order dictionary from the strategy.
            current_bar_data (pd.Series): The current bar's data.
        """
        symbol = order['symbol']
        side = order['side']
        amount = order['amount']
        raw_price = current_bar_data['Close'] # Assume fill at close for market orders
        price = None
        if isinstance(raw_price, pd.Series):
            try:
                price = raw_price.item() # Extract scalar if it's a single-item Series
            except ValueError:
                logger.error(f"[{current_bar_data.name.date()}] Error: 'Close' price for {symbol} at {current_bar_data.name} is a non-scalar Series: {raw_price}. Skipping execution for this order.")
                return
        else:
            price = float(raw_price) # Ensure it's a float

        if pd.isna(price):
            logger.warning(f"[{current_bar_data.name.date()}] NaN price for {symbol} at {current_bar_data.name}. Skipping execution.")
            return

        order_type = order['type'] # e.g. 'buy-market', 'sell-market'

        trade_cost = 0
        current_position = self.positions.get(symbol, 0)

        logger.info(f"[{current_bar_data.name.date()}] Attempting to execute: {side} {amount} {symbol} at ~{float(price):.2f}")

        if side == 'buy':
            trade_cost = amount * price
            if self.current_capital >= trade_cost:
                self.current_capital -= trade_cost
                self.positions[symbol] = current_position + amount
                self.trades.append({
                    'timestamp': current_bar_data.name,
                    'symbol': symbol,
                    'side': 'BUY',
                    'amount': amount,
                    'price': price,
                    'cost': trade_cost
                })
                logger.info(f"    Executed BUY: {amount} {symbol} at {price:.2f}. Cost: {trade_cost:.2f}. Capital: {self.current_capital:.2f}")
            else:
                logger.warning(f"    Insufficient capital to BUY {amount} {symbol}. Required: {trade_cost:.2f}, Available: {self.current_capital:.2f}")
        elif side == 'sell':
            # For simplicity, allow short selling or selling existing long position
            proceeds = amount * price
            self.current_capital += proceeds
            self.positions[symbol] = current_position - amount
            self.trades.append({
                'timestamp': current_bar_data.name,
                'symbol': symbol,
                'side': 'SELL',
                'amount': amount,
                'price': price,
                'proceeds': proceeds
            })
            logger.info(f"    Executed SELL: {amount} {symbol} at {price:.2f}. Proceeds: {proceeds:.2f}. Capital: {self.current_capital:.2f}")
        else:
            logger.warning(f"Unsupported order side: {side}")

    def run(self):
        """Runs the backtest event loop."""
        logger.info(f"Starting backtest for strategy {self.strategy.__class__.__name__} on {self.strategy.symbol}")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")

        for timestamp, current_bar in self.data.iterrows():
            # 1. Generate signal from strategy
            # The strategy itself might update its internal state (like MAs)
            order = self.strategy.generate_signal(current_bar)

            # 2. Simulate execution if signal is generated
            if order:
                self._simulate_execution(order, current_bar)
            
            # 3. Update portfolio value for this period
            current_portfolio_value = self.current_capital
            for symbol, quantity in self.positions.items():
                if quantity != 0:
                    raw_close_price = current_bar['Close']
                    close_price_scalar = None
                    if isinstance(raw_close_price, pd.Series):
                        try:
                            close_price_scalar = raw_close_price.item()
                        except ValueError:
                            logger.warning(f"[{timestamp.date()}] Mark-to-market: 'Close' price for {symbol} is non-scalar Series: {raw_close_price}. Using NaN.")
                            close_price_scalar = float('nan')
                    else:
                        close_price_scalar = float(raw_close_price)
                    
                    if not pd.isna(close_price_scalar):
                        current_portfolio_value += quantity * close_price_scalar # Mark-to-market
                    else:
                        logger.warning(f"[{timestamp.date()}] Mark-to-market: NaN close price for {symbol}. Position value not updated for this bar.")
            self.portfolio_history.append({'timestamp': timestamp, 'value': current_portfolio_value})

        logger.info("Backtest finished.")

        # Calculate periodic returns from portfolio history
        if len(self.portfolio_history) > 1:
            portfolio_values = pd.Series([p['value'] for p in self.portfolio_history])
            self.periodic_returns = portfolio_values.pct_change().dropna().to_list()
        else:
            self.periodic_returns = []

        return self.print_summary()

    def print_summary(self):
        """Prints a summary of the backtest results and returns key metrics."""
        """Prints a summary of the backtest results."""
        raw_final_portfolio_value = self.portfolio_history[-1]['value'] if self.portfolio_history else self.initial_capital
        final_portfolio_value = None

        if isinstance(raw_final_portfolio_value, pd.Series):
            try:
                final_portfolio_value = raw_final_portfolio_value.item()
            except ValueError:
                logger.error(f"Error: final portfolio value is a non-scalar Series: {raw_final_portfolio_value}. Using initial capital for summary.")
                final_portfolio_value = float(self.initial_capital)
        elif pd.isna(raw_final_portfolio_value):
            logger.warning("Warning: final portfolio value is NaN. Using initial capital for summary.")
            final_portfolio_value = float(self.initial_capital)
        else:
            final_portfolio_value = float(raw_final_portfolio_value)

        if pd.isna(final_portfolio_value):
            # Fallback if it's still NaN after attempts (e.g. initial_capital was also problematic, though unlikely)
            final_portfolio_value = 0.0 

        # Recalculate total_return_pct with the scalar final_portfolio_value
        if self.initial_capital == 0: # Avoid division by zero
            total_return_pct = 0.0 if final_portfolio_value == 0 else float('inf')
        else:
            total_return_pct = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        num_trades = len(self.trades)

        logger.info("--- Backtest Summary ---")
        logger.info(f"Symbol: {self.strategy.symbol}")
        logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Period: {self.data.index.min().date()} to {self.data.index.max().date()}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${float(final_portfolio_value):,.2f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"Total Trades Executed (legs): {num_trades}")

        # --- Advanced Metrics ---
        if self.periodic_returns:
            sharpe_ratio = self._calculate_sharpe_ratio(self.periodic_returns)
            logger.info(f"Annualized Sharpe Ratio (rf=0%): {sharpe_ratio:.2f}")
        else:
            logger.info("Sharpe Ratio: N/A (Not enough data for returns)")

        max_drawdown = self._calculate_max_drawdown([p['value'] for p in self.portfolio_history])
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")

        win_rate, profit_factor = self._calculate_win_rate_profit_factor()
        if win_rate is not None:
            logger.info(f"Win Rate (Round Trips): {win_rate:.2%}")
            logger.info(f"Profit Factor (Round Trips): {profit_factor:.2f}")
        else:
            logger.info("Win Rate/Profit Factor: N/A (No round trip trades)")
            win_rate = 0.0 # Default if not calculable
            profit_factor = 0.0 # Default if not calculable

        if num_trades > 0:
            logger.info("--- Trades --- ")

        # Prepare metrics dictionary for return
        metrics = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return_pct': total_return_pct,
            'total_trades_executed': num_trades,
            'sharpe_ratio': sharpe_ratio if 'sharpe_ratio' in locals() and self.periodic_returns else 0.0,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate if win_rate is not None else 0.0,
            'profit_factor': profit_factor if profit_factor is not None else 0.0
        }
        return metrics
            for trade in self.trades:
                log_msg = f"  {trade['timestamp'].date()}: {trade['side']} {trade['amount']} {trade['symbol']} at {trade['price']:.2f}"
                if 'cost' in trade:
                    log_msg += f" (Cost: {trade['cost']:.2f})"
                elif 'proceeds' in trade:
                    log_msg += f" (Proceeds: {trade['proceeds']:.2f})"
                logger.info(log_msg)

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0, periods_per_year=252):
        """Calculates annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        if std_return == 0:
            return np.inf if mean_return > risk_free_rate else 0.0 # Avoid division by zero
        sharpe = (mean_return - (risk_free_rate / periods_per_year)) / std_return
        return sharpe * np.sqrt(periods_per_year)

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculates maximum drawdown."""
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        values_series = pd.Series(portfolio_values)
        cumulative_max = values_series.cummax()
        drawdown = (values_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown if not pd.isna(max_drawdown) else 0.0

    def _calculate_win_rate_profit_factor(self):
        """Calculates win rate and profit factor from trades (simplified round trips)."""
        if not self.trades:
            return None, None

        num_winning_trades = 0
        num_losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        # Create a structure to track open buy positions for FIFO-like matching
        # symbol -> list of {'price': float, 'amount': float, 'timestamp': datetime}
        open_buy_positions = {}

        sorted_trades = sorted(self.trades, key=lambda x: x['timestamp'])

        for trade in sorted_trades:
            symbol = trade['symbol']
            amount = trade['amount']
            price = trade['price']

            if symbol not in open_buy_positions:
                open_buy_positions[symbol] = []

            if trade['side'] == 'BUY':
                open_buy_positions[symbol].append({'price': price, 'amount': amount, 'timestamp': trade['timestamp']})
            elif trade['side'] == 'SELL':
                amount_to_sell = amount
                while amount_to_sell > 0 and open_buy_positions[symbol]:
                    buy_trade_info = open_buy_positions[symbol][0] # FIFO
                    
                    match_amount = min(amount_to_sell, buy_trade_info['amount'])
                    
                    pnl_per_share = price - buy_trade_info['price']
                    trade_pnl = pnl_per_share * match_amount

                    if trade_pnl > 0:
                        num_winning_trades += 1
                        gross_profit += trade_pnl
                    elif trade_pnl < 0:
                        num_losing_trades += 1
                        gross_loss += abs(trade_pnl)
                    # If trade_pnl is 0, it's a break-even trade, not counted as win or loss here for simplicity
                    # but contributes to total trades for win rate calculation denominator.

                    buy_trade_info['amount'] -= match_amount
                    amount_to_sell -= match_amount

                    if buy_trade_info['amount'] <= 1e-9: # Effectively zero, considering float precision
                        open_buy_positions[symbol].pop(0)
                
                if amount_to_sell > 1e-9: # If sell amount couldn't be fully matched (i.e. short selling without prior buy)
                    # This simple model doesn't fully support complex short selling P&L without a clear opening short trade
                    # For now, such unmatched sells won't contribute to win/loss stats unless paired later.
                    pass 

        total_closed_trades = num_winning_trades + num_losing_trades
        if total_closed_trades == 0:
            return None, None # No round trips completed

        win_rate = num_winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
        
        return win_rate, profit_factor

if __name__ == '__main__':
    logger.info("Setting up Moving Average Crossover backtest...")

    # 1. Fetch Data
    historical_data = fetch_historical_data(
        ticker=config.MAC_TICKER,
        start_date=config.MAC_START_DATE,
        end_date=config.MAC_END_DATE,
        interval=config.MAC_INTERVAL
    )

    if historical_data is not None and not historical_data.empty:
        # 2. Initialize Strategy
        mac_strategy = MovingAverageCrossoverStrategy(
            symbol=config.MAC_TICKER,
            short_window=config.MAC_SHORT_WINDOW,
            long_window=config.MAC_LONG_WINDOW,
            order_size=config.MAC_ORDER_SIZE
            # No order_queue or risk_manager needed for this simple backtester
        )

        # 3. Initialize and Run Backtester
        backtester = Backtester(strategy=mac_strategy, data=historical_data, initial_capital=100000.0)
        results = backtester.run()
        # The results are now also printed by print_summary, which is called by run().
        # If you want to use the results dict here, you can:
        # logger.info(f"Returned metrics: {results}")
    else:
        logger.error("Could not fetch historical data. Aborting backtest.")
