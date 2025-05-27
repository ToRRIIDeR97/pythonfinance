import asyncio
import time
import uuid
import pandas as pd  # Added for MovingAverageCrossoverStrategy

import config
from utils import logger

class BaseStrategy:
    """
    Base class for all trading strategies.
    """
    def __init__(self, symbol, order_queue=None, risk_manager=None):
        self.symbol = symbol
        self.order_queue = order_queue  # For sending orders (live or simulated)
        self.risk_manager = risk_manager  # For risk checks (live or simulated)
        self.last_signal_time = 0
        self.signal_cooldown = 0  # Can be overridden by subclasses

    def _generate_client_order_id(self):
        return f"{self.__class__.__name__.lower().replace('strategy', '')}_{uuid.uuid4()}"[:32]

    def create_order(self, signal_side, price, order_type, amount):
        """
        Helper function to create a standardized order dictionary.
        """
        return {
            'client_order_id': self._generate_client_order_id(),
            'symbol': self.symbol,
            'side': signal_side.lower(),
            'type': order_type,
            'amount': amount,
            'price': price,
            'timestamp': time.time()
        }

    async def send_order_to_queue(self, order):
        """
        Sends an order to the order_queue after risk validation (if applicable).
        This is primarily for async strategies using an asyncio.Queue.
        Returns True if order sent/validated, False otherwise.
        """
        if self.risk_manager:
            # Assuming risk_manager.validate_order can be async if needed by live components
            # For a synchronous backtester, a synchronous mock/stub would be used.
            is_valid = await self.risk_manager.validate_order(order) if asyncio.iscoroutinefunction(self.risk_manager.validate_order) \
                         else self.risk_manager.validate_order(order)
            if is_valid:
                logger.info(f"Order validated by Risk Manager: {order['client_order_id']}")
                if self.order_queue:
                    await self.order_queue.put(order)
                return True
            else:
                rejection_reason = getattr(self.risk_manager, 'last_rejection_reason', 'N/A')
                logger.warning(f"Order REJECTED by Risk Manager: {order['client_order_id']}, Reason: {rejection_reason}")
                return False
        elif self.order_queue:  # No risk manager, send directly if queue exists
            await self.order_queue.put(order)
            return True
        logger.debug("No risk manager or order queue to send order.")
        return False # No risk manager and no order queue, or risk check failed

    # For backtesting strategies, this is called by the backtester per bar
    def generate_signal(self, current_bar_data):
        """
        Generates a trading signal based on the current data bar.
        This method should be implemented by subclasses designed for backtesting.
        'current_bar_data' is typically a pandas Series representing the current bar.
        Should return an order dictionary or None.
        """
        raise NotImplementedError("Subclasses for backtesting should implement this method.")

    # For live, event-driven strategies
    async def run(self):
        """
        Main execution loop for live, event-driven strategies.
        """
        raise NotImplementedError("Live strategies should implement this method if they run continuously.")

class OrderFlowImbalanceStrategy(BaseStrategy):
    """A simple strategy based on Order Flow Imbalance (OFI). Designed for live trading."""

    def __init__(self, symbol, strategy_input_queue, order_queue, risk_manager, imbalance_threshold, order_size, cooldown_seconds=0.5):
        super().__init__(symbol=symbol, order_queue=order_queue, risk_manager=risk_manager)
        self.strategy_input_queue = strategy_input_queue  # Receives market state from DataHandler
        self.imbalance_threshold = imbalance_threshold
        self.order_size = order_size
        self.signal_cooldown = cooldown_seconds

    def _calculate_signal_from_market_state(self, market_state):
        """
        Core logic to calculate a signal from a market state.
        Returns a potential order dictionary or None.
        """
        imbalance = market_state.get('order_flow_imbalance')
        best_bid = market_state.get('best_bid')
        best_ask = market_state.get('best_ask')

        if imbalance is None or best_bid is None or best_ask is None:
            logger.warning(f"[{self.symbol}] Incomplete market state for OFI, skipping signal.")
            return None

        signal_side = None
        price = None
        order_type = None

        if imbalance > self.imbalance_threshold:
            signal_side = 'BUY'
            price = best_ask
            order_type = "buy-limit"
            logger.info(f"[{self.symbol}] OFI Signal: BUY (Imbalance={imbalance:.3f} > {self.imbalance_threshold}) at Price ~{price}")
        elif imbalance < (1 - self.imbalance_threshold):  # Assuming imbalance is 0-1
            signal_side = 'SELL'
            price = best_bid
            order_type = "sell-limit"
            logger.info(f"[{self.symbol}] OFI Signal: SELL (Imbalance={imbalance:.3f} < {1 - self.imbalance_threshold}) at Price ~{price}")

        if signal_side and price:
            return self.create_order(signal_side, price, order_type, self.order_size)
        return None

    async def run(self):
        """Continuously processes market state and generates signals for live trading."""
        logger.info(f"{self.__class__.__name__} starting for {self.symbol}...")
        while True:
            try:
                market_state = await self.strategy_input_queue.get()

                current_time = time.time()
                if current_time - self.last_signal_time < self.signal_cooldown:
                    self.strategy_input_queue.task_done()
                    continue

                order_to_place = self._calculate_signal_from_market_state(market_state)

                if order_to_place:
                    if await self.send_order_to_queue(order_to_place):
                        self.last_signal_time = current_time  # Reset cooldown timer

                self.strategy_input_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"{self.__class__.__name__} for {self.symbol} stopping...")
                break
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__} ({self.symbol}) loop: {e}", exc_info=True)
                await asyncio.sleep(1)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    A strategy based on the crossover of two moving averages.
    Designed for backtesting with OHLCV data (bar-by-bar).
    """
    def __init__(self, symbol, short_window, long_window, order_size=1.0, order_queue=None, risk_manager=None):
        super().__init__(symbol=symbol, order_queue=order_queue, risk_manager=risk_manager)
        if short_window >= long_window:
            raise ValueError("Short window must be less than long window for MA Crossover.")
        self.short_window = short_window
        self.long_window = long_window
        self.order_size = order_size  # Amount of asset to trade
        self.position = 0  # 0 = flat, 1 = long, -1 = short (simple internal tracking)
        # Stores only scalar Close prices, indexed by Timestamp
        self.close_price_history = pd.Series(dtype=float, name='Close')
        self.close_price_history.index = pd.to_datetime(self.close_price_history.index) # Ensure DatetimeIndex

    def generate_signal(self, current_bar_data):
        """
        Generates a trading signal for the current data bar.
        Args:
            current_bar_data (pd.Series): The current bar's data (OHLCV).
                                          Must have a 'Close' price and a datetime name/index.
        Returns:
            dict: An order dictionary if a signal is generated, otherwise None.
        """
        signal = None

        if not isinstance(current_bar_data, pd.Series) or 'Close' not in current_bar_data:
            logger.error(f"[{self.symbol}] Invalid current_bar_data format or missing 'Close' price for bar name: {getattr(current_bar_data, 'name', 'N/A')}.")
            return None

        try:
            current_timestamp = pd.to_datetime(current_bar_data.name)
        except Exception as e:
            logger.error(f"[{self.symbol}] Could not parse timestamp from current_bar_data.name: {current_bar_data.name}. Error: {e}")
            return None

        raw_close_price = current_bar_data['Close']
        current_close_price = None
        if isinstance(raw_close_price, pd.Series):
            try:
                current_close_price = raw_close_price.item() # Extract scalar if it's a single-item Series
            except ValueError:
                logger.warning(f"[{self.symbol}] 'Close' price at {current_timestamp} is a non-scalar Series: {raw_close_price}. Using NaN.")
                current_close_price = float('nan')
        else:
            current_close_price = float(raw_close_price) # Ensure it's a float

        if pd.isna(current_close_price):
            logger.warning(f"[{self.symbol}] NaN Close price at {current_timestamp}. Skipping bar.")
            return None

        self.close_price_history.loc[current_timestamp] = current_close_price
        self.close_price_history.sort_index(inplace=True)

        required_length = self.long_window + 5
        if len(self.close_price_history) > required_length:
            self.close_price_history = self.close_price_history.iloc[-required_length:]

        if len(self.close_price_history) < self.long_window + 1: # Need +1 for prev_long_mavg
            return None

        short_mavg_series = self.close_price_history.rolling(window=self.short_window).mean()
        long_mavg_series = self.close_price_history.rolling(window=self.long_window).mean()

        if len(short_mavg_series) < 2 or len(long_mavg_series) < 2:
            return None # Not enough data points in MA series for .iloc[-1] and .iloc[-2]

        ma_values = {
            'c_short': short_mavg_series.iloc[-1],
            'p_short': short_mavg_series.iloc[-2],
            'c_long': long_mavg_series.iloc[-1],
            'p_long': long_mavg_series.iloc[-2]
        }

        # Ensure all MA values are scalar
        for key, val in ma_values.items():
            if isinstance(val, pd.Series):
                try:
                    ma_values[key] = val.item()
                except ValueError:
                    logger.warning(f"[{self.symbol}] MA value '{key}' at {current_timestamp} is non-scalar Series: {val}. Using NaN.")
                    ma_values[key] = float('nan')
            elif not isinstance(val, (int, float)):
                 ma_values[key] = float(val) # Coerce to float if not already numeric

        current_short_mavg = ma_values['c_short']
        prev_short_mavg = ma_values['p_short']
        current_long_mavg = ma_values['c_long']
        prev_long_mavg = ma_values['p_long']

        if pd.isna(current_short_mavg) or pd.isna(current_long_mavg) or \
           pd.isna(prev_short_mavg) or pd.isna(prev_long_mavg):
            return None

        # Buy signal: short MA crosses above long MA
        if current_short_mavg > current_long_mavg and prev_short_mavg <= prev_long_mavg:
            if self.position <= 0:
                logger.info(f"[{self.symbol}] BUY Signal: Short MA ({current_short_mavg:.2f}) crossed above Long MA ({current_long_mavg:.2f}) at {current_close_price:.2f} on {current_timestamp.date()}. Position before: {self.position}")
                signal = self.create_order(signal_side='BUY', price=current_close_price, order_type='buy-market', amount=self.order_size)
                self.position = 1
        # Sell signal: short MA crosses below long MA
        elif current_short_mavg < current_long_mavg and prev_short_mavg >= prev_long_mavg:
            if self.position >= 0:
                logger.info(f"[{self.symbol}] SELL Signal: Short MA ({current_short_mavg:.2f}) crossed below Long MA ({current_long_mavg:.2f}) at {current_close_price:.2f} on {current_timestamp.date()}. Position before: {self.position}")
                signal = self.create_order(signal_side='SELL', price=current_close_price, order_type='sell-market', amount=self.order_size)
                self.position = -1

        return signal


# --- Example of how main.py might need to change for OFI strategy initialization ---
# (This is a comment and won't be part of the executed code here)
# Ensure config.py has: SYMBOL, OFI_IMBALANCE_THRESHOLD, OFI_ORDER_SIZE_ASSET, OFI_COOLDOWN_S
# 
# In main.py, where strategy is initialized:
# from strategy import OrderFlowImbalanceStrategy
# strategy = OrderFlowImbalanceStrategy(
#     symbol=config.TRADING_PAIR, # e.g., "btcusdt"
#     strategy_input_queue=strategy_input_queue, # from main.py
#     order_queue=order_request_queue,          # from main.py
#     risk_manager=risk_manager,                # from main.py
#     imbalance_threshold=config.OFI_IMBALANCE_THRESHOLD, # e.g., 0.6
#     order_size=config.OFI_ORDER_SIZE_ASSET,         # e.g., 0.001 (for BTC)
#     cooldown_seconds=config.OFI_COOLDOWN_S          # e.g., 0.5
# )
# 
# And ensure these new config variables are defined in config.py, for example:
# OFI_IMBALANCE_THRESHOLD = 0.6
# OFI_ORDER_SIZE_ASSET = 0.001
# OFI_COOLDOWN_S = 0.5
# (TRADING_PAIR is likely already there)