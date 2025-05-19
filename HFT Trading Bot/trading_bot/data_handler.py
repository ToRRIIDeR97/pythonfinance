import asyncio
from collections import deque
import time

import config
from utils import logger

class DataHandler:
    """Processes raw market data from the queue and maintains market state."""

    def __init__(self, data_queue, strategy_queue):
        self.data_queue = data_queue
        self.strategy_queue = strategy_queue # Queue to send processed data/events to Strategy
        self.order_book = {'bids': {}, 'asks': {}} # Price -> Size
        self.last_trade_price = None
        self.last_update_time = None
        # Optional: For rolling calculations like moving averages or volatility
        self.trade_history = deque(maxlen=100) # Store last 100 trades

    def _update_order_book(self, data):
        """Updates the local order book based on a depth snapshot or update."""
        # ** PARSE HTX's DEPTH MESSAGE FORMAT CAREFULLY **
        # This assumes a snapshot format like {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        # Incremental updates require more complex logic (handling additions, changes, deletions)
        if 'tick' in data and 'bids' in data['tick'] and 'asks' in data['tick']:
            snapshot = data['tick']
            # Use dicts for faster lookups/updates compared to sorted lists for full book
            self.order_book['bids'] = {float(price): float(size) for price, size in snapshot['bids']}
            self.order_book['asks'] = {float(price): float(size) for price, size in snapshot['asks']}
            self.last_update_time = data.get('ts', time.time() * 1000) # Use exchange timestamp if available
            # logger.debug(f"Order book updated. Best Bid: {self.get_best_bid()}, Best Ask: {self.get_best_ask()}")
            return True # Indicate book was updated
        else:
            logger.warning(f"Received unexpected depth data format: {data}")
            return False

    def _update_trades(self, data):
        """Processes incoming trade messages."""
        # ** PARSE HTX's TRADE MESSAGE FORMAT **
        if 'tick' in data and 'data' in data['tick']:
            for trade in data['tick']['data']:
                trade_price = float(trade['price'])
                trade_size = float(trade['amount'])
                trade_time = trade.get('ts', time.time() * 1000)
                self.last_trade_price = trade_price
                self.trade_history.append({'price': trade_price, 'size': trade_size, 'time': trade_time})
                # logger.debug(f"New Trade: Price={trade_price}, Size={trade_size}")
            return True # Indicate trades were processed
        else:
            logger.warning(f"Received unexpected trade data format: {data}")
            return False


    def get_best_bid(self):
        """Returns the highest bid price."""
        if not self.order_book['bids']:
            return None
        return max(self.order_book['bids'].keys())

    def get_best_ask(self):
        """Returns the lowest ask price."""
        if not self.order_book['asks']:
            return None
        return min(self.order_book['asks'].keys())

    def get_mid_price(self):
        """Calculates the mid-price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None

    def calculate_order_flow_imbalance(self, levels=5):
        """Calculates simple order flow imbalance (OFI)."""
        # Sort bids descending, asks ascending
        sorted_bids = sorted(self.order_book['bids'].items(), key=lambda item: item[0], reverse=True)
        sorted_asks = sorted(self.order_book['asks'].items(), key=lambda item: item[0])

        bid_volume = sum(size for _, size in sorted_bids[:levels])
        ask_volume = sum(size for _, size in sorted_asks[:levels])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.5 # Neutral if no volume

        imbalance = bid_volume / total_volume
        # logger.debug(f"OFI Calculation: Bids={bid_volume:.4f}, Asks={ask_volume:.4f}, Total={total_volume:.4f}, Imbalance={imbalance:.4f}")
        return imbalance


    async def run(self):
        """Continuously processes messages from the data queue."""
        logger.info("DataHandler starting...")
        while True:
            try:
                message = await self.data_queue.get()
                message_type = message.get('type')
                data = message.get('data')
                processed = False

                if message_type == 'depth':
                   processed = self._update_order_book(data)
                elif message_type == 'trade':
                   processed = self._update_trades(data)
                else:
                    logger.warning(f"DataHandler received unknown message type: {message_type}")

                # If data was processed successfully, create a market state snapshot
                # and send it to the strategy
                if processed and self.order_book['bids'] and self.order_book['asks']:
                    market_state = {
                        'timestamp': self.last_update_time or time.time() * 1000,
                        'best_bid': self.get_best_bid(),
                        'best_ask': self.get_best_ask(),
                        'mid_price': self.get_mid_price(),
                        'last_trade_price': self.last_trade_price,
                        'order_flow_imbalance': self.calculate_order_flow_imbalance()
                        # Add other relevant state: spread, recent volatility, book depth etc.
                    }
                    await self.strategy_queue.put(market_state)

                self.data_queue.task_done()

            except asyncio.CancelledError:
                logger.info("DataHandler stopping...")
                break
            except Exception as e:
                logger.error(f"Error in DataHandler loop: {e}", exc_info=True)
                # Avoid tight loop on persistent error
                await asyncio.sleep(1)