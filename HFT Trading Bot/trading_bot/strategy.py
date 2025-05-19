import asyncio
import time
import uuid

import config
from utils import logger

class OrderFlowImbalanceStrategy:
    """A simple strategy based on Order Flow Imbalance (OFI)."""

    def __init__(self, strategy_queue, order_queue, risk_manager):
        self.strategy_queue = strategy_queue # Receives market state from DataHandler
        self.order_queue = order_queue     # Sends potential orders to OrderManager/RiskManager
        self.risk_manager = risk_manager
        self.last_signal_time = 0
        self.signal_cooldown = 0.5 # Cooldown in seconds to avoid rapid signals

    async def run(self):
        """Continuously processes market state and generates signals."""
        logger.info("Strategy starting...")
        while True:
            try:
                market_state = await self.strategy_queue.get()
                # logger.debug(f"Strategy received market state: {market_state}")

                current_time = time.time()
                if current_time - self.last_signal_time < self.signal_cooldown:
                    self.strategy_queue.task_done()
                    continue # Skip if within cooldown

                imbalance = market_state.get('order_flow_imbalance')
                best_bid = market_state.get('best_bid')
                best_ask = market_state.get('best_ask')

                if imbalance is None or best_bid is None or best_ask is None:
                    logger.warning("Incomplete market state received, skipping signal generation.")
                    self.strategy_queue.task_done()
                    continue

                signal = None
                price = None
                order_type = None

                # --- Basic OFI Logic ---
                if imbalance > config.IMBALANCE_THRESHOLD:
                    # High imbalance -> suggests upward pressure -> Place aggressive buy
                    signal = 'BUY'
                    price = best_ask # Cross the spread (taker) or place at best ask (maker)
                    order_type = "buy-limit" # Example: place limit order at best ask
                    logger.info(f"Signal: BUY (Imbalance={imbalance:.3f} > {config.IMBALANCE_THRESHOLD}) at Price ~{price}")

                elif imbalance < (1 - config.IMBALANCE_THRESHOLD):
                     # Low imbalance -> suggests downward pressure -> Place aggressive sell
                    signal = 'SELL'
                    price = best_bid # Cross the spread (taker) or place at best bid (maker)
                    order_type = "sell-limit" # Example: place limit order at best bid
                    logger.info(f"Signal: SELL (Imbalance={imbalance:.3f} < {1-config.IMBALANCE_THRESHOLD}) at Price ~{price}")

                # --- Generate Order ---
                if signal and price:
                    client_order_id = f"ofi_{uuid.uuid4()}"[:32] # Unique ID for tracking
                    order = {
                        'client_order_id': client_order_id,
                        'symbol': config.TRADING_PAIR,
                        'side': signal.lower(),
                        'type': order_type, # e.g., 'buy-limit', 'sell-market'
                        'amount': config.ORDER_SIZE_BTC,
                        'price': price,
                        'timestamp': time.time()
                    }

                    # Send order to Risk Manager for approval
                    if await self.risk_manager.validate_order(order):
                       logger.info(f"Order validated by Risk Manager: {client_order_id}")
                       await self.order_queue.put(order)
                       self.last_signal_time = current_time # Reset cooldown timer
                    else:
                       logger.warning(f"Order REJECTED by Risk Manager: {client_order_id}, Reason: {self.risk_manager.last_rejection_reason}")

                self.strategy_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Strategy stopping...")
                break
            except Exception as e:
                logger.error(f"Error in Strategy loop: {e}", exc_info=True)
                await asyncio.sleep(1) # Avoid tight loop