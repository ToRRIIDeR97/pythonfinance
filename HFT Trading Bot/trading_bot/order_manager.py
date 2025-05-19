import asyncio
import time
from collections import defaultdict

import config
from utils import logger

class OrderManager:
    """Tracks open orders, positions, and calculates P&L."""

    def __init__(self, execution_queue, htx_client):
        self.execution_queue = execution_queue # Receives execution reports from HtxClient
        self.htx_client = htx_client # To send orders/cancellations
        self.open_orders = {} # client_order_id -> order details
        self.positions = defaultdict(float) # symbol -> size (positive for long, negative for short)
        self.avg_entry_price = defaultdict(float)
        self.realized_pnl = defaultdict(float)
        self.unrealized_pnl = defaultdict(float) # Needs market price updates
        self.trade_log = [] # Log of filled trades

    async def place_order(self, order_details):
        """Sends an order to the exchange via HtxClient."""
        client_order_id = order_details['client_order_id']
        symbol = order_details['symbol']
        order_type = order_details['type']
        amount = order_details['amount']
        price = order_details.get('price') # May be None for market orders

        # Store order intent locally BEFORE sending
        self.open_orders[client_order_id] = {**order_details, 'status': 'pending_send', 'exchange_id': None}
        logger.info(f"Order Manager storing pending order: {client_order_id}")

        exchange_order_id = await self.htx_client.place_order(
            symbol=symbol,
            order_type=order_type,
            amount=amount,
            price=price,
            client_order_id=client_order_id
        )

        # Note: Execution queue might update status based on API response/WS message
        # We might not need to update status here if WS confirmation is reliable
        if exchange_order_id:
             if client_order_id in self.open_orders:
                 self.open_orders[client_order_id]['status'] = 'submitted' # Or 'ack' if confirmed by API response
                 self.open_orders[client_order_id]['exchange_id'] = exchange_order_id
                 logger.info(f"Order {client_order_id} submitted to exchange, ID: {exchange_order_id}")
        else:
             # Handle placement failure - order might already be marked as error via queue
             if client_order_id in self.open_orders:
                 logger.error(f"Order {client_order_id} failed to place (API error). Removing from open orders.")
                 del self.open_orders[client_order_id] # Or mark as failed explicitly


    async def handle_execution_report(self, report):
        """Processes execution updates from the execution_queue."""
        report_type = report.get('type')

        if report_type == 'order_ack':
            # Confirmation that order was received by exchange
            client_id = report.get('client_order_id')
            exchange_id = report.get('exchange_order_id')
            if client_id in self.open_orders:
                self.open_orders[client_id]['status'] = 'acknowledged' # Or 'submitted'
                self.open_orders[client_id]['exchange_id'] = exchange_id
                logger.info(f"Order ACK received: ClientID={client_id}, ExchangeID={exchange_id}")

        elif report_type == 'order_error':
            # Order placement failed or rejected
            client_id = report.get('client_order_id')
            error_msg = report.get('error_message', 'Unknown error')
            if client_id in self.open_orders:
                logger.error(f"Order ERROR received: ClientID={client_id}. Reason: {error_msg}. Removing order.")
                # Mark as error or remove
                if client_id in self.open_orders:
                    del self.open_orders[client_id]
            else:
                 logger.warning(f"Received error for unknown/closed ClientID: {client_id}")


        elif report_type == 'execution':
            # A trade fill or order status change from WebSocket
            # ** PARSE HTX's EXECUTION REPORT FORMAT CAREFULLY **
            # Example structure - adapt based on actual format
            exec_data = report.get('data', {})
            client_id = exec_data.get('clientOrderId') # Check actual field name
            exchange_id = exec_data.get('orderId')
            symbol = exec_data.get('symbol')
            order_status = exec_data.get('orderStatus') # e.g., 'filled', 'partial-filled', 'canceled'

            # Use client_id if available, otherwise try to find by exchange_id
            order_ref = None
            if client_id and client_id in self.open_orders:
                order_ref = self.open_orders[client_id]
            elif exchange_id:
                # Find by exchange ID (less efficient)
                found_client_id = next((cid for cid, o in self.open_orders.items() if o.get('exchange_id') == exchange_id), None)
                if found_client_id:
                    order_ref = self.open_orders[found_client_id]
                    client_id = found_client_id # Set client_id for consistency

            if not order_ref:
                logger.warning(f"Received execution report for unknown order: ClientID={client_id}, ExchangeID={exchange_id}, Status={order_status}")
                return

            logger.info(f"Execution Report: ClientID={client_id}, ExchID={exchange_id}, Status={order_status}")
            order_ref['status'] = order_status

            # Handle Fills
            if order_status in ['partial-filled', 'filled']:
                fill_price = float(exec_data.get('tradePrice', 0)) # Price of this specific fill
                fill_amount = float(exec_data.get('tradeVolume', 0)) # Amount of this specific fill
                # total_filled = float(exec_data.get('filledAmount', 0)) # Total filled amount for the order
                # fee = float(exec_data.get('tradeFee', 0)) # Fee for this fill
                # fee_asset = exec_data.get('feeCurrency')

                if fill_amount > 0 and fill_price > 0:
                    logger.info(f"FILL DETECTED: ClientID={client_id}, Amount={fill_amount}, Price={fill_price}, Symbol={symbol}")
                    self.update_position(symbol, order_ref['side'], fill_amount, fill_price)
                    # Log the trade
                    self.trade_log.append({
                         'timestamp': exec_data.get('tradeTime', time.time()*1000),
                         'client_order_id': client_id,
                         'exchange_order_id': exchange_id,
                         'symbol': symbol,
                         'side': order_ref['side'],
                         'amount': fill_amount,
                         'price': fill_price,
                         # 'fee': fee,
                         # 'fee_asset': fee_asset
                    })
                    # Update order's filled amount if needed (complex if multiple partial fills)

            # Remove order if fully filled or canceled/rejected
            if order_status in ['filled', 'canceled', 'rejected', 'expired']: # Check exact status names
                 logger.info(f"Order {client_id} is now closed (Status: {order_status}). Removing from open orders.")
                 if client_id in self.open_orders:
                     del self.open_orders[client_id]

        else:
            logger.warning(f"OrderManager received unknown report type: {report_type}")


    def update_position(self, symbol, side, amount, price):
        """Updates position size and average entry price."""
        current_position = self.positions[symbol]
        current_avg_price = self.avg_entry_price[symbol]
        trade_value = amount * price

        logger.debug(f"Before update: Pos={current_position:.4f}, AvgPx={current_avg_price:.4f}")

        if side == 'buy':
            new_position = current_position + amount
            if new_position == 0: # Closing a short position
                realized = (current_avg_price - price) * amount # PNL per unit * amount
                self.realized_pnl[symbol] += realized
                self.avg_entry_price[symbol] = 0 # Reset avg price when flat
                logger.info(f"Closed short {symbol}. Amount={amount}, Price={price}, Realized PNL: {realized:.4f}")
            elif current_position < 0: # Reducing short position
                 realized = (current_avg_price - price) * min(amount, abs(current_position))
                 self.realized_pnl[symbol] += realized
                 # Avg price remains the same if still short
                 logger.info(f"Reduced short {symbol}. Amount={amount}, Price={price}, Realized PNL: {realized:.4f}")
            else: # Adding to long or opening long
                new_avg_price = ((current_position * current_avg_price) + trade_value) / new_position if new_position != 0 else 0
                self.avg_entry_price[symbol] = new_avg_price
                logger.info(f"Opened/Increased long {symbol}. Amount={amount}, Price={price}")

        else: # side == 'sell'
            new_position = current_position - amount
            if new_position == 0: # Closing a long position
                realized = (price - current_avg_price) * amount
                self.realized_pnl[symbol] += realized
                self.avg_entry_price[symbol] = 0
                logger.info(f"Closed long {symbol}. Amount={amount}, Price={price}, Realized PNL: {realized:.4f}")
            elif current_position > 0: # Reducing long position
                realized = (price - current_avg_price) * min(amount, current_position)
                self.realized_pnl[symbol] += realized
                # Avg price remains the same if still long
                logger.info(f"Reduced long {symbol}. Amount={amount}, Price={price}, Realized PNL: {realized:.4f}")
            else: # Adding to short or opening short
                # Flip amount sign for calculation if opening short
                new_avg_price = ((abs(current_position) * current_avg_price) + trade_value) / abs(new_position) if new_position != 0 else 0
                self.avg_entry_price[symbol] = new_avg_price
                logger.info(f"Opened/Increased short {symbol}. Amount={amount}, Price={price}")

        self.positions[symbol] = new_position
        logger.info(f"Position Updated: {symbol} = {self.positions[symbol]:.4f}, Avg Entry: {self.avg_entry_price[symbol]:.4f}")


    def get_position(self, symbol):
        """Returns the current position size for a symbol."""
        return self.positions.get(symbol, 0.0)

    def get_all_positions(self):
        """Returns all current positions."""
        return self.positions

    def get_open_orders(self):
         """Returns a copy of the open orders dict."""
         return self.open_orders.copy()

    async def cancel_all_orders(self, reason=""):
        """Attempts to cancel all open orders."""
        logger.warning(f"Attempting to cancel all open orders. Reason: {reason}")
        open_orders_copy = list(self.open_orders.items()) # Iterate over a copy
        cancelled_count = 0
        for client_id, order in open_orders_copy:
            if order.get('status') in ['pending_send', 'submitted', 'acknowledged', 'partial-filled']: # Cancellable states
                exchange_id = order.get('exchange_id')
                if exchange_id:
                    logger.info(f"Requesting cancellation for ExchID: {exchange_id} (ClientID: {client_id})")
                    success = await self.htx_client.cancel_order(exchange_id)
                    if success:
                        cancelled_count += 1
                        # Mark as pending cancel, wait for execution report to confirm
                        self.open_orders[client_id]['status'] = 'pending_cancel'
                    else:
                        logger.error(f"Failed to submit cancellation request for ExchID: {exchange_id}")
                else:
                     logger.warning(f"Cannot cancel order {client_id}, no exchange ID known yet.")
            else:
                 logger.debug(f"Skipping cancellation for order {client_id}, status is {order.get('status')}")
        logger.info(f"Cancellation requests submitted for {cancelled_count} orders.")


    async def run(self):
        """Continuously processes execution reports from the queue."""
        logger.info("OrderManager starting...")
        while True:
            try:
                report = await self.execution_queue.get()
                await self.handle_execution_report(report)
                self.execution_queue.task_done()
            except asyncio.CancelledError:
                logger.info("OrderManager stopping...")
                # Optionally cancel remaining orders on shutdown
                # await self.cancel_all_orders("System shutdown")
                break
            except Exception as e:
                logger.error(f"Error in OrderManager loop: {e}", exc_info=True)
                await asyncio.sleep(1)