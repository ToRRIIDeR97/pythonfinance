import asyncio
import config
from utils import logger

class RiskManager:
    """Enforces risk limits before orders are placed."""

    def __init__(self, order_manager):
        self.order_manager = order_manager # Needs access to current positions/P&L
        self.max_position_btc = config.MAX_POSITION_BTC
        self.max_order_value_usd = config.MAX_ORDER_SIZE_USD # Requires price feed
        self.max_drawdown_percent = config.MAX_DRAWDOWN_PERCENT # Requires P&L tracking
        self.kill_switch_active = config.KILL_SWITCH_ACTIVE
        self.last_rejection_reason = ""

    async def validate_order(self, order):
        """Checks if a proposed order violates risk rules."""
        self.last_rejection_reason = ""

        if self.kill_switch_active:
            self.last_rejection_reason = "Kill switch active"
            return False

        # 1. Check Max Order Size (Value Check - needs price)
        # Simplification: Use order price if available, else skip check for market orders
        order_price = order.get('price')
        if order_price:
            order_value_usd = order['amount'] * order_price # Approximation
            if order_value_usd > self.max_order_value_usd:
                self.last_rejection_reason = f"Order value {order_value_usd:.2f} exceeds max {self.max_order_value_usd:.2f}"
                return False
        else:
             # For market orders, we might estimate using best ask/bid, or have a separate size limit
             pass # Skipping value check for market orders in this example

        # 2. Check Max Position Size
        current_position = self.order_manager.get_position(config.TRADING_PAIR)
        proposed_position = current_position
        if order['side'] == 'buy':
            proposed_position += order['amount']
        else: # sell
            proposed_position -= order['amount']

        if abs(proposed_position) > self.max_position_btc:
            self.last_rejection_reason = f"Proposed position {proposed_position:.4f} exceeds limit {self.max_position_btc:.4f}"
            return False

        # 3. Check Drawdown (Needs P&L - Simplistic check here)
        # Real drawdown check is more complex (peak equity vs current)
        # current_pnl = self.order_manager.get_total_pnl() # Needs implementation
        # if current_pnl < - (self.max_drawdown_percent / 100.0) * INITIAL_CAPITAL: # Need initial capital
        #    self.last_rejection_reason = "Max drawdown limit reached"
        #    self.trigger_kill_switch("Max drawdown")
        #    return False

        # Add more checks: Rate limits (internal), price collars, etc.

        return True # Order is valid

    async def trigger_kill_switch(self, reason=""):
        """Activates the kill switch, halting trading and cancelling orders."""
        if not self.kill_switch_active:
            self.kill_switch_active = True
            logger.critical(f"!!! KILL SWITCH ACTIVATED !!! Reason: {reason}")
            # Signal OrderManager to cancel all open orders
            await self.order_manager.cancel_all_orders("Kill switch activated")
            # Potentially add logic to flatten positions (market order everything) - DANGEROUS

    def reset_kill_switch(self):
        """Manually resets the kill switch (use with caution)."""
        logger.warning("Resetting kill switch manually.")
        self.kill_switch_active = False

    # Add methods for monitoring VaR, exposure, etc. as needed