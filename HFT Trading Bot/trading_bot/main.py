import asyncio
import signal

import config
from utils import logger
from htx_client import HtxClient
from data_handler import DataHandler
from strategy import OrderFlowImbalanceStrategy
from order_manager import OrderManager
from risk_manager import RiskManager

# --- Global Queues ---
# Use asyncio queues for safe communication between async tasks
market_data_queue = asyncio.Queue() # Raw data from WS to DataHandler
strategy_input_queue = asyncio.Queue() # Processed data from DataHandler to Strategy
order_request_queue = asyncio.Queue() # Orders from Strategy to OrderManager (via RiskManager)
execution_report_queue = asyncio.Queue() # Execution reports from WS/REST to OrderManager

# --- Global Components ---
htx_client = None
data_handler = None
strategy = None
order_manager = None
risk_manager = None
tasks = [] # To keep track of running asyncio tasks

async def main():
    global htx_client, data_handler, strategy, order_manager, risk_manager, tasks
    logger.info("Initializing Trading Bot System...")

    # 1. Initialize Components
    htx_client = HtxClient(
        api_key=config.HTX_API_KEY,
        secret_key=config.HTX_SECRET_KEY,
        rest_url=config.HTX_REST_URL,
        ws_url=config.HTX_WS_URL,
        ws_auth_url=config.HTX_WS_AUTH_URL,
        data_queue=market_data_queue,
        execution_queue=execution_report_queue
    )
    order_manager = OrderManager(execution_report_queue, htx_client)
    risk_manager = RiskManager(order_manager)
    data_handler = DataHandler(market_data_queue, strategy_input_queue)
    strategy = OrderFlowImbalanceStrategy(
        symbol=config.TRADING_PAIR,
        strategy_input_queue=strategy_input_queue,
        order_queue=order_request_queue,
        risk_manager=risk_manager,
        imbalance_threshold=config.OFI_IMBALANCE_THRESHOLD,
        order_size=config.OFI_ORDER_SIZE_ASSET,
        cooldown_seconds=config.OFI_COOLDOWN_S
    )

    # 2. Create asyncio Tasks for each component's run loop
    tasks.append(asyncio.create_task(htx_client.connect_public_ws()))
    tasks.append(asyncio.create_task(htx_client.connect_auth_ws())) # Connects to authenticated WS
    tasks.append(asyncio.create_task(data_handler.run()))
    tasks.append(asyncio.create_task(strategy.run()))
    tasks.append(asyncio.create_task(order_manager.run()))

    # 3. Task for handling order requests from the strategy queue
    async def order_router():
        logger.info("Order Router starting...")
        while True:
            try:
                order = await order_request_queue.get()
                logger.info(f"Order Router received order request: {order.get('client_order_id')}")
                # Order already validated by Risk Manager in Strategy, send to Order Manager
                await order_manager.place_order(order)
                order_request_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Order Router stopping...")
                break
            except Exception as e:
                logger.error(f"Error in Order Router loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    tasks.append(asyncio.create_task(order_router()))


    # 4. Keep the main loop alive, or wait for tasks to complete
    logger.info("Trading Bot System started. Press Ctrl+C to stop.")
    # Wait for any task to finish (which might indicate an error or shutdown)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in done:
        try:
            task.result() # Raise exception if task finished with error
        except Exception as e:
            logger.critical(f"A critical task finished unexpectedly: {e}", exc_info=True)

    logger.warning("One or more tasks finished. Initiating shutdown...")
    await shutdown()


async def shutdown(signame=None):
    global tasks, htx_client
    if signame:
        logger.info(f"Received signal {signame}. Shutting down gracefully...")

    # 1. Signal components to stop (e.g., by cancelling tasks)
    for task in tasks:
        if not task.done():
            task.cancel()

    # 2. Cancel all open orders (optional, depends on strategy/risk policy)
    if order_manager:
        try:
           logger.info("Attempting to cancel open orders on shutdown...")
           await order_manager.cancel_all_orders(f"Shutdown signal {signame}")
        except Exception as e:
           logger.error(f"Error cancelling orders during shutdown: {e}")


    # 3. Wait for tasks to actually cancel
    await asyncio.gather(*tasks, return_exceptions=True) # Allow tasks to finish cleanup

    # 4. Close network connections
    if htx_client:
        await htx_client.close()

    logger.info("Trading Bot System shut down complete.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Add signal handlers for graceful shutdown
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda signame=signame: asyncio.create_task(shutdown(signame))
        )

    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
         logger.info("Main loop cancelled during shutdown.")
    finally:
        # Final cleanup if loop exits unexpectedly
        if not loop.is_closed():
             # Run pending cleanup tasks if shutdown wasn't fully completed
             pending = asyncio.all_tasks(loop=loop)
             pending.remove(asyncio.current_task(loop=loop)) # Don't wait for self
             if pending:
                  logger.info(f"Running {len(pending)} pending cleanup tasks...")
                  loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
             loop.close()
             logger.info("Event loop closed.")