import asyncio
import json
import websockets
import aiohttp
import time
from urllib.parse import urlparse

import config
from utils import logger, create_signature_v2 # Import the example signature function

class HtxClient:
    """Handles communication with the HTX API (REST & WebSocket)."""

    def __init__(self, api_key, secret_key, rest_url, ws_url, ws_auth_url, data_queue, execution_queue):
        self.api_key = api_key
        self.secret_key = secret_key
        self.rest_url = rest_url
        self.ws_url = ws_url # Public data WebSocket
        self.ws_auth_url = ws_auth_url # Authenticated WebSocket (orders/accounts)
        self._session = None # aiohttp client session
        self.data_queue = data_queue # Queue to send market data to DataHandler
        self.execution_queue = execution_queue # Queue to send order updates to OrderManager
        self.ws_public = None
        self.ws_auth = None
        self.rest_host = urlparse(rest_url).netloc
        self._rate_limiter = asyncio.Semaphore(10) # Basic example: limit to 10 concurrent requests
        self._last_request_time = 0
        self._request_interval = 0.1 # Minimum 100ms between requests (adjust based on limits)


    async def _ensure_session(self):
        """Creates an aiohttp ClientSession if it doesn't exist."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.API_TIMEOUT_SECONDS))
            logger.info("aiohttp session created.")

    async def close_session(self):
        """Closes the aiohttp ClientSession."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("aiohttp session closed.")

    async def _request(self, method, path, params=None, data=None, signed=False):
        """Makes an authenticated or public REST API request."""
        await self._ensure_session()
        url = f"{self.rest_url}{path}"
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        query_params = params if params else {}

        # Basic rate limiting (adapt based on actual HTX rules)
        async with self._rate_limiter:
            now = time.monotonic()
            wait_time = self._request_interval - (now - self._last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

            if signed:
                if method == 'GET':
                    # Signature includes query params
                    signed_params = create_signature_v2(self.api_key, self.secret_key, method, self.rest_host, path, query_params)
                    query_params = signed_params
                elif method == 'POST':
                    # Signature usually doesn't include JSON body but check HTX docs
                    # For POST, signature params might go in query string OR body
                    # This example assumes they go in query string
                    base_params_for_sig = {} # May need specific params for POST sig
                    signed_params = create_signature_v2(self.api_key, self.secret_key, method, self.rest_host, path, base_params_for_sig)
                    query_params = signed_params # Add signature to query string

            logger.debug(f"Request: {method} {url} Params: {query_params} Data: {data}")
            try:
                async with self._session.request(method, url, params=query_params, json=data, headers=headers) as response:
                    response_text = await response.text()
                    logger.debug(f"Response Status: {response.status}, Body: {response_text[:500]}") # Log truncated response
                    response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP Error: {e.status} {e.message} - URL: {e.request_info.url}")
                # Consider specific handling for 429 Rate Limit Exceeded
                if e.status == 429:
                    logger.warning("Rate limit hit! Consider increasing request interval or using backoff.")
                    # Implement backoff logic here if needed
                return None # Or raise a custom exception
            except asyncio.TimeoutError:
                logger.error(f"Request timed out: {method} {url}")
                return None
            except Exception as e:
                logger.error(f"Request failed: {method} {url} - Error: {e}", exc_info=True)
                return None

    async def get_accounts(self):
        """Fetches account information (Example)."""
        # ** FIND THE CORRECT PATH IN HTX DOCS **
        path = "/v1/account/accounts"
        return await self._request('GET', path, signed=True)

    async def place_order(self, symbol, order_type, amount, price=None, client_order_id=None):
        """Places an order (Example)."""
        # ** FIND THE CORRECT PATH AND PARAMETERS IN HTX DOCS **
        path = "/v1/order/orders/place"
        data = {
            "account-id": "YOUR_ACCOUNT_ID", # Fetch this first or have it configured
            "symbol": symbol,
            "type": order_type, # e.g., "buy-limit", "sell-market"
            "amount": str(amount),
        }
        if price and "limit" in order_type:
            data["price"] = str(price)
        if client_order_id:
            data["client-order-id"] = client_order_id

        # Ensure account-id is fetched/set correctly before calling
        if "YOUR_ACCOUNT_ID" in data.values():
             logger.error("Account ID not set. Fetch accounts first.")
             # You might fetch accounts here if needed, or ensure it's pre-configured
             # accounts = await self.get_accounts()
             # if accounts and accounts.get('status') == 'ok' and accounts.get('data'):
             #     # Assuming spot account type, check docs
             #     spot_account = next((acc for acc in accounts['data'] if acc['type'] == 'spot'), None)
             #     if spot_account:
             #          data["account-id"] = spot_account['id']
             #     else:
             #          logger.error("Spot account ID not found.")
             #          return None
             # else:
             #     logger.error("Failed to fetch accounts or no data returned.")
             #     return None
             return None # Temporary return until account ID logic is solid


        logger.info(f"Placing order: {data}")
        response = await self._request('POST', path, data=data, signed=True)
        if response and response.get('status') == 'ok':
            order_id = response.get('data')
            logger.info(f"Order placed successfully. Exchange Order ID: {order_id}")
            # Optionally put confirmation on execution queue
            await self.execution_queue.put({
                'type': 'order_ack',
                'client_order_id': client_order_id,
                'exchange_order_id': order_id,
                'symbol': symbol,
                'status': 'submitted' # Or 'ack'
            })
            return order_id
        else:
            error_code = response.get('err-code', 'N/A') if response else 'N/A'
            error_msg = response.get('err-msg', 'Request failed or got unexpected response') if response else 'Request failed'
            logger.error(f"Failed to place order. Code: {error_code}, Msg: {error_msg}, Response: {response}")
            # Put error on execution queue
            await self.execution_queue.put({
                'type': 'order_error',
                'client_order_id': client_order_id,
                'symbol': symbol,
                'error_message': f"Code: {error_code}, Msg: {error_msg}"
            })
            return None

    async def cancel_order(self, order_id):
        """Cancels an order by exchange ID (Example)."""
        # ** FIND THE CORRECT PATH IN HTX DOCS **
        path = f"/v1/order/orders/{order_id}/submitcancel"
        logger.info(f"Cancelling order: {order_id}")
        response = await self._request('POST', path, signed=True)
        # Add handling for response and execution queue update
        if response and response.get('status') == 'ok':
             logger.info(f"Cancel request for order {order_id} submitted successfully. Final status pending.")
             # HTX might return the order ID again or just confirmation
             # You still need execution reports to confirm cancellation
             return response.get('data') # Or True
        else:
            error_code = response.get('err-code', 'N/A') if response else 'N/A'
            error_msg = response.get('err-msg', 'Request failed or got unexpected response') if response else 'Request failed'
            logger.error(f"Failed to submit cancel for order {order_id}. Code: {error_code}, Msg: {error_msg}")
            return False


    # --- WebSocket Methods ---

    async def connect_public_ws(self):
        """Connects to the public WebSocket for market data."""
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws_public = ws
                    logger.info(f"Connected to public WebSocket: {self.ws_url}")
                    # Subscribe to necessary market data streams
                    await self.subscribe_market_data(ws, config.TRADING_PAIR)

                    async for message in ws:
                        try:
                            decompressed_message = message # HTX might use gzip, check docs
                            data = json.loads(decompressed_message)

                            # Handle ping/pong
                            if 'ping' in data:
                                await ws.send(json.dumps({'pong': data['ping']}))
                                logger.debug("Sent WebSocket Pong")
                                continue
                            if 'action' in data and data['action'] == 'ping': # Another possible ping format
                                await ws.send(json.dumps({
                                    'action': 'pong',
                                    'data': {'ts': data['data']['ts']}
                                }))
                                logger.debug("Sent WebSocket Pong (action format)")
                                continue

                            # Handle subscription confirmations if necessary
                            if 'subbed' in data or ('op' in data and data['op'] == 'sub'):
                                logger.info(f"Subscription confirmed: {data}")
                                continue

                            # Put relevant market data onto the queue
                            # ** PARSE ACCORDING TO HTX's ACTUAL FORMAT **
                            if 'ch' in data and 'depth.step' in data['ch']: # Example depth channel
                                await self.data_queue.put({'type': 'depth', 'data': data})
                            elif 'ch' in data and 'trade.detail' in data['ch']: # Example trade channel
                                await self.data_queue.put({'type': 'trade', 'data': data})
                            else:
                                logger.debug(f"Received unhandled public WS message: {data}")

                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON from public WS: {message}")
                        except Exception as e:
                            logger.error(f"Error processing public WS message: {e}", exc_info=True)

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, asyncio.TimeoutError) as e:
                logger.warning(f"Public WebSocket connection lost: {e}. Reconnecting in 5 seconds...")
                self.ws_public = None
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in public WebSocket connection: {e}. Reconnecting in 15 seconds...", exc_info=True)
                self.ws_public = None
                await asyncio.sleep(15)


    async def subscribe_market_data(self, ws, symbol):
        """Subscribes to market data streams (Example)."""
        # ** FIND THE CORRECT SUBSCRIPTION FORMATS IN HTX DOCS **
        sub_depth_msg = {
            "sub": f"market.{symbol}.depth.step0", # step0 for full order book usually
            "id": f"depth_{symbol}_{int(time.time())}"
        }
        sub_trade_msg = {
            "sub": f"market.{symbol}.trade.detail",
            "id": f"trade_{symbol}_{int(time.time())}"
        }
        await ws.send(json.dumps(sub_depth_msg))
        logger.info(f"Subscribed to depth: {sub_depth_msg['sub']}")
        await ws.send(json.dumps(sub_trade_msg))
        logger.info(f"Subscribed to trades: {sub_trade_msg['sub']}")


    async def connect_auth_ws(self):
        """Connects to the authenticated WebSocket for orders/executions."""
        # ** HTX AUTH WS IS COMPLEX - CAREFULLY FOLLOW DOCS **
        # Often involves sending an authentication request first after connecting.
        while True:
            try:
                 # Example V2 Auth Process (Check official docs!)
                parsed_url = urlparse(self.ws_auth_url)
                host = parsed_url.netloc
                path = parsed_url.path

                # 1. Prepare auth parameters
                auth_params = {
                    'action': 'req',
                    'ch': 'auth',
                    'params': create_signature_v2( # Reuse signing logic
                        self.api_key, self.secret_key,
                        'GET', # Usually GET for WS auth signature
                        host, path,
                        {'authType': 'api'} # Add any specific params needed by HTX WS auth
                    )
                }
                # Remove the Signature itself from the inner params if it was added by create_signature_v2
                # The signature should be top-level in the 'params' dict's value for the outer JSON
                # This depends heavily on HTX's exact format!
                signature_value = auth_params['params'].pop('Signature', None)
                if signature_value:
                    auth_params['params']['signature'] = signature_value # Renaming might be needed


                async with websockets.connect(self.ws_auth_url) as ws:
                    self.ws_auth = ws
                    logger.info(f"Connected to authenticated WebSocket: {self.ws_auth_url}")

                    # 2. Send Authentication Request
                    await ws.send(json.dumps(auth_params))
                    logger.info("Sent authentication request to WS.")

                    # 3. Wait for Auth Response & Subscribe
                    auth_success = False
                    while not auth_success:
                        response = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(response)
                        logger.debug(f"Auth WS response: {data}")
                        if data.get('action') == 'req' and data.get('ch') == 'auth' and data.get('code') == 200:
                            logger.info("WebSocket authentication successful.")
                            auth_success = True
                            # Subscribe to order updates AFTER successful auth
                            await self.subscribe_order_updates(ws, config.TRADING_PAIR)
                        elif data.get('code') != 200:
                            logger.error(f"WebSocket authentication failed: {data}")
                            raise websockets.exceptions.ConnectionClosedOnError(None, None) # Force reconnect

                    # 4. Listen for messages (including heartbeats and order updates)
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            # Handle ping/pong or other control messages (check HTX docs)
                            if data.get('action') == 'ping':
                                await ws.send(json.dumps({'action': 'pong', 'data': {'ts': data['data']['ts']}}))
                                logger.debug("Sent Auth WS Pong")
                                continue

                             # ** PARSE ORDER UPDATES ACCORDING TO HTX's FORMAT **
                            if data.get('action') == 'push' and data.get('ch') and 'orders' in data['ch']:
                                logger.debug(f"Received order update: {data}")
                                await self.execution_queue.put({'type': 'execution', 'data': data['data']})
                            elif data.get('action') == 'sub' and data.get('code') == 200:
                                logger.info(f"Order subscription confirmed: {data}")
                            else:
                                logger.debug(f"Received unhandled auth WS message: {data}")

                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON from auth WS: {message}")
                        except Exception as e:
                            logger.error(f"Error processing auth WS message: {e}", exc_info=True)

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, asyncio.TimeoutError, websockets.exceptions.InvalidStatusCode) as e:
                logger.warning(f"Authenticated WebSocket connection lost or failed: {e}. Reconnecting in 5 seconds...")
                self.ws_auth = None
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in authenticated WebSocket connection: {e}. Reconnecting in 15 seconds...", exc_info=True)
                self.ws_auth = None
                await asyncio.sleep(15)

    async def subscribe_order_updates(self, ws, symbol):
        """Subscribes to order updates on the authenticated WebSocket (Example)."""
        # ** FIND THE CORRECT SUBSCRIPTION FORMAT IN HTX DOCS **
        sub_msg = {
            "action": "sub",
            "ch": f"orders#{symbol}" # Or maybe orders.* for all symbols
            #"ch": "orders#*" # Example for all symbols
        }
        await ws.send(json.dumps(sub_msg))
        logger.info(f"Subscribing to order updates: {sub_msg['ch']}")

    async def close(self):
        """Closes WebSocket connections and the HTTP session."""
        if self.ws_public:
            await self.ws_public.close()
            logger.info("Public WebSocket closed.")
        if self.ws_auth:
            await self.ws_auth.close()
            logger.info("Authenticated WebSocket closed.")
        await self.close_session()