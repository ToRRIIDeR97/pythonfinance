import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file if it exists

# --- API Configuration ---
# **NEVER HARDCODE KEYS IN PRODUCTION - Use Environment Variables or Vault**
HTX_API_KEY = os.getenv("HTX_API_KEY", "YOUR_API_KEY")
HTX_SECRET_KEY = os.getenv("HTX_SECRET_KEY", "YOUR_SECRET_KEY")

# **Placeholder URLs - GET THESE FROM HTX DOCUMENTATION**
HTX_REST_URL = "https://api.htx.com" # Example REST endpoint
HTX_WS_URL = "wss://api.htx.com/ws"  # Example WebSocket endpoint
HTX_WS_AUTH_URL = "wss://api-aws.huobi.pro/ws/v2" # Example Auth WS endpoint (check docs!)

# --- Trading Parameters ---
TRADING_PAIR = "btcusdt" # Example trading pair
ORDER_SIZE_BTC = 0.001    # Example order size in BTC
MAX_POSITION_BTC = 0.01   # Max allowed position size in BTC

# --- Strategy Parameters ---
IMBALANCE_THRESHOLD = 0.6 # Example threshold for order flow imbalance

# --- Risk Parameters ---
MAX_ORDER_SIZE_USD = 1000 # Example max order value
MAX_DRAWDOWN_PERCENT = 5.0 # Max portfolio loss percentage before halting
KILL_SWITCH_ACTIVE = False # Global kill switch

# --- Latency / Timing ---
API_TIMEOUT_SECONDS = 0.5 # Timeout for REST requests (very short for HFT)
HEARTBEAT_INTERVAL = 5    # WebSocket heartbeat interval (check HTX docs)

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"