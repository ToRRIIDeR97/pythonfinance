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
INITIAL_CAPITAL = 100000.0   # Initial capital for backtesting and simulation
TRADING_PAIR = "btcusdt" # Example trading pair for live OFI strategy (e.g., from HTX)
# ORDER_SIZE_BTC is now OFI_ORDER_SIZE_ASSET under Strategy Parameters
MAX_POSITION_BTC = 0.01   # Max allowed position size in BTC

# --- Strategy Parameters ---
# Parameters for OrderFlowImbalanceStrategy
OFI_IMBALANCE_THRESHOLD = 0.6 # Example threshold for order flow imbalance
OFI_ORDER_SIZE_ASSET = 0.001    # Example order size for OFI strategy (e.g., in BTC for btcusdt)
OFI_COOLDOWN_S = 0.5          # Cooldown in seconds between OFI signals

# Parameters for MovingAverageCrossoverStrategy (for backtesting)
MAC_TICKER = 'BTC-USD'         # Ticker symbol for Yahoo Finance (e.g., 'BTC-USD', 'AAPL')
MAC_SHORT_WINDOW = 10          # Short moving average window (e.g., 10 days/periods)
MAC_LONG_WINDOW = 30           # Long moving average window (e.g., 30 days/periods)
MAC_ORDER_SIZE = 0.1           # Order size for MAC strategy (e.g., 0.1 BTC or 10 shares)
MAC_START_DATE = '2022-01-01'  # Start date for backtesting data
MAC_END_DATE = '2023-12-31'    # End date for backtesting data
MAC_INTERVAL = '1d'            # Data interval for backtesting ('1d', '1h', etc.)


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