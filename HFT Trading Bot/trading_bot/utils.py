import logging
import datetime
import config
import hmac
import hashlib
import base64
import json
from urllib.parse import urlencode

def setup_logger():
    """Configures the logger."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    # Suppress noisy library logs if needed
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    return logging.getLogger("TradingBot")

# --- Authentication Helper (EXAMPLE - Adapt to HTX's specific method) ---
def create_signature_v2(api_key, secret_key, method, host, path, params):
    """Creates a V2 signature for HTX API (Example based on common patterns)."""
    # 1. Prepare parameters string
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    params_to_sign = {
        'AccessKeyId': api_key,
        'SignatureMethod': 'HmacSHA256',
        'SignatureVersion': '2',
        'Timestamp': timestamp,
        **params # Add specific endpoint params
    }
    # 2. Sort parameters alphabetically by key
    sorted_params = sorted(params_to_sign.items())
    encode_params = urlencode(sorted_params)

    # 3. Construct string to sign
    payload = f"{method}\n{host}\n{path}\n{encode_params}"

    # 4. Calculate HMAC SHA256 signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).digest()

    # 5. Base64 encode the signature
    signature_b64 = base64.b64encode(signature).decode('utf-8')

    # 6. Add signature to parameters for the final request
    final_params = {**params_to_sign, 'Signature': signature_b64}
    return final_params

logger = setup_logger()