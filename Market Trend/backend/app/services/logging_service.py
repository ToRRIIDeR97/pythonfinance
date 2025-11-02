import logging

logger = logging.getLogger("market_trend")
logging.basicConfig(level=logging.INFO)

class LoggingService:
    def log_info(self, message: str):
        logger.info(message)

    def log_warn(self, message: str):
        logger.warning(message)

    def log_error(self, message: str, exception: Exception | None = None):
        logger.error(message)
        if exception:
            logger.exception(exception)