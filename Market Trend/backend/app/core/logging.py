"""
Logging configuration for the Market Trend API.
Provides structured logging with multiple outputs and formats.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str = None,
    enable_console: bool = True
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'text')
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handlers = {}
    
    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "json" if log_format == "json" else "text",
        }
    
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        }
        
        # Separate error log file
        handlers["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(Path(log_file).with_suffix(".error.log")),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": "ERROR",
        }
    
    formatters = {
        "json": {
            "()": jsonlogger.JsonFormatter,
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
        },
        "text": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "app": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


# Request ID middleware for tracing
class RequestIDMiddleware:
    """Middleware to add request ID to all logs."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            import uuid
            request_id = str(uuid.uuid4())
            
            # Add request ID to structlog context
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)
            
            # Add to headers for response
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append([b"x-request-id", request_id.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Performance logging decorator
def log_performance(func_name: str = None):
    """Decorator to log function performance metrics."""
    def decorator(func):
        import time
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger("performance")
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    "Function completed",
                    function=name,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function failed",
                    function=name,
                    duration_ms=round(duration * 1000, 2),
                    status="error",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger("performance")
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    "Function completed",
                    function=name,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function failed",
                    function=name,
                    duration_ms=round(duration * 1000, 2),
                    status="error",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    
    return decorator