import os
from typing import List, Dict, Any, Optional
from functools import lru_cache


class Settings:
    # Application info
    app_name: str = "Market Trend API"
    version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database: default to local SQLite for dev; override with POSTGRES URL
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./market_trend.db")
    database_echo: bool = os.getenv("DATABASE_ECHO", "false").lower() == "true"
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))

    # Redis cache
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # CORS
    cors_origins: List[str] = os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:3000,"
            "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,"
            "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175,http://127.0.0.1:5176"
        )
    ).split(",")
    cors_credentials: bool = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"

    # yfinance rate limiting
    yfinance_rate: int = int(os.getenv("YFINANCE_RATE", "30"))
    yfinance_per_seconds: int = int(os.getenv("YFINANCE_PER_SECONDS", "60"))

    # yfinance retries
    yfinance_max_retries: int = int(os.getenv("YFINANCE_MAX_RETRIES", "3"))
    yfinance_backoff_base_seconds: float = float(os.getenv("YFINANCE_BACKOFF_BASE_SECONDS", "0.5"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")  # "json" or "text"
    log_file_enabled: bool = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
    log_file_path: str = os.getenv("LOG_FILE_PATH", "logs/app.log")
    log_console_enabled: bool = os.getenv("LOG_CONSOLE_ENABLED", "true").lower() == "true"

    # Monitoring
    monitoring_enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    metrics_path: str = os.getenv("METRICS_PATH", "/metrics")
    health_path: str = os.getenv("HEALTH_PATH", "/health")
    metrics_collection_interval: int = int(os.getenv("METRICS_COLLECTION_INTERVAL", "30"))

    # Alerting
    alerting_enabled: bool = os.getenv("ALERTING_ENABLED", "true").lower() == "true"
    
    # Email alerting
    alert_email_enabled: bool = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    alert_email_smtp_server: str = os.getenv("ALERT_EMAIL_SMTP_SERVER", "localhost")
    alert_email_smtp_port: int = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
    alert_email_username: Optional[str] = os.getenv("ALERT_EMAIL_USERNAME")
    alert_email_password: Optional[str] = os.getenv("ALERT_EMAIL_PASSWORD")
    alert_email_from: Optional[str] = os.getenv("ALERT_EMAIL_FROM")
    alert_email_to: List[str] = os.getenv("ALERT_EMAIL_TO", "").split(",") if os.getenv("ALERT_EMAIL_TO") else []
    
    # Slack alerting
    alert_slack_enabled: bool = os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true"
    alert_slack_webhook_url: Optional[str] = os.getenv("ALERT_SLACK_WEBHOOK_URL")
    
    # Webhook alerting
    alert_webhook_enabled: bool = os.getenv("ALERT_WEBHOOK_ENABLED", "false").lower() == "true"
    alert_webhook_url: Optional[str] = os.getenv("ALERT_WEBHOOK_URL")

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def get_alerting_config(self) -> Dict[str, Any]:
        """Get alerting configuration dictionary."""
        config = {}
        
        if self.alert_email_enabled and self.alert_email_to:
            config["email"] = {
                "smtp_server": self.alert_email_smtp_server,
                "smtp_port": self.alert_email_smtp_port,
                "username": self.alert_email_username,
                "password": self.alert_email_password,
                "from_email": self.alert_email_from,
                "to_emails": self.alert_email_to
            }
        
        if self.alert_slack_enabled and self.alert_slack_webhook_url:
            config["slack"] = {
                "webhook_url": self.alert_slack_webhook_url
            }
        
        if self.alert_webhook_enabled and self.alert_webhook_url:
            config["webhook"] = {
                "url": self.alert_webhook_url,
                "headers": {}
            }
        
        # Default routing configuration
        config["routing"] = {
            "critical": ["email", "slack"] if self.alert_email_enabled or self.alert_slack_enabled else [],
            "high": ["email", "slack"] if self.alert_email_enabled or self.alert_slack_enabled else [],
            "medium": ["slack"] if self.alert_slack_enabled else [],
            "low": ["slack"] if self.alert_slack_enabled else [],
            "info": []
        }
        
        return config


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
