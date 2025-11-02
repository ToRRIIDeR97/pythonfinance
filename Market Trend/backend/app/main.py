import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .core.config import settings
from .core.db import engine, Base
from .core.logging import setup_logging, get_logger, RequestIDMiddleware
from .core.monitoring import (
    setup_monitoring, health_checker, metrics_collector, 
    MonitoringMiddleware, check_database_health, REGISTRY
)
from .core.alerting import setup_alerting, alert_manager

# Setup logging first
setup_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_file=settings.log_file_path if settings.log_file_enabled else None,
    enable_console=settings.log_console_enabled
)

logger = get_logger("main")

try:
    from .api.v1.markets import router as markets_router
    from .api.v1.sectors import router as sectors_router
    from .api.v1.indicators import router as indicators_router
    from .api.v1.summary import router as summary_router
    from .api.v1.auth import router as auth_router
    from .api.v1.websocket import router as websocket_router
except Exception as e:
    logger.warning("Some routers not available", error=str(e))
    # Allow running even before routers are created
    markets_router = None
    sectors_router = None
    indicators_router = None
    summary_router = None
    auth_router = None
    websocket_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Market Trend API", version=settings.version)
    
    # Startup
    try:
        # Auto-create tables for dev
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        # Setup monitoring
        if settings.monitoring_enabled:
            setup_monitoring(settings.app_name, settings.version)
            
            # Register health checks
            health_checker.register_check("database", lambda: check_database_health(engine))
            
            # Start metrics collection
            await metrics_collector.start()
            logger.info("Monitoring system started")
        
        # Setup alerting
        if settings.alerting_enabled:
            alerting_config = settings.get_alerting_config()
            setup_alerting(alerting_config)
            logger.info("Alerting system configured")
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error("Error during startup", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Market Trend API")
    try:
        if settings.monitoring_enabled:
            await metrics_collector.stop()
            logger.info("Monitoring system stopped")
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Financial market data and analysis API with comprehensive monitoring",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None
)

# Add middleware
app.add_middleware(RequestIDMiddleware)

if settings.monitoring_enabled:
    app.add_middleware(MonitoringMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(settings.health_path)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Run all health checks
        health_results = await health_checker.run_all_checks()
        overall_status = health_checker.get_overall_status(health_results)
        
        # Prepare response
        response_data = {
            "status": overall_status,
            "timestamp": time.time(),
            "version": settings.version,
            "environment": settings.environment,
            "checks": {name: {
                "status": check.status,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "details": check.details
            } for name, check in health_results.items()}
        }
        
        # Set appropriate HTTP status
        status_code = 200 if overall_status == "healthy" else 503
        
        # Log health check
        logger.info(
            "Health check completed",
            overall_status=overall_status,
            checks_count=len(health_results)
        )
        
        return JSONResponse(
            content=response_data,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.get(settings.metrics_path)
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.monitoring_enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    try:
        metrics_data = generate_latest(REGISTRY)
        return PlainTextResponse(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Error generating metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error generating metrics")


@app.get("/api/v1/alerts")
async def get_alerts():
    """Get active alerts."""
    if not settings.alerting_enabled:
        raise HTTPException(status_code=404, detail="Alerting not enabled")
    
    try:
        active_alerts = alert_manager.get_active_alerts()
        return {
            "alerts": [alert.to_dict() for alert in active_alerts],
            "count": len(active_alerts)
        }
    except Exception as e:
        logger.error("Error retrieving alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving alerts")


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = "api"):
    """Acknowledge an alert."""
    if not settings.alerting_enabled:
        raise HTTPException(status_code=404, detail="Alerting not enabled")
    
    try:
        await alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        return {"message": "Alert acknowledged", "alert_id": alert_id}
    except Exception as e:
        logger.error("Error acknowledging alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error acknowledging alert")


@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    if not settings.alerting_enabled:
        raise HTTPException(status_code=404, detail="Alerting not enabled")
    
    try:
        await alert_manager.resolve_alert(alert_id)
        return {"message": "Alert resolved", "alert_id": alert_id}
    except Exception as e:
        logger.error("Error resolving alert", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error resolving alert")


# Include API routers
if markets_router:
    app.include_router(markets_router, prefix="/api/v1/markets", tags=["markets"]) 
if sectors_router:
    app.include_router(sectors_router, prefix="/api/v1", tags=["sectors"])
if indicators_router:
    app.include_router(indicators_router, prefix="/api/v1", tags=["indicators"])
if summary_router:
    app.include_router(summary_router, prefix="/api/v1", tags=["summary"])
if auth_router:
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
if websocket_router:
    app.include_router(websocket_router, prefix="/api/v1", tags=["websocket"])


# Background task for alert rule evaluation
async def alert_evaluation_task():
    """Background task to evaluate alert rules."""
    while True:
        try:
            if settings.alerting_enabled:
                # Collect current system context
                import psutil
                context = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "health_checks": {
                        name: {"status": check.status, "message": check.message}
                        for name, check in health_checker.last_results.items()
                    }
                }
                
                # Evaluate alert rules
                await alert_manager.evaluate_rules(context)
            
            # Wait before next evaluation
            await asyncio.sleep(60)  # Evaluate every minute
            
        except Exception as e:
            logger.error("Error in alert evaluation task", error=str(e))
            await asyncio.sleep(30)  # Wait before retrying


# Start background tasks
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks."""
    if settings.alerting_enabled:
        asyncio.create_task(alert_evaluation_task())
        logger.info("Alert evaluation task started")
