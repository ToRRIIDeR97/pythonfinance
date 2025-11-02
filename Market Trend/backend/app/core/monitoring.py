"""
Monitoring and metrics collection for the Market Trend API.
Provides Prometheus metrics, health checks, and performance monitoring.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from sqlalchemy import text
from redis import Redis

from .logging import get_logger, LoggerMixin


# Prometheus metrics
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Database metrics
DB_QUERY_COUNT = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table'],
    registry=REGISTRY
)

DB_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    registry=REGISTRY
)

DB_CONNECTION_POOL = Gauge(
    'database_connection_pool_size',
    'Database connection pool size',
    ['status'],
    registry=REGISTRY
)

# External API metrics
EXTERNAL_API_CALLS = Counter(
    'external_api_calls_total',
    'Total external API calls',
    ['service', 'endpoint', 'status'],
    registry=REGISTRY
)

EXTERNAL_API_DURATION = Histogram(
    'external_api_duration_seconds',
    'External API call duration in seconds',
    ['service', 'endpoint'],
    registry=REGISTRY
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    ['type'],
    registry=REGISTRY
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_bytes',
    'System disk usage in bytes',
    ['type'],
    registry=REGISTRY
)

# Application metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    registry=REGISTRY
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],
    registry=REGISTRY
)

# Application info
APP_INFO = Info(
    'app_info',
    'Application information',
    registry=REGISTRY
)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0


class HealthChecker(LoggerMixin):
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info("Health check registered", check_name=name)
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a single health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, dict):
                return HealthCheck(
                    name=name,
                    status=result.get("status", "healthy"),
                    message=result.get("message", "OK"),
                    details=result.get("details", {}),
                    duration_ms=duration_ms
                )
            else:
                return HealthCheck(
                    name=name,
                    status="healthy" if result else "unhealthy",
                    message="OK" if result else "Check failed",
                    duration_ms=duration_ms
                )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Health check failed",
                check_name=name,
                error=str(e),
                duration_ms=duration_ms
            )
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            self.last_results[name] = result
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheck]) -> str:
        """Get overall system health status."""
        if not results:
            return "unknown"
        
        statuses = [check.status for check in results.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        else:
            return "degraded"


class MetricsCollector(LoggerMixin):
    """Collects and updates system metrics."""
    
    def __init__(self):
        self.collection_interval = 30  # seconds
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error collecting metrics", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.labels(type="total").set(memory.total)
            SYSTEM_MEMORY_USAGE.labels(type="used").set(memory.used)
            SYSTEM_MEMORY_USAGE.labels(type="available").set(memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.labels(type="total").set(disk.total)
            SYSTEM_DISK_USAGE.labels(type="used").set(disk.used)
            SYSTEM_DISK_USAGE.labels(type="free").set(disk.free)
            
        except Exception as e:
            self.logger.error("Error collecting system metrics", error=str(e))


class MonitoringMiddleware:
    """FastAPI middleware for request monitoring."""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("monitoring")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request_start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Track active connections
        ACTIVE_CONNECTIONS.inc()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - request_start_time
                
                # Update metrics
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)
                
                # Log request
                self.logger.info(
                    "HTTP request completed",
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration * 1000, 2)
                )
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ACTIVE_CONNECTIONS.dec()


# Health check functions
async def check_database_health(engine) -> HealthCheck:
    """Check database connectivity and performance."""
    try:
        start_time = time.time()
        
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="database",
            status="healthy",
            message="Database connection successful",
            details={"response_time_ms": round(duration_ms, 2)},
            duration_ms=duration_ms
        )
    
    except Exception as e:
        return HealthCheck(
            name="database",
            status="unhealthy",
            message=f"Database connection failed: {str(e)}"
        )


async def check_redis_health(redis_client: Redis) -> HealthCheck:
    """Check Redis connectivity and performance."""
    try:
        start_time = time.time()
        
        # Test basic operations
        await redis_client.ping()
        test_key = "health_check_test"
        await redis_client.set(test_key, "test_value", ex=10)
        value = await redis_client.get(test_key)
        await redis_client.delete(test_key)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="redis",
            status="healthy",
            message="Redis connection successful",
            details={
                "response_time_ms": round(duration_ms, 2),
                "test_operation": "success"
            },
            duration_ms=duration_ms
        )
    
    except Exception as e:
        return HealthCheck(
            name="redis",
            status="unhealthy",
            message=f"Redis connection failed: {str(e)}"
        )


def check_system_resources() -> HealthCheck:
    """Check system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Define thresholds
        cpu_threshold = 80
        memory_threshold = 85
        disk_threshold = 90
        
        issues = []
        status = "healthy"
        
        if cpu_percent > cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent}%")
            status = "degraded"
        
        if memory.percent > memory_threshold:
            issues.append(f"High memory usage: {memory.percent}%")
            status = "degraded"
        
        if (disk.used / disk.total * 100) > disk_threshold:
            disk_percent = disk.used / disk.total * 100
            issues.append(f"High disk usage: {disk_percent:.1f}%")
            status = "degraded"
        
        message = "System resources OK" if not issues else "; ".join(issues)
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.used / disk.total * 100,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        )
    
    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status="unhealthy",
            message=f"Failed to check system resources: {str(e)}"
        )


# Global instances
health_checker = HealthChecker()
metrics_collector = MetricsCollector()


def setup_monitoring(app_name: str, app_version: str):
    """Setup monitoring and metrics collection."""
    # Set application info
    APP_INFO.info({
        'name': app_name,
        'version': app_version,
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
    })
    
    # Register default health checks
    health_checker.register_check("system_resources", check_system_resources)
    
    logger = get_logger("monitoring")
    logger.info("Monitoring setup completed", app_name=app_name, app_version=app_version)