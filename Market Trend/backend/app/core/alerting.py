"""
Alerting system for the Market Trend API.
Provides alert management, notification channels, and escalation policies.
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiohttp
from pydantic import BaseModel, EmailStr

from .logging import get_logger, LoggerMixin
from .monitoring import HealthCheck


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source": self.source,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "metadata": self.metadata,
            "tags": self.tags
        }


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = get_logger(f"alerting.{name}")
    
    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Returns True if successful."""
        raise NotImplementedError


class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)
            username = self.config.get("username")
            password = self.config.get("password")
            from_email = self.config.get("from_email")
            to_emails = self.config.get("to_emails", [])
            
            if not to_emails:
                self.logger.warning("No recipient emails configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
--------------
Title: {alert.title}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Status: {alert.status.value}
Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Tags: {', '.join(alert.tags)}

Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(msg)
            
            self.logger.info(
                "Email alert sent",
                alert_id=alert.id,
                recipients=len(to_emails)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to send email alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False


class SlackChannel(NotificationChannel):
    """Slack notification channel."""
    
    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                self.logger.warning("No Slack webhook URL configured")
                return False
            
            # Color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.HIGH: "#FF8C00",
                AlertSeverity.MEDIUM: "#FFD700",
                AlertSeverity.LOW: "#32CD32",
                AlertSeverity.INFO: "#87CEEB"
            }
            
            # Create Slack message
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"[{alert.severity.value.upper()}] {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Created",
                                "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            }
                        ],
                        "footer": "Market Trend API",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Add metadata if present
            if alert.metadata:
                payload["attachments"][0]["fields"].append({
                    "title": "Metadata",
                    "value": f"```{json.dumps(alert.metadata, indent=2)}```",
                    "short": False
                })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Slack alert sent", alert_id=alert.id)
                        return True
                    else:
                        self.logger.error(
                            "Failed to send Slack alert",
                            alert_id=alert.id,
                            status_code=response.status
                        )
                        return False
                        
        except Exception as e:
            self.logger.error(
                "Failed to send Slack alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    async def send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            url = self.config.get("url")
            if not url:
                self.logger.warning("No webhook URL configured")
                return False
            
            headers = self.config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "market-trend-api"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        self.logger.info("Webhook alert sent", alert_id=alert.id)
                        return True
                    else:
                        self.logger.error(
                            "Failed to send webhook alert",
                            alert_id=alert.id,
                            status_code=response.status
                        )
                        return False
                        
        except Exception as e:
            self.logger.error(
                "Failed to send webhook alert",
                alert_id=alert.id,
                error=str(e)
            )
            return False


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        title_template: str,
        description_template: str,
        tags: List[str] = None,
        cooldown_minutes: int = 5
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.title_template = title_template
        self.description_template = description_template
        self.tags = tags or []
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered: Optional[datetime] = None
    
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if rule should trigger."""
        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return False
        
        # Check condition
        return self.condition(context)
    
    def create_alert(self, context: Dict[str, Any]) -> Alert:
        """Create alert from rule."""
        alert_id = f"{self.name}_{int(datetime.utcnow().timestamp())}"
        
        return Alert(
            id=alert_id,
            title=self.title_template.format(**context),
            description=self.description_template.format(**context),
            severity=self.severity,
            source=f"rule:{self.name}",
            metadata=context,
            tags=self.tags
        )


class AlertManager(LoggerMixin):
    """Central alert management system."""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Channel routing by severity
        self.severity_routing: Dict[AlertSeverity, List[str]] = {
            AlertSeverity.CRITICAL: [],
            AlertSeverity.HIGH: [],
            AlertSeverity.MEDIUM: [],
            AlertSeverity.LOW: [],
            AlertSeverity.INFO: []
        }
    
    def add_channel(self, channel: NotificationChannel):
        """Add notification channel."""
        self.channels[channel.name] = channel
        self.logger.info("Notification channel added", channel=channel.name)
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules[rule.name] = rule
        self.logger.info("Alert rule added", rule=rule.name)
    
    def set_routing(self, severity: AlertSeverity, channels: List[str]):
        """Set channel routing for severity level."""
        self.severity_routing[severity] = channels
        self.logger.info(
            "Alert routing configured",
            severity=severity.value,
            channels=channels
        )
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> Alert:
        """Create and process a new alert."""
        alert_id = f"{source}_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            metadata=metadata or {},
            tags=tags or []
        )
        
        await self.process_alert(alert)
        return alert
    
    async def process_alert(self, alert: Alert):
        """Process an alert through the system."""
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        self.logger.info(
            "Alert created",
            alert_id=alert.id,
            severity=alert.severity.value,
            source=alert.source
        )
        
        # Send notifications
        await self.send_notifications(alert)
    
    async def send_notifications(self, alert: Alert):
        """Send alert notifications to configured channels."""
        channels_to_notify = self.severity_routing.get(alert.severity, [])
        
        if not channels_to_notify:
            self.logger.warning(
                "No channels configured for severity",
                severity=alert.severity.value
            )
            return
        
        # Send to each channel
        for channel_name in channels_to_notify:
            if channel_name in self.channels:
                try:
                    channel = self.channels[channel_name]
                    success = await channel.send(alert)
                    
                    if success:
                        self.logger.info(
                            "Alert notification sent",
                            alert_id=alert.id,
                            channel=channel_name
                        )
                    else:
                        self.logger.error(
                            "Alert notification failed",
                            alert_id=alert.id,
                            channel=channel_name
                        )
                        
                except Exception as e:
                    self.logger.error(
                        "Error sending alert notification",
                        alert_id=alert.id,
                        channel=channel_name,
                        error=str(e)
                    )
            else:
                self.logger.warning(
                    "Channel not found",
                    channel=channel_name
                )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.utcnow()
            
            self.logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info("Alert resolved", alert_id=alert_id)
    
    async def evaluate_rules(self, context: Dict[str, Any]):
        """Evaluate all alert rules against current context."""
        for rule_name, rule in self.rules.items():
            try:
                if rule.should_trigger(context):
                    alert = rule.create_alert(context)
                    await self.process_alert(alert)
                    rule.last_triggered = datetime.utcnow()
                    
            except Exception as e:
                self.logger.error(
                    "Error evaluating alert rule",
                    rule=rule_name,
                    error=str(e)
                )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]


# Global alert manager instance
alert_manager = AlertManager()


# Predefined alert rules
def create_default_rules():
    """Create default alert rules."""
    
    # High CPU usage rule
    def high_cpu_condition(context: Dict[str, Any]) -> bool:
        return context.get("cpu_percent", 0) > 80
    
    alert_manager.add_rule(AlertRule(
        name="high_cpu_usage",
        condition=high_cpu_condition,
        severity=AlertSeverity.HIGH,
        title_template="High CPU Usage: {cpu_percent:.1f}%",
        description_template="System CPU usage is at {cpu_percent:.1f}%, which exceeds the 80% threshold.",
        tags=["system", "performance"],
        cooldown_minutes=10
    ))
    
    # High memory usage rule
    def high_memory_condition(context: Dict[str, Any]) -> bool:
        return context.get("memory_percent", 0) > 85
    
    alert_manager.add_rule(AlertRule(
        name="high_memory_usage",
        condition=high_memory_condition,
        severity=AlertSeverity.HIGH,
        title_template="High Memory Usage: {memory_percent:.1f}%",
        description_template="System memory usage is at {memory_percent:.1f}%, which exceeds the 85% threshold.",
        tags=["system", "memory"],
        cooldown_minutes=10
    ))
    
    # Database connection failure rule
    def db_failure_condition(context: Dict[str, Any]) -> bool:
        health_checks = context.get("health_checks", {})
        db_check = health_checks.get("database", {})
        return db_check.get("status") == "unhealthy"
    
    alert_manager.add_rule(AlertRule(
        name="database_failure",
        condition=db_failure_condition,
        severity=AlertSeverity.CRITICAL,
        title_template="Database Connection Failure",
        description_template="Database health check failed: {health_checks[database][message]}",
        tags=["database", "critical"],
        cooldown_minutes=5
    ))
    
    # High error rate rule
    def high_error_rate_condition(context: Dict[str, Any]) -> bool:
        error_rate = context.get("error_rate_percent", 0)
        return error_rate > 5  # 5% error rate threshold
    
    alert_manager.add_rule(AlertRule(
        name="high_error_rate",
        condition=high_error_rate_condition,
        severity=AlertSeverity.HIGH,
        title_template="High Error Rate: {error_rate_percent:.1f}%",
        description_template="API error rate is at {error_rate_percent:.1f}%, which exceeds the 5% threshold.",
        tags=["api", "errors"],
        cooldown_minutes=15
    ))


def setup_alerting(config: Dict[str, Any]):
    """Setup alerting system with configuration."""
    logger = get_logger("alerting")
    
    # Setup notification channels
    if "email" in config:
        email_config = config["email"]
        email_channel = EmailChannel("email", email_config)
        alert_manager.add_channel(email_channel)
    
    if "slack" in config:
        slack_config = config["slack"]
        slack_channel = SlackChannel("slack", slack_config)
        alert_manager.add_channel(slack_channel)
    
    if "webhook" in config:
        webhook_config = config["webhook"]
        webhook_channel = WebhookChannel("webhook", webhook_config)
        alert_manager.add_channel(webhook_channel)
    
    # Setup routing
    routing = config.get("routing", {})
    for severity_str, channels in routing.items():
        try:
            severity = AlertSeverity(severity_str)
            alert_manager.set_routing(severity, channels)
        except ValueError:
            logger.warning("Invalid severity level in routing", severity=severity_str)
    
    # Create default rules
    create_default_rules()
    
    logger.info("Alerting system setup completed")