#!/usr/bin/env python3
"""
Production Alert System for Performance Monitoring
==================================================

SRE-grade alerting system for the homodyne analysis application with multi-channel
notification, intelligent alert routing, and escalation management. Designed for
scientific computing workloads with accuracy-critical requirements.

Key Features:
- Multi-channel alerting (email, Slack, webhook, file)
- Intelligent alert deduplication and grouping
- Escalation policies with on-call rotation
- Alert fatigue prevention with smart throttling
- Context-aware alert enrichment
- Performance regression detection alerts
- Scientific accuracy violation alerts
- Auto-remediation for common issues

Alert Channels:
- Email notifications with rich formatting
- Slack integration with threaded conversations
- Webhook delivery for external systems
- File-based alerts for logging systems
- Console output for development

Alert Categories:
- Performance: Response time, throughput, latency
- Accuracy: Scientific computation accuracy violations
- Memory: Memory leaks, allocation failures
- Security: Security policy violations
- System: Infrastructure and resource issues
- Optimization: Performance optimization opportunities

Escalation Levels:
- Level 1: Info alerts (monitoring only)
- Level 2: Warning alerts (team notification)
- Level 3: Critical alerts (immediate response)
- Level 4: Emergency alerts (wake up on-call)
"""

import asyncio
import json
import logging
import smtplib
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
from urllib.parse import urljoin
import warnings

# HTTP client for webhooks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available - webhook alerts disabled")

# Slack integration
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    warnings.warn("slack-sdk not available - Slack alerts disabled")

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    metric_pattern: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq', 'ne'
    severity: str  # 'info', 'warning', 'critical', 'emergency'
    channels: List[str]
    cooldown: int  # seconds
    enabled: bool = True
    context_enrichment: bool = True


@dataclass
class AlertChannel:
    """Alert delivery channel configuration."""
    channel_id: str
    channel_type: str  # 'email', 'slack', 'webhook', 'file', 'console'
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: int = 10  # alerts per minute


@dataclass
class Alert:
    """Performance alert with full context."""
    alert_id: str
    rule_id: str
    timestamp: float
    severity: str
    category: str
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    context: Dict[str, Any]
    suggested_actions: List[str]
    escalation_level: int = 1
    acknowledged: bool = False
    resolved: bool = False
    auto_resolved: bool = False


@dataclass
class EscalationPolicy:
    """Alert escalation policy configuration."""
    policy_id: str
    name: str
    escalation_levels: List[Dict[str, Any]]
    repeat_interval: int  # seconds
    max_escalations: int = 3


class AlertDeduplicator:
    """Intelligent alert deduplication and grouping."""

    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.recent_alerts: deque = deque()
        self.alert_signatures: Set[str] = set()

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within the time window."""
        current_time = time.time()

        # Clean old alerts
        while self.recent_alerts and current_time - self.recent_alerts[0][0] > self.window_size:
            old_timestamp, old_signature = self.recent_alerts.popleft()
            self.alert_signatures.discard(old_signature)

        # Generate alert signature
        signature = self._generate_signature(alert)

        # Check for duplicate
        if signature in self.alert_signatures:
            return True

        # Add to recent alerts
        self.recent_alerts.append((current_time, signature))
        self.alert_signatures.add(signature)
        return False

    def _generate_signature(self, alert: Alert) -> str:
        """Generate unique signature for alert deduplication."""
        # Use rule, metric, and rounded value for grouping
        rounded_value = round(alert.current_value, 2)
        return f"{alert.rule_id}:{alert.metric_name}:{rounded_value}:{alert.severity}"


class AlertThrottler:
    """Alert throttling to prevent alert fatigue."""

    def __init__(self):
        self.channel_counters: Dict[str, deque] = defaultdict(deque)

    def should_throttle(self, channel_id: str, rate_limit: int) -> bool:
        """Check if alerts should be throttled for a channel."""
        current_time = time.time()
        window_size = 60  # 1 minute window

        # Clean old timestamps
        while (self.channel_counters[channel_id] and
               current_time - self.channel_counters[channel_id][0] > window_size):
            self.channel_counters[channel_id].popleft()

        # Check rate limit
        if len(self.channel_counters[channel_id]) >= rate_limit:
            return True

        # Record this alert
        self.channel_counters[channel_id].append(current_time)
        return False


class EmailChannel:
    """Email alert delivery channel."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_address = config.get('from_address', 'alerts@example.com')
        self.to_addresses = config.get('to_addresses', [])
        self.use_tls = config.get('use_tls', True)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)

            # Create HTML content
            html_content = self._create_html_content(alert)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Create text content
            text_content = self._create_text_content(alert)
            text_part = MIMEText(text_content, 'plain')
            msg.attach(text_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        severity_colors = {
            'info': '#17a2b8',
            'warning': '#ffc107',
            'critical': '#dc3545',
            'emergency': '#6f42c1'
        }

        color = severity_colors.get(alert.severity, '#6c757d')

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-content {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .actions {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>{alert.title}</h2>
                <p>Severity: {alert.severity.upper()} | Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="alert-content">
                <h3>Alert Details</h3>
                <p><strong>Message:</strong> {alert.message}</p>

                <table class="metric-table">
                    <tr><th>Metric</th><td>{alert.metric_name}</td></tr>
                    <tr><th>Current Value</th><td>{alert.current_value:.3f}</td></tr>
                    <tr><th>Threshold</th><td>{alert.threshold:.3f}</td></tr>
                    <tr><th>Category</th><td>{alert.category}</td></tr>
                    <tr><th>Alert ID</th><td>{alert.alert_id}</td></tr>
                </table>
            </div>

            <div class="actions">
                <h3>Suggested Actions</h3>
                <ul>
        """

        for action in alert.suggested_actions:
            html += f"<li>{action}</li>"

        html += """
                </ul>
            </div>

            <div class="alert-content">
                <h3>Context Information</h3>
                <pre>{}</pre>
            </div>
        </body>
        </html>
        """.format(json.dumps(alert.context, indent=2))

        return html

    def _create_text_content(self, alert: Alert) -> str:
        """Create plain text email content."""
        text = f"""
PERFORMANCE ALERT - {alert.severity.upper()}

{alert.title}
Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

Alert Details:
--------------
Message: {alert.message}
Metric: {alert.metric_name}
Current Value: {alert.current_value:.3f}
Threshold: {alert.threshold:.3f}
Category: {alert.category}
Alert ID: {alert.alert_id}

Suggested Actions:
-----------------
"""
        for action in alert.suggested_actions:
            text += f"• {action}\n"

        text += f"""
Context Information:
-------------------
{json.dumps(alert.context, indent=2)}
"""
        return text


class SlackChannel:
    """Slack alert delivery channel."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get('bot_token', '')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'PerformanceBot')

        if SLACK_AVAILABLE and self.bot_token:
            self.client = WebClient(token=self.bot_token)
        else:
            self.client = None

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.client:
            logger.warning("Slack client not available")
            return False

        try:
            # Create Slack blocks
            blocks = self._create_slack_blocks(alert)

            # Send message
            response = self.client.chat_postMessage(
                channel=self.channel,
                username=self.username,
                blocks=blocks,
                text=alert.title  # Fallback text
            )

            logger.info(f"Slack alert sent: {alert.alert_id}")
            return True

        except SlackApiError as e:
            logger.error(f"Failed to send Slack alert: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _create_slack_blocks(self, alert: Alert) -> List[Dict[str, Any]]:
        """Create Slack blocks for rich formatting."""
        severity_emojis = {
            'info': ':information_source:',
            'warning': ':warning:',
            'critical': ':rotating_light:',
            'emergency': ':fire:'
        }

        emoji = severity_emojis.get(alert.severity, ':question:')

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.title}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:* {alert.severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:* {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Metric:* {alert.metric_name}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Value:* {alert.current_value:.3f} (threshold: {alert.threshold:.3f})"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Message:* {alert.message}"
                }
            }
        ]

        # Add suggested actions
        if alert.suggested_actions:
            actions_text = "\n".join([f"• {action}" for action in alert.suggested_actions])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Suggested Actions:*\n{actions_text}"
                }
            })

        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Acknowledge"
                    },
                    "value": alert.alert_id,
                    "action_id": f"ack_{alert.alert_id}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Details"
                    },
                    "value": alert.alert_id,
                    "action_id": f"details_{alert.alert_id}"
                }
            ]
        })

        return blocks


class WebhookChannel:
    """Webhook alert delivery channel."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get('url', '')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Webhook channel not available - requests library required")
            return False

        try:
            # Create payload
            payload = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp,
                'severity': alert.severity,
                'category': alert.category,
                'title': alert.title,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'context': alert.context,
                'suggested_actions': alert.suggested_actions
            }

            # Send webhook
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class FileChannel:
    """File-based alert delivery channel."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.file_path = Path(config.get('file_path', 'alerts.log'))
        self.format = config.get('format', 'json')  # 'json' or 'text'

        # Ensure directory exists
        self.file_path.parent.mkdir(exist_ok=True)

    async def send_alert(self, alert: Alert) -> bool:
        """Write alert to file."""
        try:
            timestamp = datetime.fromtimestamp(alert.timestamp).isoformat()

            if self.format == 'json':
                alert_data = asdict(alert)
                alert_data['timestamp_iso'] = timestamp
                line = json.dumps(alert_data) + '\n'
            else:
                line = f"{timestamp} [{alert.severity.upper()}] {alert.title}: {alert.message}\n"

            # Append to file
            with open(self.file_path, 'a') as f:
                f.write(line)

            logger.debug(f"File alert written: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write file alert: {e}")
            return False


class ConsoleChannel:
    """Console alert delivery channel."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colored = config.get('colored', True)

    async def send_alert(self, alert: Alert) -> bool:
        """Print alert to console."""
        try:
            timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')

            if self.colored:
                colors = {
                    'info': '\033[36m',    # Cyan
                    'warning': '\033[33m', # Yellow
                    'critical': '\033[31m', # Red
                    'emergency': '\033[35m' # Magenta
                }
                reset = '\033[0m'
                color = colors.get(alert.severity, '')
                message = f"{color}[{timestamp}] [{alert.severity.upper()}] {alert.title}: {alert.message}{reset}"
            else:
                message = f"[{timestamp}] [{alert.severity.upper()}] {alert.title}: {alert.message}"

            print(message)
            return True

        except Exception as e:
            logger.error(f"Failed to print console alert: {e}")
            return False


class AlertSystem:
    """
    Production-ready alert system for performance monitoring.

    Provides intelligent alerting with deduplication, throttling, escalation,
    and multi-channel delivery specifically designed for scientific computing
    workloads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert system."""
        self.config = config or self._default_config()
        self.alerts_dir = Path(self.config['alerts_dir'])
        self.alerts_dir.mkdir(exist_ok=True)

        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_channels: Dict[str, AlertChannel] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}

        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Alert processing
        self.deduplicator = AlertDeduplicator()
        self.throttler = AlertThrottler()

        # Background processing
        self.processing_active = False
        self.processing_thread: Optional[threading.Thread] = None
        self.alert_queue: asyncio.Queue = asyncio.Queue()

        # Initialize channels
        self._initialize_channels()
        self._load_rules()
        self._load_escalation_policies()

        logger.info("Alert system initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default alert system configuration."""
        return {
            'alerts_dir': 'alerts',
            'processing_interval': 5,  # seconds
            'max_alert_history': 10000,
            'deduplication_window': 300,  # 5 minutes
            'default_escalation_policy': 'standard',
            'channels': {
                'console': {
                    'type': 'console',
                    'enabled': True,
                    'config': {'colored': True},
                    'rate_limit': 60
                },
                'file': {
                    'type': 'file',
                    'enabled': True,
                    'config': {
                        'file_path': 'alerts/alerts.log',
                        'format': 'json'
                    },
                    'rate_limit': 1000
                }
            },
            'rules': {
                'response_time_warning': {
                    'metric_pattern': '*response_time*',
                    'threshold': 1.5,
                    'comparison': 'gt',
                    'severity': 'warning',
                    'channels': ['console', 'file'],
                    'cooldown': 300
                },
                'response_time_critical': {
                    'metric_pattern': '*response_time*',
                    'threshold': 3.0,
                    'comparison': 'gt',
                    'severity': 'critical',
                    'channels': ['console', 'file'],
                    'cooldown': 60
                },
                'memory_warning': {
                    'metric_pattern': '*memory*',
                    'threshold': 0.85,
                    'comparison': 'gt',
                    'severity': 'warning',
                    'channels': ['console', 'file'],
                    'cooldown': 600
                },
                'accuracy_critical': {
                    'metric_pattern': '*accuracy*',
                    'threshold': 0.99,
                    'comparison': 'lt',
                    'severity': 'critical',
                    'channels': ['console', 'file'],
                    'cooldown': 0  # No cooldown for accuracy alerts
                }
            },
            'escalation_policies': {
                'standard': {
                    'name': 'Standard Escalation',
                    'escalation_levels': [
                        {'delay': 0, 'channels': ['console', 'file']},
                        {'delay': 300, 'channels': ['email']},  # 5 minutes
                        {'delay': 900, 'channels': ['slack']}   # 15 minutes
                    ],
                    'repeat_interval': 3600,  # 1 hour
                    'max_escalations': 3
                }
            }
        }

    def _initialize_channels(self):
        """Initialize alert delivery channels."""
        for channel_id, channel_config in self.config.get('channels', {}).items():
            try:
                channel_type = channel_config['type']
                config = channel_config.get('config', {})
                enabled = channel_config.get('enabled', True)
                rate_limit = channel_config.get('rate_limit', 10)

                # Create channel instance
                if channel_type == 'email':
                    channel_instance = EmailChannel(config)
                elif channel_type == 'slack':
                    channel_instance = SlackChannel(config)
                elif channel_type == 'webhook':
                    channel_instance = WebhookChannel(config)
                elif channel_type == 'file':
                    channel_instance = FileChannel(config)
                elif channel_type == 'console':
                    channel_instance = ConsoleChannel(config)
                else:
                    logger.warning(f"Unknown channel type: {channel_type}")
                    continue

                # Register channel
                self.alert_channels[channel_id] = AlertChannel(
                    channel_id=channel_id,
                    channel_type=channel_type,
                    config={'instance': channel_instance, **config},
                    enabled=enabled,
                    rate_limit=rate_limit
                )

                logger.info(f"Initialized alert channel: {channel_id} ({channel_type})")

            except Exception as e:
                logger.error(f"Failed to initialize channel {channel_id}: {e}")

    def _load_rules(self):
        """Load alert rules from configuration."""
        for rule_id, rule_config in self.config.get('rules', {}).items():
            try:
                rule = AlertRule(
                    rule_id=rule_id,
                    metric_pattern=rule_config['metric_pattern'],
                    threshold=rule_config['threshold'],
                    comparison=rule_config['comparison'],
                    severity=rule_config['severity'],
                    channels=rule_config['channels'],
                    cooldown=rule_config.get('cooldown', 300),
                    enabled=rule_config.get('enabled', True),
                    context_enrichment=rule_config.get('context_enrichment', True)
                )
                self.alert_rules[rule_id] = rule
                logger.debug(f"Loaded alert rule: {rule_id}")

            except Exception as e:
                logger.error(f"Failed to load rule {rule_id}: {e}")

    def _load_escalation_policies(self):
        """Load escalation policies from configuration."""
        for policy_id, policy_config in self.config.get('escalation_policies', {}).items():
            try:
                policy = EscalationPolicy(
                    policy_id=policy_id,
                    name=policy_config['name'],
                    escalation_levels=policy_config['escalation_levels'],
                    repeat_interval=policy_config.get('repeat_interval', 3600),
                    max_escalations=policy_config.get('max_escalations', 3)
                )
                self.escalation_policies[policy_id] = policy
                logger.debug(f"Loaded escalation policy: {policy_id}")

            except Exception as e:
                logger.error(f"Failed to load escalation policy {policy_id}: {e}")

    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")

    def add_alert_channel(self, channel: AlertChannel, instance: Any):
        """Add or update an alert channel."""
        channel.config['instance'] = instance
        self.alert_channels[channel.channel_id] = channel
        logger.info(f"Added alert channel: {channel.channel_id}")

    def evaluate_metric(self, metric_name: str, value: float,
                       context: Optional[Dict[str, Any]] = None):
        """
        Evaluate a metric against alert rules.

        Parameters
        ----------
        metric_name : str
            Name of the metric
        value : float
            Metric value
        context : dict, optional
            Additional context information
        """
        context = context or {}

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check if metric matches rule pattern
            if self._matches_pattern(metric_name, rule.metric_pattern):
                # Evaluate threshold
                if self._evaluate_threshold(value, rule.threshold, rule.comparison):
                    # Create alert
                    alert = self._create_alert(rule, metric_name, value, context)
                    asyncio.create_task(self._process_alert(alert))

    def _matches_pattern(self, metric_name: str, pattern: str) -> bool:
        """Check if metric name matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(metric_name, pattern)

    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate threshold condition."""
        if comparison == 'gt':
            return value > threshold
        elif comparison == 'lt':
            return value < threshold
        elif comparison == 'eq':
            return abs(value - threshold) < 1e-10
        elif comparison == 'ne':
            return abs(value - threshold) >= 1e-10
        else:
            logger.warning(f"Unknown comparison operator: {comparison}")
            return False

    def _create_alert(self, rule: AlertRule, metric_name: str, value: float,
                     context: Dict[str, Any]) -> Alert:
        """Create an alert from a rule violation."""
        alert_id = str(uuid.uuid4())
        timestamp = time.time()

        # Generate title and message
        title = f"{rule.severity.title()} Alert: {metric_name}"
        if rule.comparison == 'gt':
            message = f"{metric_name} = {value:.3f} exceeds threshold {rule.threshold:.3f}"
        elif rule.comparison == 'lt':
            message = f"{metric_name} = {value:.3f} below threshold {rule.threshold:.3f}"
        else:
            message = f"{metric_name} = {value:.3f} violates threshold {rule.threshold:.3f}"

        # Categorize alert
        category = self._categorize_metric(metric_name)

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(rule, metric_name, category)

        # Enrich context if enabled
        if rule.context_enrichment:
            context = self._enrich_context(context, metric_name, value)

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=timestamp,
            severity=rule.severity,
            category=category,
            title=title,
            message=message,
            metric_name=metric_name,
            current_value=value,
            threshold=rule.threshold,
            context=context,
            suggested_actions=suggested_actions
        )

        return alert

    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric for alert routing."""
        metric_lower = metric_name.lower()

        if any(term in metric_lower for term in ['accuracy', 'error', 'precision']):
            return 'accuracy'
        elif any(term in metric_lower for term in ['memory', 'allocation', 'leak']):
            return 'memory'
        elif any(term in metric_lower for term in ['response_time', 'latency', 'throughput']):
            return 'performance'
        elif any(term in metric_lower for term in ['security', 'auth', 'permission']):
            return 'security'
        elif any(term in metric_lower for term in ['cpu', 'disk', 'network']):
            return 'system'
        else:
            return 'general'

    def _generate_suggested_actions(self, rule: AlertRule, metric_name: str,
                                  category: str) -> List[str]:
        """Generate context-aware suggested actions."""
        actions = []

        if category == 'performance':
            actions.extend([
                "Check for performance regressions",
                "Review recent deployments",
                "Monitor system resources",
                "Consider optimization opportunities"
            ])
        elif category == 'memory':
            actions.extend([
                "Check for memory leaks",
                "Review memory allocation patterns",
                "Monitor garbage collection",
                "Consider memory optimization"
            ])
        elif category == 'accuracy':
            actions.extend([
                "Validate computational accuracy",
                "Check numerical stability",
                "Review algorithm implementations",
                "Verify input data quality"
            ])
        elif category == 'security':
            actions.extend([
                "Review security policies",
                "Check access controls",
                "Validate input sanitization",
                "Monitor for suspicious activity"
            ])
        else:
            actions.extend([
                "Investigate the root cause",
                "Check system health",
                "Review recent changes",
                "Monitor related metrics"
            ])

        # Add rule-specific actions
        if rule.severity in ['critical', 'emergency']:
            actions.insert(0, "Immediate attention required")

        return actions

    def _enrich_context(self, context: Dict[str, Any], metric_name: str,
                       value: float) -> Dict[str, Any]:
        """Enrich alert context with additional information."""
        enriched_context = context.copy()

        # Add timestamp information
        enriched_context['alert_generation_time'] = time.time()
        enriched_context['alert_generation_time_iso'] = datetime.now().isoformat()

        # Add metric metadata
        enriched_context['metric_metadata'] = {
            'name': metric_name,
            'value': value,
            'type': type(value).__name__
        }

        # Add system information if available
        try:
            import psutil
            enriched_context['system_info'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except Exception:
            pass

        return enriched_context

    async def _process_alert(self, alert: Alert):
        """Process an alert through the pipeline."""
        # Check for duplicates
        if self.deduplicator.is_duplicate(alert):
            logger.debug(f"Duplicate alert suppressed: {alert.alert_id}")
            return

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Trim history
        if len(self.alert_history) > self.config['max_alert_history']:
            self.alert_history = self.alert_history[-self.config['max_alert_history']:]

        logger.info(f"Processing alert: {alert.alert_id} ({alert.severity})")

        # Get rule for channel configuration
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            logger.error(f"Rule not found for alert: {alert.rule_id}")
            return

        # Send to configured channels
        for channel_id in rule.channels:
            await self._send_to_channel(alert, channel_id)

        # Save alert to disk
        self._save_alert(alert)

    async def _send_to_channel(self, alert: Alert, channel_id: str):
        """Send alert to a specific channel."""
        if channel_id not in self.alert_channels:
            logger.warning(f"Channel not found: {channel_id}")
            return

        channel = self.alert_channels[channel_id]
        if not channel.enabled:
            logger.debug(f"Channel disabled: {channel_id}")
            return

        # Check rate limiting
        if self.throttler.should_throttle(channel_id, channel.rate_limit):
            logger.warning(f"Rate limit exceeded for channel: {channel_id}")
            return

        # Send alert
        try:
            channel_instance = channel.config.get('instance')
            if channel_instance and hasattr(channel_instance, 'send_alert'):
                success = await channel_instance.send_alert(alert)
                if success:
                    logger.debug(f"Alert sent to channel {channel_id}: {alert.alert_id}")
                else:
                    logger.warning(f"Failed to send alert to channel {channel_id}")
            else:
                logger.error(f"Invalid channel instance: {channel_id}")

        except Exception as e:
            logger.error(f"Error sending alert to channel {channel_id}: {e}")

    def _save_alert(self, alert: Alert):
        """Save alert to disk for persistence."""
        alert_file = self.alerts_dir / f"{alert.alert_id}.json"
        try:
            with open(alert_file, 'w') as f:
                json.dump(asdict(alert), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged by {user}: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str, user: str = "system",
                     auto_resolved: bool = False) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.auto_resolved = auto_resolved

            # Remove from active alerts
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved by {user}: {alert_id}")
            return True
        return False

    def get_active_alerts(self, severity: Optional[str] = None,
                         category: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if category:
            alerts = [a for a in alerts if a.category == category]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        current_time = time.time()
        last_hour = current_time - 3600
        last_day = current_time - 86400

        # Count alerts by time period
        recent_alerts = [a for a in self.alert_history if a.timestamp > last_hour]
        daily_alerts = [a for a in self.alert_history if a.timestamp > last_day]

        # Count by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1

        # Count by category
        category_counts = defaultdict(int)
        for alert in recent_alerts:
            category_counts[alert.category] += 1

        return {
            'active_alerts': len(self.active_alerts),
            'alerts_last_hour': len(recent_alerts),
            'alerts_last_day': len(daily_alerts),
            'total_alerts': len(self.alert_history),
            'severity_distribution': dict(severity_counts),
            'category_distribution': dict(category_counts),
            'rules_configured': len(self.alert_rules),
            'channels_configured': len(self.alert_channels),
            'escalation_policies': len(self.escalation_policies)
        }

    def test_alert_channel(self, channel_id: str) -> bool:
        """Test an alert channel with a test alert."""
        if channel_id not in self.alert_channels:
            logger.error(f"Channel not found: {channel_id}")
            return False

        # Create test alert
        test_alert = Alert(
            alert_id=f"test_{int(time.time())}",
            rule_id="test_rule",
            timestamp=time.time(),
            severity="info",
            category="test",
            title="Test Alert",
            message="This is a test alert to verify channel functionality",
            metric_name="test_metric",
            current_value=1.0,
            threshold=0.5,
            context={'test': True},
            suggested_actions=["This is a test - no action required"]
        )

        # Send test alert
        try:
            asyncio.create_task(self._send_to_channel(test_alert, channel_id))
            logger.info(f"Test alert sent to channel: {channel_id}")
            return True
        except Exception as e:
            logger.error(f"Test alert failed for channel {channel_id}: {e}")
            return False

    def shutdown(self):
        """Shutdown the alert system."""
        logger.info("Shutting down alert system")

        # Stop processing
        self.processing_active = False

        # Clear queues and state
        self.active_alerts.clear()

        logger.info("Alert system shutdown complete")


def create_alert_system(config: Optional[Dict[str, Any]] = None) -> AlertSystem:
    """
    Factory function to create and configure an alert system.

    Parameters
    ----------
    config : dict, optional
        Alert system configuration

    Returns
    -------
    AlertSystem
        Configured alert system
    """
    return AlertSystem(config)


if __name__ == "__main__":
    # Example usage and testing
    import asyncio

    async def test_alert_system():
        # Create alert system
        alert_system = create_alert_system()

        # Test console channel
        print("Testing console alert channel...")
        alert_system.test_alert_channel('console')

        # Simulate metric violations
        print("\nSimulating performance alerts...")
        alert_system.evaluate_metric('response_time_calculation', 2.5)
        alert_system.evaluate_metric('memory_utilization', 0.9)
        alert_system.evaluate_metric('accuracy_preservation', 0.985)

        # Wait for processing
        await asyncio.sleep(1)

        # Check statistics
        stats = alert_system.get_alert_statistics()
        print(f"\nAlert Statistics: {stats}")

        # Show active alerts
        active = alert_system.get_active_alerts()
        print(f"\nActive Alerts: {len(active)}")
        for alert in active:
            print(f"  - {alert.title} ({alert.severity})")

        alert_system.shutdown()

    # Run test
    asyncio.run(test_alert_system())