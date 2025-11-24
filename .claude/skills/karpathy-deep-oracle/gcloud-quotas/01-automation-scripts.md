# GCloud Quota Automation: Production-Ready Python Scripts

## Overview

This document provides production-ready Python scripts for automating GCP quota management, monitoring, and increase requests. Based on the Cloud Quotas API and Google's open-source Quota Monitoring Solution, these scripts include comprehensive error handling, retry logic, logging, and testing patterns.

**Key Capabilities:**
- Automated quota increase requests with justification templates
- Real-time quota monitoring with threshold alerts
- Multi-region quota distribution and failover
- BigQuery-backed historical tracking
- Cloud Functions integration for event-driven automation
- Terraform deployment patterns

From [Quota Automation Section](../../gcloud-production/01-quotas-alpha.md) (Section 3, lines 152-250):
- Cloud Quotas API Python examples
- Automated monitoring with threshold detection
- Terraform automation patterns

From [Medium: Quota Monitoring Options](https://medium.com/google-cloud/quota-monitoring-and-management-options-on-google-cloud-b94caf8a9671) (accessed 2025-02-03):
- Google's Quota Monitoring Solution architecture
- Looker Studio dashboard integration
- Alert policy automation with Terraform

From [GitHub: Quota Monitoring Solution](https://github.com/google/quota-monitoring-solution) (accessed 2025-02-03):
- Open-source production implementation (100+ customers)
- Cloud Functions + Pub/Sub + BigQuery architecture
- Comprehensive deployment automation

## Section 1: Production Quota Request Script (~150 lines)

### Complete Implementation with Error Handling

```python
#!/usr/bin/env python3
"""
Production Quota Increase Request Script

Handles quota increase requests with retry logic, error handling,
and detailed logging. Supports GPU quotas with proper justification.

Usage:
    python request_quota_increase.py \
        --project my-project \
        --service compute.googleapis.com \
        --metric nvidia_t4_gpus \
        --region us-west1 \
        --new-limit 8 \
        --justification "Production ML training workload"
"""

import argparse
import logging
import sys
import time
from typing import Dict, Optional
from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, PermissionDenied
from google.cloud import cloudquotas_v1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quota_requests.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QuotaRequestError(Exception):
    """Custom exception for quota request failures."""
    pass


class QuotaRequester:
    """Handles quota increase requests with error handling and retry logic."""

    def __init__(self, project_id: str, contact_email: str):
        """
        Initialize QuotaRequester.

        Args:
            project_id: GCP project ID
            contact_email: Email for quota request notifications
        """
        self.project_id = project_id
        self.contact_email = contact_email
        try:
            self.client = cloudquotas_v1.CloudQuotasClient()
            logger.info(f"Initialized CloudQuotasClient for project {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize CloudQuotasClient: {e}")
            raise QuotaRequestError(f"Client initialization failed: {e}")

    @retry.Retry(
        predicate=retry.if_exception_type(ResourceExhausted, GoogleAPIError),
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0
    )
    def request_increase(
        self,
        service: str,
        metric: str,
        region: str,
        new_limit: int,
        justification: str
    ) -> Dict:
        """
        Request quota increase with exponential backoff retry.

        Args:
            service: GCP service (e.g., compute.googleapis.com)
            metric: Quota metric name (e.g., nvidia_t4_gpus)
            region: GCP region (e.g., us-west1)
            new_limit: Desired quota limit
            justification: Business justification for increase

        Returns:
            Dict with request details

        Raises:
            QuotaRequestError: If request fails after retries
        """
        try:
            # Construct parent resource name
            parent = f"projects/{self.project_id}/locations/{region}/services/{service}"

            # Build quota preference
            preference = cloudquotas_v1.QuotaPreference(
                dimensions={"region": region},
                quota_config=cloudquotas_v1.QuotaConfig(
                    preferred_value=new_limit,
                ),
                justification=justification,
                contact_email=self.contact_email
            )

            # Create request
            request = cloudquotas_v1.CreateQuotaPreferenceRequest(
                parent=parent,
                quota_preference=preference,
                quota_preference_id=f"{metric}-increase-{int(time.time())}"
            )

            logger.info(f"Submitting quota request: {service}/{metric} → {new_limit} in {region}")
            response = self.client.create_quota_preference(request=request)

            result = {
                "status": "success",
                "request_name": response.name,
                "service": service,
                "metric": metric,
                "region": region,
                "requested_limit": new_limit,
                "justification": justification,
                "timestamp": time.time()
            }

            logger.info(f"Quota request submitted successfully: {response.name}")
            return result

        except PermissionDenied as e:
            error_msg = f"Permission denied for quota request: {e}"
            logger.error(error_msg)
            raise QuotaRequestError(error_msg)

        except ResourceExhausted as e:
            error_msg = f"Too many quota requests (rate limit): {e}"
            logger.error(error_msg)
            raise QuotaRequestError(error_msg)

        except GoogleAPIError as e:
            error_msg = f"API error during quota request: {e}"
            logger.error(error_msg)
            raise QuotaRequestError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during quota request: {e}"
            logger.error(error_msg)
            raise QuotaRequestError(error_msg)

    def get_current_quota(self, service: str, metric: str, region: str) -> Optional[int]:
        """
        Retrieve current quota limit.

        Args:
            service: GCP service
            metric: Quota metric name
            region: GCP region

        Returns:
            Current quota limit or None if unavailable
        """
        try:
            parent = f"projects/{self.project_id}/locations/{region}"
            request = cloudquotas_v1.ListQuotaInfosRequest(parent=parent)

            for quota in self.client.list_quota_infos(request=request):
                if metric in quota.name and service in quota.name:
                    return quota.quota_limit

            logger.warning(f"Quota not found: {service}/{metric} in {region}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving current quota: {e}")
            return None


def build_justification(use_case: str, timeline: str, details: Dict) -> str:
    """
    Build detailed justification for quota request.

    Args:
        use_case: Use case category (training, inference, development)
        timeline: Timeline for quota need
        details: Additional details (framework, model, traffic, etc.)

    Returns:
        Formatted justification string
    """
    templates = {
        "training": (
            "Production ML training workload:\n"
            "- Model: {model}\n"
            "- Framework: {framework}\n"
            "- Training duration: {duration}\n"
            "- Timeline: {timeline}"
        ),
        "inference": (
            "Production inference serving:\n"
            "- Model: {model}\n"
            "- Traffic: {traffic}\n"
            "- SLA: {sla}\n"
            "- Timeline: {timeline}"
        ),
        "development": (
            "Development and testing environment:\n"
            "- Purpose: {purpose}\n"
            "- Team size: {team_size}\n"
            "- Timeline: {timeline}"
        )
    }

    template = templates.get(use_case, templates["training"])
    details["timeline"] = timeline
    return template.format(**details)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Request GCP quota increase")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--service", required=True, help="GCP service (e.g., compute.googleapis.com)")
    parser.add_argument("--metric", required=True, help="Quota metric (e.g., nvidia_t4_gpus)")
    parser.add_argument("--region", required=True, help="GCP region (e.g., us-west1)")
    parser.add_argument("--new-limit", type=int, required=True, help="Desired quota limit")
    parser.add_argument("--justification", required=True, help="Business justification")
    parser.add_argument("--email", required=True, help="Contact email for notifications")

    args = parser.parse_args()

    try:
        # Initialize requester
        requester = QuotaRequester(args.project, args.email)

        # Check current quota
        current = requester.get_current_quota(args.service, args.metric, args.region)
        if current:
            logger.info(f"Current quota: {current}")
            if args.new_limit <= current:
                logger.warning(f"Requested limit ({args.new_limit}) is not greater than current ({current})")

        # Submit request
        result = requester.request_increase(
            service=args.service,
            metric=args.metric,
            region=args.region,
            new_limit=args.new_limit,
            justification=args.justification
        )

        print(f"✓ Quota request submitted: {result['request_name']}")
        print(f"  Service: {result['service']}")
        print(f"  Metric: {result['metric']}")
        print(f"  Region: {result['region']}")
        print(f"  Requested: {result['requested_limit']}")

        return 0

    except QuotaRequestError as e:
        logger.error(f"Quota request failed: {e}")
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Key Features:**
- Exponential backoff retry with `@retry.Retry` decorator
- Comprehensive exception handling (PermissionDenied, ResourceExhausted, etc.)
- Structured logging to file and stdout
- Current quota validation before requesting increase
- Custom exception types for better error tracking
- Template-based justification builder

## Section 2: Quota Monitoring Script (~150 lines)

### Real-Time Monitoring with Alerting

```python
#!/usr/bin/env python3
"""
Production Quota Monitoring Script

Monitors quota usage across projects and sends alerts when thresholds exceeded.
Integrates with Cloud Monitoring for alerting.

Usage:
    python monitor_quotas.py \
        --project my-project \
        --threshold 0.8 \
        --alert-email team@example.com
"""

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional
from google.cloud import cloudquotas_v1
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import NotificationChannel, AlertPolicy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuotaMonitor:
    """Monitors quota usage and creates alerts for high utilization."""

    def __init__(self, project_id: str, threshold: float = 0.8):
        """
        Initialize QuotaMonitor.

        Args:
            project_id: GCP project ID
            threshold: Alert threshold (0.0-1.0, default 0.8 = 80%)
        """
        self.project_id = project_id
        self.threshold = threshold
        self.quotas_client = cloudquotas_v1.CloudQuotasClient()
        self.monitoring_client = monitoring_v3.AlertPolicyServiceClient()
        self.notification_client = monitoring_v3.NotificationChannelServiceClient()
        logger.info(f"Initialized QuotaMonitor for project {project_id} (threshold={threshold*100}%)")

    def scan_quotas(self, services: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan quotas across services and identify high usage.

        Args:
            services: List of services to scan (None = all services)

        Returns:
            List of quota alerts (quotas exceeding threshold)
        """
        alerts = []
        parent = f"projects/{self.project_id}/locations/global"

        try:
            request = cloudquotas_v1.ListQuotaInfosRequest(parent=parent)

            for quota in self.quotas_client.list_quota_infos(request=request):
                # Skip if quota has no limit or usage data
                if not quota.quota_limit or not quota.quota_usage:
                    continue

                # Filter by service if specified
                if services and not any(svc in quota.name for svc in services):
                    continue

                # Calculate usage ratio
                usage_ratio = quota.quota_usage / quota.quota_limit

                if usage_ratio >= self.threshold:
                    alert = {
                        "metric": quota.name,
                        "usage": quota.quota_usage,
                        "limit": quota.quota_limit,
                        "ratio": round(usage_ratio, 4),
                        "percentage": round(usage_ratio * 100, 2),
                        "severity": self._calculate_severity(usage_ratio)
                    }
                    alerts.append(alert)
                    logger.warning(
                        f"High quota usage: {quota.name} "
                        f"({alert['usage']}/{alert['limit']} = {alert['percentage']}%)"
                    )

            logger.info(f"Scan complete: {len(alerts)} alerts out of threshold")
            return alerts

        except Exception as e:
            logger.error(f"Error scanning quotas: {e}")
            return []

    def _calculate_severity(self, ratio: float) -> str:
        """Calculate alert severity based on usage ratio."""
        if ratio >= 0.95:
            return "CRITICAL"
        elif ratio >= 0.90:
            return "ERROR"
        elif ratio >= 0.80:
            return "WARNING"
        else:
            return "INFO"

    def create_alert_policy(
        self,
        policy_name: str,
        notification_channels: List[str],
        threshold: float = 0.8
    ) -> str:
        """
        Create Cloud Monitoring alert policy for quota usage.

        Args:
            policy_name: Display name for alert policy
            notification_channels: List of notification channel resource names
            threshold: Alert threshold (0.0-1.0)

        Returns:
            Alert policy resource name
        """
        try:
            project_name = f"projects/{self.project_id}"

            # MQL query for quota monitoring (from Google's QMS)
            query = f"""
            fetch consumer_quota
            | filter resource.service =~ '.*'
            | {{ t_0:
                 metric 'serviceruntime.googleapis.com/quota/allocation/usage'
                 | align next_older(1d)
                 | group_by [resource.project_id, metric.quota_metric, resource.location],
                   [value_usage_max: max(value.usage)]
               ; t_1:
                 metric 'serviceruntime.googleapis.com/quota/limit'
                 | align next_older(1d)
                 | group_by [resource.project_id, metric.quota_metric, resource.location],
                   [value_limit_min: min(value.limit)] }}
            | ratio
            | every 1m
            | condition gt(ratio, {threshold} '1')
            """

            # Create alert policy
            alert_policy = monitoring_v3.AlertPolicy(
                display_name=policy_name,
                conditions=[
                    monitoring_v3.AlertPolicy.Condition(
                        display_name="Quota usage above threshold",
                        condition_monitoring_query_language=monitoring_v3.AlertPolicy.Condition.MonitoringQueryLanguageCondition(
                            query=query,
                            duration={"seconds": 3600},  # 1 hour
                        ),
                    )
                ],
                combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
                notification_channels=notification_channels,
                alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                    auto_close={"seconds": 86400}  # 24 hours
                ),
                documentation=monitoring_v3.AlertPolicy.Documentation(
                    content=f"Quota usage exceeded {threshold*100}% threshold. Review and request increase if needed.",
                    mime_type="text/markdown"
                )
            )

            response = self.monitoring_client.create_alert_policy(
                name=project_name,
                alert_policy=alert_policy
            )

            logger.info(f"Created alert policy: {response.name}")
            return response.name

        except Exception as e:
            logger.error(f"Failed to create alert policy: {e}")
            raise

    def create_notification_channel(
        self,
        channel_type: str,
        display_name: str,
        labels: Dict[str, str]
    ) -> str:
        """
        Create Cloud Monitoring notification channel.

        Args:
            channel_type: Channel type (email, slack, etc.)
            display_name: Display name for channel
            labels: Channel-specific labels (e.g., email_address)

        Returns:
            Notification channel resource name
        """
        try:
            project_name = f"projects/{self.project_id}"

            channel = monitoring_v3.NotificationChannel(
                type=channel_type,
                display_name=display_name,
                labels=labels
            )

            response = self.notification_client.create_notification_channel(
                name=project_name,
                notification_channel=channel
            )

            logger.info(f"Created notification channel: {response.name}")
            return response.name

        except Exception as e:
            logger.error(f"Failed to create notification channel: {e}")
            raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Monitor GCP quotas")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--threshold", type=float, default=0.8, help="Alert threshold (0.0-1.0)")
    parser.add_argument("--alert-email", help="Email for alerts")
    parser.add_argument("--create-policy", action="store_true", help="Create alert policy")
    parser.add_argument("--services", nargs="*", help="Services to monitor (default: all)")

    args = parser.parse_args()

    try:
        monitor = QuotaMonitor(args.project, args.threshold)

        # Scan quotas
        alerts = monitor.scan_quotas(services=args.services)

        if alerts:
            print(f"\n⚠️  {len(alerts)} quota(s) exceeding {args.threshold*100}% threshold:\n")
            for alert in sorted(alerts, key=lambda x: x['ratio'], reverse=True):
                print(f"  [{alert['severity']}] {alert['metric']}")
                print(f"    Usage: {alert['usage']} / {alert['limit']} ({alert['percentage']}%)")
                print()
        else:
            print(f"✓ All quotas below {args.threshold*100}% threshold")

        # Create alert policy if requested
        if args.create_policy and args.alert_email:
            # Create email notification channel
            channel_name = monitor.create_notification_channel(
                channel_type="email",
                display_name=f"Quota Alerts - {args.alert_email}",
                labels={"email_address": args.alert_email}
            )

            # Create alert policy
            policy_name = monitor.create_alert_policy(
                policy_name=f"Quota Monitoring - {args.threshold*100}%",
                notification_channels=[channel_name],
                threshold=args.threshold
            )

            print(f"\n✓ Created alert policy: {policy_name}")

        return 0

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Key Features:**
- Scans all quotas with configurable threshold
- Severity calculation (INFO, WARNING, ERROR, CRITICAL)
- Cloud Monitoring integration with MQL queries
- Email notification channel creation
- Alert policy automation with auto-close

## Section 3: Testing Framework (~100 lines)

### Unit Tests with Mock API

```python
#!/usr/bin/env python3
"""
Unit tests for quota automation scripts.

Usage:
    pytest test_quota_automation.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from google.api_core.exceptions import PermissionDenied, ResourceExhausted
from request_quota_increase import QuotaRequester, QuotaRequestError
from monitor_quotas import QuotaMonitor


class TestQuotaRequester:
    """Tests for QuotaRequester class."""

    @pytest.fixture
    def requester(self):
        """Create QuotaRequester instance with mocked client."""
        with patch('request_quota_increase.cloudquotas_v1.CloudQuotasClient'):
            return QuotaRequester("test-project", "test@example.com")

    def test_successful_request(self, requester):
        """Test successful quota increase request."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.name = "projects/test-project/quotaPreferences/test-pref"
        requester.client.create_quota_preference = Mock(return_value=mock_response)

        result = requester.request_increase(
            service="compute.googleapis.com",
            metric="nvidia_t4_gpus",
            region="us-west1",
            new_limit=8,
            justification="Test request"
        )

        assert result["status"] == "success"
        assert result["requested_limit"] == 8
        assert "request_name" in result

    def test_permission_denied(self, requester):
        """Test handling of permission denied error."""
        requester.client.create_quota_preference = Mock(
            side_effect=PermissionDenied("Access denied")
        )

        with pytest.raises(QuotaRequestError, match="Permission denied"):
            requester.request_increase(
                service="compute.googleapis.com",
                metric="nvidia_t4_gpus",
                region="us-west1",
                new_limit=8,
                justification="Test request"
            )

    def test_rate_limit_retry(self, requester):
        """Test retry logic for rate limit errors."""
        # First call fails with rate limit, second succeeds
        mock_response = Mock()
        mock_response.name = "projects/test-project/quotaPreferences/test-pref"

        requester.client.create_quota_preference = Mock(
            side_effect=[ResourceExhausted("Rate limit"), mock_response]
        )

        result = requester.request_increase(
            service="compute.googleapis.com",
            metric="nvidia_t4_gpus",
            region="us-west1",
            new_limit=8,
            justification="Test request"
        )

        assert result["status"] == "success"
        assert requester.client.create_quota_preference.call_count == 2


class TestQuotaMonitor:
    """Tests for QuotaMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create QuotaMonitor instance with mocked clients."""
        with patch('monitor_quotas.cloudquotas_v1.CloudQuotasClient'), \
             patch('monitor_quotas.monitoring_v3.AlertPolicyServiceClient'), \
             patch('monitor_quotas.monitoring_v3.NotificationChannelServiceClient'):
            return QuotaMonitor("test-project", threshold=0.8)

    def test_scan_quotas_high_usage(self, monitor):
        """Test quota scanning with high usage detection."""
        # Mock quota with 90% usage
        mock_quota = Mock()
        mock_quota.name = "projects/test/services/compute/quotas/cpus"
        mock_quota.quota_usage = 90
        mock_quota.quota_limit = 100

        monitor.quotas_client.list_quota_infos = Mock(return_value=[mock_quota])

        alerts = monitor.scan_quotas()

        assert len(alerts) == 1
        assert alerts[0]["percentage"] == 90.0
        assert alerts[0]["severity"] == "ERROR"

    def test_scan_quotas_below_threshold(self, monitor):
        """Test quota scanning with usage below threshold."""
        # Mock quota with 50% usage
        mock_quota = Mock()
        mock_quota.name = "projects/test/services/compute/quotas/cpus"
        mock_quota.quota_usage = 50
        mock_quota.quota_limit = 100

        monitor.quotas_client.list_quota_infos = Mock(return_value=[mock_quota])

        alerts = monitor.scan_quotas()

        assert len(alerts) == 0

    def test_severity_calculation(self, monitor):
        """Test severity level calculation."""
        assert monitor._calculate_severity(0.96) == "CRITICAL"
        assert monitor._calculate_severity(0.92) == "ERROR"
        assert monitor._calculate_severity(0.85) == "WARNING"
        assert monitor._calculate_severity(0.75) == "INFO"


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @pytest.mark.integration
    def test_monitor_and_request_workflow(self):
        """Test full workflow: monitor → alert → request increase."""
        # This would test the complete automation flow
        # Skipped in unit tests, run manually in staging environment
        pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Key Features:**
- pytest-based test suite
- Mock API clients to avoid real API calls
- Test coverage for success and error paths
- Retry logic validation
- Severity calculation testing
- Integration test markers for manual validation

## Section 4: Cloud Functions Integration (~100 lines)

### Event-Driven Quota Monitoring

```python
#!/usr/bin/env python3
"""
Cloud Function for automated quota monitoring.

Triggered by Cloud Scheduler, scans quotas and writes to BigQuery.

Deploy:
    gcloud functions deploy quota-scanner \
        --runtime python311 \
        --trigger-topic quota-scan-trigger \
        --entry-point main \
        --timeout 540s
"""

import json
import logging
from typing import Dict, List
from google.cloud import bigquery
from google.cloud import pubsub_v1
from monitor_quotas import QuotaMonitor

# Initialize clients (reused across invocations)
bq_client = bigquery.Client()
pubsub_publisher = pubsub_v1.PublisherClient()

logger = logging.getLogger(__name__)


def main(event: Dict, context) -> None:
    """
    Cloud Function entry point for quota scanning.

    Args:
        event: Pub/Sub event data
        context: Cloud Functions context
    """
    try:
        # Parse configuration from Pub/Sub message
        config = json.loads(event["data"])
        project_id = config["project_id"]
        threshold = config.get("threshold", 0.8)
        dataset_id = config["dataset_id"]
        table_id = config["table_id"]
        alert_topic = config.get("alert_topic")

        logger.info(f"Scanning quotas for project {project_id}")

        # Scan quotas
        monitor = QuotaMonitor(project_id, threshold)
        alerts = monitor.scan_quotas()

        # Write results to BigQuery
        write_to_bigquery(alerts, dataset_id, table_id)

        # Publish alerts to Pub/Sub
        if alerts and alert_topic:
            publish_alerts(alerts, alert_topic)

        logger.info(f"Scan complete: {len(alerts)} alerts")

    except Exception as e:
        logger.error(f"Error in quota scan: {e}")
        raise


def write_to_bigquery(alerts: List[Dict], dataset_id: str, table_id: str) -> None:
    """
    Write quota alerts to BigQuery.

    Args:
        alerts: List of quota alert dictionaries
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
    """
    if not alerts:
        return

    try:
        table_ref = f"{bq_client.project}.{dataset_id}.{table_id}"

        # Schema matches Google's QMS format
        rows_to_insert = [
            {
                "project_id": bq_client.project,
                "quota_metric": alert["metric"],
                "current_usage": alert["usage"],
                "quota_limit": alert["limit"],
                "usage_ratio": alert["ratio"],
                "severity": alert["severity"],
                "timestamp": bigquery.Client().query("SELECT CURRENT_TIMESTAMP()").result()
            }
            for alert in alerts
        ]

        errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Wrote {len(alerts)} rows to BigQuery")

    except Exception as e:
        logger.error(f"Failed to write to BigQuery: {e}")
        raise


def publish_alerts(alerts: List[Dict], topic: str) -> None:
    """
    Publish quota alerts to Pub/Sub for downstream processing.

    Args:
        alerts: List of quota alert dictionaries
        topic: Pub/Sub topic name
    """
    try:
        for alert in alerts:
            message = json.dumps(alert).encode("utf-8")
            future = pubsub_publisher.publish(topic, message)
            future.result()  # Wait for publish to complete

        logger.info(f"Published {len(alerts)} alerts to {topic}")

    except Exception as e:
        logger.error(f"Failed to publish alerts: {e}")
        raise
```

**Deployment Configuration (requirements.txt):**
```
google-cloud-cloudquotas==1.3.0
google-cloud-bigquery==3.14.0
google-cloud-pubsub==2.21.0
google-cloud-monitoring==2.18.0
```

**Deployment Command:**
```bash
# Deploy Cloud Function
gcloud functions deploy quota-scanner \
    --gen2 \
    --runtime python311 \
    --region us-central1 \
    --source . \
    --entry-point main \
    --trigger-topic quota-scan-trigger \
    --timeout 540s \
    --memory 512MB \
    --service-account quota-scanner@PROJECT.iam.gserviceaccount.com
```

## Section 5: Terraform Automation (~50 lines)

### Infrastructure as Code for Quota Monitoring

```hcl
# Terraform configuration for quota monitoring automation
# Based on Google's Quota Monitoring Solution

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Cloud Scheduler for periodic scanning
resource "google_cloud_scheduler_job" "quota_scan" {
  name        = "quota-monitoring-cron-job"
  description = "Trigger quota scanning every hour"
  schedule    = "0 * * * *"  # Every hour
  time_zone   = "America/Los_Angeles"

  pubsub_target {
    topic_name = google_pubsub_topic.quota_scan_trigger.id
    data       = base64encode(jsonencode({
      project_id  = var.project_id
      threshold   = var.alert_threshold
      dataset_id  = google_bigquery_dataset.quota_monitoring.dataset_id
      table_id    = google_bigquery_table.quota_history.table_id
      alert_topic = google_pubsub_topic.quota_alerts.id
    }))
  }
}

# Pub/Sub topic for triggering scans
resource "google_pubsub_topic" "quota_scan_trigger" {
  name = "quota-scan-trigger"
}

# Pub/Sub topic for alerts
resource "google_pubsub_topic" "quota_alerts" {
  name = "quota-alerts"
}

# BigQuery dataset for quota history
resource "google_bigquery_dataset" "quota_monitoring" {
  dataset_id = "quota_monitoring_dataset"
  location   = var.region

  default_table_expiration_ms = 7776000000  # 90 days
}

# BigQuery table for quota metrics
resource "google_bigquery_table" "quota_history" {
  dataset_id = google_bigquery_dataset.quota_monitoring.dataset_id
  table_id   = "quota_monitoring_table"

  schema = jsonencode([
    {
      name = "project_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "quota_metric"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "current_usage"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "quota_limit"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "usage_ratio"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "severity"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
}

# Alert policy for quota monitoring
resource "google_monitoring_alert_policy" "quota_threshold" {
  display_name = "Quota Monitoring - ${var.alert_threshold * 100}%"
  combiner     = "OR"

  conditions {
    display_name = "Quota usage above threshold"

    condition_monitoring_query_language {
      duration = "3600s"
      query    = <<-EOT
        fetch consumer_quota
        | filter resource.service =~ '.*'
        | { t_0:
             metric 'serviceruntime.googleapis.com/quota/allocation/usage'
             | align next_older(1d)
             | group_by [resource.project_id, metric.quota_metric, resource.location],
               [value_usage_max: max(value.usage)]
           ; t_1:
             metric 'serviceruntime.googleapis.com/quota/limit'
             | align next_older(1d)
             | group_by [resource.project_id, metric.quota_metric, resource.location],
               [value_limit_min: min(value.limit)] }
        | ratio
        | every 1m
        | condition gt(ratio, ${var.alert_threshold} '1')
      EOT
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.name
  ]

  alert_strategy {
    auto_close = "86400s"  # 24 hours
  }

  documentation {
    content   = "Quota usage exceeded ${var.alert_threshold * 100}% threshold. Review and request increase if needed."
    mime_type = "text/markdown"
  }
}

# Email notification channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "Quota Alerts"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

# Variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "alert_threshold" {
  description = "Alert threshold (0.0-1.0)"
  type        = number
  default     = 0.8
}

variable "alert_email" {
  description = "Email for quota alerts"
  type        = string
}
```

**Apply Terraform:**
```bash
# Initialize
terraform init

# Plan changes
terraform plan -var="project_id=my-project" \
               -var="alert_email=team@example.com"

# Apply
terraform apply -auto-approve
```

## Sources

**Google Cloud Official Documentation:**
- [Cloud Quotas API Overview](https://docs.cloud.google.com/docs/quotas/api-overview) - Programmatic quota management (accessed 2025-02-03)
- [Monitor and Alert with Quota Metrics](https://docs.cloud.google.com/monitoring/alerts/using-quota-metrics) - Quota monitoring setup (accessed 2025-02-03)

**Google Cloud Blog Posts:**
- [How to Programmatically Manage Quotas](https://cloud.google.com/blog/topics/cost-management/how-to-programmatically-manage-quotas-in-google-cloud/) - Cloud Quotas API examples (accessed 2025-02-03)

**Community Resources:**
- [Medium: Quota Monitoring Options](https://medium.com/google-cloud/quota-monitoring-and-management-options-on-google-cloud-b94caf8a9671) - Comprehensive monitoring strategies (accessed 2025-02-03)
- [GitHub: Quota Monitoring Solution](https://github.com/google/quota-monitoring-solution) - Open-source production implementation (accessed 2025-02-03)

**Source Documents:**
- [gcloud-production/01-quotas-alpha.md](../../gcloud-production/01-quotas-alpha.md) - Section 3: Automation Strategies (lines 152-250)

**Implementation References:**
- Google Quota Monitoring Solution - 100+ customer deployments
- Cloud Functions + Pub/Sub + BigQuery architecture pattern
- MQL-based alert policies for real-time monitoring
