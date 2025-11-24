# Cloud Logging Advanced Techniques

## Overview

Advanced Cloud Logging capabilities enable sophisticated log analysis, cost optimization, and operational insights. This guide covers advanced query syntax, log sinks for long-term storage, structured logging patterns, and cost optimization strategies essential for production systems.

**Key capabilities:**
- Advanced query language with pipe syntax (2024 feature)
- Log sinks to BigQuery, Cloud Storage, Pub/Sub
- Structured logging for enhanced searchability
- Cost optimization techniques
- Integration with Error Reporting and monitoring

From web research on Cloud Logging documentation and best practices (accessed 2025-02-03).

---

## Section 1: Advanced Query Language (~150 lines)

### Logging Query Language Basics

Cloud Logging uses a powerful query language for filtering and analyzing log data:

**Basic query structure:**
```
resource.type="gce_instance"
severity>=ERROR
timestamp>="2024-01-01T00:00:00Z"
```

**Field types:**
- **Log fields**: `textPayload`, `jsonPayload`, `protoPayload`
- **Resource fields**: `resource.type`, `resource.labels.*`
- **System fields**: `timestamp`, `severity`, `logName`, `insertId`

From [Logging Query Language documentation](https://docs.cloud.google.com/logging/docs/view/logging-query-language) (accessed 2025-02-03).

### Pipe Syntax (2024 Feature)

**Revolutionary new query syntax** introduced October 2024:

**Traditional syntax:**
```
resource.type="k8s_container"
severity>=WARNING
jsonPayload.error_code="500"
```

**Pipe syntax (new):**
```
resource.type="k8s_container"
|> filter severity >= WARNING
|> filter jsonPayload.error_code = "500"
|> aggregate count() by resource.labels.pod_name
```

**Key advantages:**
- Linear, top-down structure (like Unix pipes)
- Easier to read and maintain
- Natural aggregation flow
- Compatible with BigQuery's pipe syntax

**Common pipe operators:**
- `|> filter` - Filter log entries
- `|> aggregate` - Aggregate and group results
- `|> limit` - Limit number of results
- `|> fields` - Select specific fields

From [Introducing pipe syntax in BigQuery and Cloud Logging](https://cloud.google.com/blog/products/data-analytics/introducing-pipe-syntax-in-bigquery-and-cloud-logging) (accessed 2025-02-03).

### Advanced Query Examples

**Example 1: Error rate by service**
```
resource.type="cloud_run_revision"
|> filter severity >= ERROR
|> aggregate count() by resource.labels.service_name, timestamp_trunc(timestamp, HOUR)
|> fields timestamp, service_name, error_count
|> order by error_count desc
|> limit 100
```

**Example 2: Slow queries with regex**
```
resource.type="cloudsql_database"
|> filter textPayload =~ "Query took.*[5-9][0-9]{3}ms"
|> aggregate avg(duration) by resource.labels.database_id
```

**Example 3: Find specific error patterns**
```
jsonPayload.message =~ "timeout|connection refused|OOM"
severity = ERROR
timestamp >= timestamp_sub(timestamp("now"), interval 1 hour)
```

**Example 4: Aggregate by multiple dimensions**
```
resource.type="k8s_container"
|> filter severity >= WARNING
|> aggregate count(),
            avg(jsonPayload.latency_ms),
            max(jsonPayload.memory_mb)
   by resource.labels.namespace_name,
      resource.labels.pod_name
|> order by count desc
```

### Wildcards and Regex

**Text search operators:**
- `:` - Contains (case-insensitive)
- `=~` - Regex match
- `!~` - Regex not match

**Examples:**
```
# Contains search
text:"connection timeout"

# Regex patterns
textPayload =~ "Error: [0-9]{3,}"

# Negative patterns
textPayload !~ "DEBUG|INFO"
```

### Time-Based Filtering

**Timestamp operations:**
```
# Relative time
timestamp >= timestamp_sub(timestamp("now"), interval 30 minute)

# Specific range
timestamp >= "2024-01-15T00:00:00Z"
timestamp < "2024-01-16T00:00:00Z"

# Truncate for grouping
timestamp_trunc(timestamp, HOUR)
timestamp_trunc(timestamp, DAY)
```

### Finding Log Entries Quickly

**Performance tips:**
1. **Always specify resource type** - Dramatically reduces search space
2. **Use time bounds** - Narrow time window first
3. **Index-backed fields** - Use `severity`, `resource.*`, `labels.*`
4. **Avoid full-text search** - Use structured fields when possible

**Fast query pattern:**
```
resource.type="gce_instance"  # Indexed, fast
resource.labels.instance_id="12345"  # Indexed, fast
severity>=ERROR  # Indexed, fast
timestamp>="2024-02-01T00:00:00Z"  # Indexed, fast
```

**Slow query pattern (avoid):**
```
textPayload =~ ".*error.*"  # Full-text scan, slow
```

From [Guide to GCP's Logging Query Language](https://luisrangelc.medium.com/guide-to-gcps-logging-query-language-bc08a5ce4acb) (accessed 2025-02-03).

---

## Section 2: Log Sinks (~150 lines)

### Log Sink Architecture

**Log sinks export logs to:**
1. **BigQuery** - SQL analysis, long-term storage
2. **Cloud Storage** - Archival, compliance
3. **Pub/Sub** - Real-time processing, streaming
4. **Other Cloud Logging buckets** - Cross-project aggregation

**Sink components:**
- **Filter** - Which logs to export
- **Destination** - Where to send logs
- **Writer identity** - Service account for export

### Creating Log Sinks

**Command-line sink creation:**
```bash
# Export to BigQuery
gcloud logging sinks create my-bigquery-sink \
  bigquery.googleapis.com/projects/PROJECT_ID/datasets/DATASET_ID \
  --log-filter='resource.type="gce_instance" severity>=WARNING'

# Export to Cloud Storage
gcloud logging sinks create my-gcs-sink \
  storage.googleapis.com/my-log-bucket \
  --log-filter='resource.type="k8s_container"'

# Export to Pub/Sub
gcloud logging sinks create my-pubsub-sink \
  pubsub.googleapis.com/projects/PROJECT_ID/topics/TOPIC_ID \
  --log-filter='severity>=ERROR'
```

**Get sink writer identity:**
```bash
gcloud logging sinks describe my-bigquery-sink \
  --format='value(writerIdentity)'
```

**Grant permissions to writer identity:**
```bash
# BigQuery
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:WRITER_IDENTITY" \
  --role="roles/bigquery.dataEditor"

# Cloud Storage
gsutil iam ch serviceAccount:WRITER_IDENTITY:objectCreator \
  gs://my-log-bucket

# Pub/Sub
gcloud pubsub topics add-iam-policy-binding TOPIC_ID \
  --member="serviceAccount:WRITER_IDENTITY" \
  --role="roles/pubsub.publisher"
```

From [Troubleshoot routing and storing logs](https://docs.cloud.google.com/logging/docs/export/troubleshoot) (accessed 2025-02-03).

### BigQuery Sink Patterns

**Schema auto-detection:**
- BigQuery automatically creates tables based on log structure
- Tables named by log type: `cloudaudit_googleapis_com_activity`
- Partitioned by timestamp for cost optimization

**Query example (BigQuery):**
```sql
SELECT
  timestamp,
  resource.labels.project_id,
  protoPayload.methodName,
  protoPayload.authenticationInfo.principalEmail
FROM `project_id.dataset_id.cloudaudit_googleapis_com_activity`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND protoPayload.serviceName = "compute.googleapis.com"
ORDER BY timestamp DESC
LIMIT 1000
```

**Cost optimization:**
- Use table expiration (90 days default for logs)
- Partition by day/month for large volumes
- Use clustering on frequently filtered columns

### Cloud Storage Sink Patterns

**File organization:**
```
gs://bucket-name/
  2024/
    01/
      15/
        00:00:00_00:59:59_S0.json
        01:00:00_01:59:59_S0.json
```

**Use cases:**
- Long-term archival (7+ years)
- Compliance requirements
- Cost-effective storage ($0.01/GB/month for Coldline)

**Lifecycle policy example:**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 2555}
      }
    ]
  }
}
```

### Pub/Sub Sink Patterns

**Real-time log processing:**
- Stream logs to Cloud Functions
- Trigger alerts based on log patterns
- Forward to external SIEM systems

**Cloud Function example (Python):**
```python
import base64
import json

def process_log_entry(event, context):
    """Process Pub/Sub log message"""
    log_entry = json.loads(base64.b64decode(event['data']).decode('utf-8'))

    # Check for critical errors
    if log_entry.get('severity') == 'ERROR':
        error_message = log_entry.get('jsonPayload', {}).get('message', '')

        # Alert logic here
        send_alert(error_message)
```

**Subscription with filter:**
```bash
gcloud pubsub subscriptions create my-error-sub \
  --topic=log-export-topic \
  --ack-deadline=60 \
  --message-filter='attributes.severity="ERROR"'
```

### Aggregated Sinks

**Organization-level sinks** export logs from all projects:

```bash
gcloud logging sinks create org-wide-audit-logs \
  bigquery.googleapis.com/projects/CENTRAL_PROJECT/datasets/org_logs \
  --organization=ORG_ID \
  --log-filter='protoPayload.serviceName="cloudaudit.googleapis.com"' \
  --include-children
```

**Benefits:**
- Centralized security monitoring
- Compliance auditing across organization
- Cost tracking by project

From [REST Resource: projects.sinks](https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks) (accessed 2025-02-03).

---

## Section 3: Structured Logging (~150 lines)

### Why Structured Logging

**Benefits over plain text:**
- Efficient filtering and aggregation
- Automatic field indexing
- Integration with Log Analytics
- Better cost optimization (smaller payloads)

**Comparison:**

**Plain text (BAD):**
```
User john@example.com logged in from IP 192.168.1.1 at 2024-02-03 10:30:00
```

**Structured JSON (GOOD):**
```json
{
  "event": "user_login",
  "user_email": "john@example.com",
  "source_ip": "192.168.1.1",
  "timestamp": "2024-02-03T10:30:00Z",
  "success": true
}
```

From [Structured logging | Cloud Logging](https://docs.cloud.google.com/logging/docs/structured-logging) (accessed 2025-02-03).

### Implementation Patterns

**Python (using standard library):**
```python
import logging
import json
from datetime import datetime

class StructuredMessage:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            **self.kwargs
        })

logger = logging.getLogger(__name__)
logger.info(StructuredMessage(
    event='user_action',
    user_id='12345',
    action='file_upload',
    file_size_bytes=1024000,
    duration_ms=250
))
```

**Python (using google-cloud-logging):**
```python
from google.cloud import logging as cloud_logging

client = cloud_logging.Client()
logger = client.logger('my-app-log')

logger.log_struct({
    'event': 'api_request',
    'method': 'POST',
    'endpoint': '/api/users',
    'status_code': 201,
    'duration_ms': 145,
    'user_id': '12345'
}, severity='INFO')
```

**Node.js (using winston):**
```javascript
const winston = require('winston');
const {LoggingWinston} = require('@google-cloud/logging-winston');

const loggingWinston = new LoggingWinston();

const logger = winston.createLogger({
  level: 'info',
  transports: [
    new winston.transports.Console(),
    loggingWinston,
  ],
});

logger.info({
  event: 'database_query',
  query_type: 'SELECT',
  table: 'users',
  duration_ms: 23,
  rows_returned: 150
});
```

**Go (using structured logging):**
```go
package main

import (
    "context"
    "log"
    "cloud.google.com/go/logging"
)

func main() {
    ctx := context.Background()
    client, _ := logging.NewClient(ctx, "my-project")
    defer client.Close()

    logger := client.Logger("my-app-log")

    logger.Log(logging.Entry{
        Severity: logging.Info,
        Payload: map[string]interface{}{
            "event": "cache_operation",
            "operation": "SET",
            "key": "user:12345",
            "ttl_seconds": 3600,
            "success": true,
        },
    })
}
```

### Standard Field Names

**Recommended field naming conventions:**

**HTTP requests:**
```json
{
  "http_request": {
    "method": "GET",
    "url": "https://api.example.com/users",
    "status": 200,
    "response_size": 4523,
    "user_agent": "Mozilla/5.0...",
    "remote_ip": "203.0.113.1",
    "latency": "0.125s"
  }
}
```

**Errors:**
```json
{
  "error": {
    "type": "DatabaseConnectionError",
    "message": "Failed to connect to database",
    "stack_trace": "...",
    "code": "DB_CONN_TIMEOUT"
  },
  "context": {
    "database_host": "db.example.com",
    "connection_attempts": 3
  }
}
```

**Business events:**
```json
{
  "event": "order_completed",
  "order_id": "ORD-12345",
  "user_id": "USR-67890",
  "amount": 125.50,
  "currency": "USD",
  "payment_method": "credit_card",
  "items_count": 3
}
```

### Log Levels and Severity

**Standard severity levels:**
- `DEFAULT` - Routine information (use sparingly)
- `DEBUG` - Debug/trace information
- `INFO` - Informational messages
- `NOTICE` - Normal but significant events
- `WARNING` - Warning conditions
- `ERROR` - Error conditions
- `CRITICAL` - Critical conditions
- `ALERT` - Action must be taken immediately
- `EMERGENCY` - System is unusable

**Best practices:**
```python
# DON'T: Log everything at INFO
logger.info("Starting function")  # Too verbose
logger.info("Calling API")
logger.info("API response received")
logger.info("Function complete")

# DO: Use appropriate levels
logger.debug("Function started with params: %s", params)  # Debug only
logger.info("Order created: order_id=%s", order_id)  # Significant event
logger.warning("API rate limit approaching: %d/%d", current, limit)
logger.error("Payment processing failed: %s", error)
```

### Special Log Fields

**Cloud Logging recognizes special fields:**

**Trace context (for distributed tracing):**
```json
{
  "logging.googleapis.com/trace": "projects/PROJECT_ID/traces/TRACE_ID",
  "logging.googleapis.com/spanId": "SPAN_ID",
  "logging.googleapis.com/trace_sampled": true
}
```

**Source location:**
```json
{
  "logging.googleapis.com/sourceLocation": {
    "file": "api/handlers.py",
    "line": "145",
    "function": "process_request"
  }
}
```

**Labels (for grouping/filtering):**
```json
{
  "logging.googleapis.com/labels": {
    "service": "user-api",
    "version": "v2.1.3",
    "environment": "production"
  }
}
```

From [Structured Logging - A Developer's Guide](https://signoz.io/blog/structured-logs/) (accessed 2025-02-03).

---

## Section 4: Cost Optimization (~100 lines)

### Understanding Logging Costs

**Pricing tiers (as of 2024):**
- **First 50 GB/month** - Free
- **51-10,000 GB** - $0.50/GB
- **10,000+ GB** - Volume discounts available

**Storage costs:**
- **Default retention (30 days)** - Included
- **Extended retention** - $0.01/GB/month

**What drives costs:**
1. Log ingestion volume (largest cost)
2. Storage beyond 30 days
3. Log Analytics queries (BigQuery pricing)

From [Cloud Logging cost management best practices](https://cloud.google.com/blog/products/devops-sre/cloud-logging-cost-management-best-practices) (accessed 2025-02-03).

### Step 1: Analyze Current Spending

**View log volume by resource:**
```
resource.type=~".*"
|> aggregate sum(size_bytes) by resource.type
|> order by sum_size_bytes desc
```

**View log volume by severity:**
```
severity=~".*"
|> aggregate sum(size_bytes) by severity
```

**Using Logs Explorer metrics:**
```bash
# Get ingestion metrics
gcloud logging metrics list --filter="name:log_ingestion"

# Create custom metric for cost tracking
gcloud logging metrics create log_volume_by_service \
  --description="Log volume by service" \
  --log-filter='resource.type="k8s_container"' \
  --value-extractor='EXTRACT(resource.labels.service_name)'
```

### Step 2: Eliminate Waste (Exclusion Filters)

**Common logs to exclude:**

**Exclude health check logs:**
```bash
# GKE health checks
resource.type="k8s_container"
jsonPayload.message=~"/healthz|/readyz"

# Load balancer health checks
resource.type="http_load_balancer"
httpRequest.requestUrl=~"/health"
```

**Exclude debug logs in production:**
```bash
resource.labels.namespace_name="production"
severity="DEBUG"
```

**Create exclusion filter:**
```bash
gcloud logging sinks create exclude-health-checks \
  logging.googleapis.com/projects/PROJECT_ID/logs/_Default \
  --log-filter='NOT (resource.type="k8s_container" AND jsonPayload.message=~"/healthz")' \
  --exclusion-filter='resource.type="k8s_container" AND jsonPayload.message=~"/healthz"'
```

**Organization-level exclusions:**
```bash
# Exclude org-wide
gcloud logging exclusions create exclude-debug-prod \
  --organization=ORG_ID \
  --log-filter='severity="DEBUG" AND resource.labels.environment="production"'
```

### Step 3: Optimize Costs Over Time

**Strategy 1: Sampling**
- Log 100% of errors
- Sample 10% of INFO logs
- Sample 1% of DEBUG logs

**Implementation (application-side):**
```python
import random
import logging

def should_log(severity, sample_rate=1.0):
    if severity >= logging.ERROR:
        return True  # Always log errors
    return random.random() < sample_rate

if should_log(logging.INFO, sample_rate=0.1):
    logger.info("User login successful")
```

**Strategy 2: Dynamic retention**
```bash
# Short retention for high-volume logs
gcloud logging buckets update _Default \
  --location=global \
  --retention-days=7 \
  --log-filter='severity<WARNING'

# Longer retention for errors
gcloud logging buckets create errors-bucket \
  --location=global \
  --retention-days=90 \
  --log-filter='severity>=ERROR'
```

**Strategy 3: Export to cheaper storage**
```bash
# Export old logs to Cloud Storage (Coldline)
gcloud logging sinks create archive-old-logs \
  storage.googleapis.com/my-coldline-bucket \
  --log-filter='timestamp < timestamp_sub(timestamp("now"), interval 30 day)'
```

**Strategy 4: Reduce payload size**
```python
# BAD: Logging entire request/response
logger.info(f"API response: {json.dumps(large_response)}")  # 10KB

# GOOD: Log only essentials
logger.info("API success", extra={
    'status': 200,
    'duration_ms': 145,
    'record_count': 150
})  # 100 bytes
```

### Real-World Cost Savings

**Example: GKE logging optimization**

**Before:**
- 500 GB/month ingestion
- Cost: $225/month (450 GB @ $0.50/GB)

**Actions taken:**
1. Excluded health check logs (200 GB saved)
2. Sampled DEBUG logs at 5% (100 GB saved)
3. Reduced payload size (50 GB saved)

**After:**
- 150 GB/month ingestion
- Cost: $50/month (100 GB @ $0.50/GB)
- **Savings: 78% ($175/month)**

From [How To Reduce Logging Costs in GCP](https://www.finout.io/blog/reducing-gcp-logging-costs) (accessed 2025-02-03).

### Cost Monitoring Dashboard

**Create budget alert:**
```bash
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Logging Budget Alert" \
  --budget-amount=500 \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100
```

**Custom metrics for cost tracking:**
```bash
# Log-based metric: bytes ingested by service
gcloud logging metrics create bytes_by_service \
  --value-extractor='EXTRACT(size_bytes)' \
  --metric-kind=DELTA \
  --value-type=INT64 \
  --log-filter='resource.type="k8s_container"'
```

---

## Section 5: Error Reporting Integration (~50 lines)

### Automatic Error Grouping

**Error Reporting** automatically groups similar errors:

**Requirements:**
- Logs with `severity=ERROR` or higher
- Stack trace in `message` or `jsonPayload.stack_trace`

**Structured error format:**
```python
logger.error("Database connection failed", extra={
    'error_type': 'DatabaseError',
    'error_message': 'Connection timeout after 30s',
    'stack_trace': traceback.format_exc(),
    'context': {
        'database_host': 'db.example.com',
        'retry_count': 3
    }
})
```

### Error Grouping Rules

**Errors are grouped by:**
1. Stack trace similarity
2. Error message pattern
3. Source location (file/line)

**Example grouping:**
```
Group 1: "DatabaseError: Connection timeout"
  - 45 occurrences in last hour
  - Affects: api-server-1, api-server-2, api-server-3

Group 2: "ValueError: Invalid user ID format"
  - 12 occurrences in last hour
  - Affects: auth-service-1
```

### Integration with Cloud Monitoring

**Create alert policy from Error Reporting:**
```bash
gcloud alpha monitoring policies create \
  --display-name="Critical Error Rate Alert" \
  --condition-display-name="Error rate > 10/min" \
  --condition-threshold-value=10 \
  --condition-threshold-duration=300s \
  --notification-channels=CHANNEL_ID
```

**Query errors from logs:**
```
resource.type="cloud_run_revision"
severity=ERROR
jsonPayload.error_type="DatabaseError"
timestamp >= timestamp_sub(timestamp("now"), interval 1 hour)
|> aggregate count() by resource.labels.service_name
```

From [Error Reporting documentation](https://cloud.google.com/error-reporting/docs) (accessed 2025-02-03).

---

## Sources

**Official Documentation:**
- [Logging query language](https://docs.cloud.google.com/logging/docs/view/logging-query-language) - Google Cloud Docs
- [Structured logging](https://docs.cloud.google.com/logging/docs/structured-logging) - Google Cloud Docs
- [Troubleshoot routing and storing logs](https://docs.cloud.google.com/logging/docs/export/troubleshoot) - Google Cloud Docs
- [REST Resource: projects.sinks](https://cloud.google.com/logging/docs/reference/v2/rest/v2/projects.sinks) - Google Cloud API Reference

**Blog Posts & Guides:**
- [Introducing pipe syntax in BigQuery and Cloud Logging](https://cloud.google.com/blog/products/data-analytics/introducing-pipe-syntax-in-bigquery-and-cloud-logging) - Google Cloud Blog (October 2024)
- [Cloud Logging cost management best practices](https://cloud.google.com/blog/products/devops-sre/cloud-logging-cost-management-best-practices) - Google Cloud Blog (May 2023)
- [Guide to GCP's Logging Query Language](https://luisrangelc.medium.com/guide-to-gcps-logging-query-language-bc08a5ce4acb) - Medium (2023)
- [How To Reduce Logging Costs in GCP](https://www.finout.io/blog/reducing-gcp-logging-costs) - Finout Blog (April 2023)

**General Best Practices:**
- [Structured Logging - A Developer's Guide](https://signoz.io/blog/structured-logs/) - SigNoz Blog (August 2024)
- [9 Logging Best Practices You Should Know](https://www.dash0.com/guides/logging-best-practices) - Dash0 Guides (October 2024)
- [Logging Best Practices: 12 Dos and Don'ts](https://betterstack.com/community/guides/logging/logging-best-practices/) - Better Stack (October 2024)

**Web Research:** All sources accessed 2025-02-03 during PART 39 knowledge acquisition.
