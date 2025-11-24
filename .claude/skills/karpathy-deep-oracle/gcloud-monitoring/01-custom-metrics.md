# Cloud Monitoring Custom Metrics

Comprehensive guide to creating, managing, and automating custom metrics in Google Cloud Monitoring using the Custom Metrics API, OpenTelemetry integration, and dashboard automation.

---

## Overview

Custom metrics extend Cloud Monitoring beyond built-in metrics to track application-specific KPIs, business metrics, and specialized performance indicators. They enable:

- Application-level observability (request latency, queue depth, cache hit rate)
- Business metrics (conversion rate, revenue per transaction, active users)
- ML training metrics (loss curves, validation accuracy, GPU utilization)
- Infrastructure health (custom health checks, resource pool availability)

**Key capabilities:**
- Write up to 10,000 custom time series per project
- 200 custom metric descriptors per project
- 1-second resolution (fastest sampling interval)
- Integration with alerting, dashboards, and SLOs

---

## Section 1: Custom Metrics API

### Creating Metric Descriptors

**Metric descriptor** defines the schema for a custom metric (like a database table schema).

```python
from google.cloud import monitoring_v3

# Initialize client
client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Define metric descriptor
descriptor = monitoring_v3.MetricDescriptor()
descriptor.type = "custom.googleapis.com/ml_training/gpu_utilization"
descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
descriptor.description = "GPU utilization percentage (0-100)"
descriptor.display_name = "GPU Utilization"
descriptor.unit = "%"

# Add labels for dimensions
descriptor.labels.append(
    monitoring_v3.LabelDescriptor(
        key="gpu_id",
        value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
        description="GPU device ID"
    )
)
descriptor.labels.append(
    monitoring_v3.LabelDescriptor(
        key="job_name",
        value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
        description="Training job name"
    )
)

# Create descriptor
descriptor = client.create_metric_descriptor(
    name=project_name,
    metric_descriptor=descriptor
)
print(f"Created metric descriptor: {descriptor.name}")
```

**Metric kinds:**
- `GAUGE`: Instantaneous measurement (CPU usage, queue length)
- `DELTA`: Change since last measurement (requests served in interval)
- `CUMULATIVE`: Running total (total requests served since start)

**Value types:**
- `DOUBLE`: Floating-point numbers (99.9% of use cases)
- `INT64`: Integers (counts, IDs)
- `BOOL`: Boolean values (is_healthy)
- `STRING`: Text values (status codes)
- `DISTRIBUTION`: Histogram of values (latency buckets)

### Writing Time Series Data

**Time series** is the actual data written to a custom metric.

```python
from google.cloud import monitoring_v3
import time

def write_gpu_utilization(project_id, gpu_id, job_name, utilization):
    """Write GPU utilization metric."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    # Create time series
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/ml_training/gpu_utilization"
    series.metric.labels["gpu_id"] = gpu_id
    series.metric.labels["job_name"] = job_name

    # Set resource (what's being monitored)
    series.resource.type = "gce_instance"
    series.resource.labels["instance_id"] = "1234567890"
    series.resource.labels["zone"] = "us-west2-a"

    # Create data point
    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10 ** 9)
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": seconds, "nanos": nanos}}
    )
    point = monitoring_v3.Point(
        {"interval": interval, "value": {"double_value": utilization}}
    )
    series.points = [point]

    # Write time series
    client.create_time_series(name=project_name, time_series=[series])
    print(f"Wrote GPU utilization: {utilization}% for GPU {gpu_id}")

# Usage
write_gpu_utilization(
    project_id="my-project",
    gpu_id="GPU-0",
    job_name="vit-training-v3",
    utilization=87.5
)
```

### Batch Writing for Performance

Write multiple time series in a single API call (up to 200 per request):

```python
def write_batch_metrics(project_id, metrics_data):
    """Write multiple metrics efficiently."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    time_series = []
    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10 ** 9)

    for metric in metrics_data:
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric["type"]
        series.metric.labels.update(metric["labels"])
        series.resource.type = metric["resource_type"]
        series.resource.labels.update(metric["resource_labels"])

        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        point = monitoring_v3.Point(
            {"interval": interval, "value": {"double_value": metric["value"]}}
        )
        series.points = [point]
        time_series.append(series)

    # Write all time series at once
    client.create_time_series(name=project_name, time_series=time_series)
    print(f"Wrote {len(time_series)} metrics in batch")

# Usage
metrics = [
    {
        "type": "custom.googleapis.com/ml_training/gpu_utilization",
        "labels": {"gpu_id": "GPU-0", "job_name": "vit-v3"},
        "resource_type": "gce_instance",
        "resource_labels": {"instance_id": "123", "zone": "us-west2-a"},
        "value": 87.5
    },
    {
        "type": "custom.googleapis.com/ml_training/training_loss",
        "labels": {"job_name": "vit-v3", "step": "1000"},
        "resource_type": "gce_instance",
        "resource_labels": {"instance_id": "123", "zone": "us-west2-a"},
        "value": 0.342
    }
]
write_batch_metrics("my-project", metrics)
```

### Distribution Metrics (Histograms)

Track latency, request sizes, or any value distribution:

```python
def write_latency_distribution(project_id, latencies):
    """Write request latency as distribution metric."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    # Create distribution descriptor (once)
    descriptor = monitoring_v3.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/api/request_latency"
    descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DISTRIBUTION
    descriptor.description = "API request latency distribution"
    descriptor.unit = "ms"

    # Define bucket boundaries (exponential buckets)
    # Buckets: <10ms, 10-25ms, 25-50ms, 50-100ms, 100-250ms, >250ms
    buckets = [10, 25, 50, 100, 250]

    # Count values in each bucket
    from collections import defaultdict
    bucket_counts = defaultdict(int)
    for latency in latencies:
        for i, boundary in enumerate(buckets):
            if latency < boundary:
                bucket_counts[i] += 1
                break
        else:
            bucket_counts[len(buckets)] += 1  # Overflow bucket

    # Create distribution value
    distribution = monitoring_v3.Distribution()
    distribution.count = len(latencies)
    distribution.mean = sum(latencies) / len(latencies)

    # Set bucket options
    distribution.bucket_options.explicit_buckets.bounds.extend(buckets)

    # Set bucket counts
    distribution.bucket_counts.extend([
        bucket_counts[i] for i in range(len(buckets) + 1)
    ])

    # Create time series
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/api/request_latency"
    series.resource.type = "gce_instance"
    series.resource.labels["instance_id"] = "123"
    series.resource.labels["zone"] = "us-west2-a"

    now = time.time()
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": int(now), "nanos": int((now % 1) * 10**9)}}
    )
    point = monitoring_v3.Point(
        {"interval": interval, "value": {"distribution_value": distribution}}
    )
    series.points = [point]

    client.create_time_series(name=project_name, time_series=[series])
    print(f"Wrote latency distribution: {len(latencies)} samples")

# Usage
latencies = [15.2, 8.3, 45.1, 12.7, 103.5, 22.8, 67.4]
write_latency_distribution("my-project", latencies)
```

---

## Section 2: OpenTelemetry Integration

OpenTelemetry provides vendor-neutral instrumentation for traces, metrics, and logs. GCP natively supports OTLP (OpenTelemetry Protocol).

### Installing OpenTelemetry SDK

```bash
pip install opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-gcp-monitoring \
    opentelemetry-instrumentation
```

### Basic Metrics Setup

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

# Create exporter for GCP
exporter = CloudMonitoringMetricsExporter(project_id="my-project")

# Create meter provider with periodic export (every 60 seconds)
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60000)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Get meter for your service
meter = metrics.get_meter("ml-training-service", version="1.0")

# Create instruments
gpu_utilization = meter.create_gauge(
    name="gpu_utilization",
    description="GPU utilization percentage",
    unit="%"
)

training_loss = meter.create_histogram(
    name="training_loss",
    description="Training loss value",
    unit="1"
)

requests_counter = meter.create_counter(
    name="api_requests_total",
    description="Total API requests",
    unit="1"
)

# Record measurements
def record_metrics(gpu_id, loss_value):
    gpu_utilization.set(87.5, {"gpu_id": gpu_id})
    training_loss.record(loss_value, {"step": "1000"})
    requests_counter.add(1, {"endpoint": "/predict", "status": "200"})

# Metrics exported automatically every 60 seconds
```

### Auto-Instrumentation for Web Frameworks

Automatically capture HTTP request metrics:

```python
from flask import Flask
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

app = Flask(__name__)

# Setup OpenTelemetry
exporter = CloudMonitoringMetricsExporter(project_id="my-project")
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60000)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Auto-instrument Flask
FlaskInstrumentor().instrument_app(app)

@app.route("/predict")
def predict():
    # Automatically tracked:
    # - http.server.duration (request latency)
    # - http.server.request.count (request count)
    # - http.server.response.size (response bytes)
    return {"prediction": 0.95}

if __name__ == "__main__":
    app.run()
```

### Custom Metric Attributes (Labels)

Add contextual dimensions to metrics:

```python
meter = metrics.get_meter("ml-training")

# Counter with attributes
job_counter = meter.create_counter(
    name="training_jobs_started",
    description="Number of training jobs started"
)

# Record with attributes (become metric labels in GCP)
job_counter.add(1, {
    "model_type": "vision_transformer",
    "framework": "pytorch",
    "gpu_type": "t4",
    "region": "us-west2"
})

# Query in MQL:
# fetch custom.googleapis.com/ml-training/training_jobs_started
# | filter metric.model_type == "vision_transformer"
# | group_by [metric.region], sum(value.value)
```

### OpenTelemetry Collector for Advanced Routing

Deploy a collector to:
- Buffer metrics before export (handle GCP API rate limits)
- Route metrics to multiple backends (GCP + Prometheus)
- Transform/enrich metrics (add resource attributes)
- Batch exports for efficiency

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 60s
    send_batch_size: 100

  resource:
    attributes:
      - key: service.namespace
        value: ml-training
        action: insert
      - key: deployment.environment
        value: production
        action: insert

exporters:
  googlecloud:
    project: my-project
    metric:
      prefix: custom.googleapis.com/otel

  prometheus:
    endpoint: "prometheus:9090"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [googlecloud, prometheus]
```

Deploy collector:

```bash
# Docker
docker run -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otel-collector-config.yaml

# Kubernetes (Cloud Run, GKE)
kubectl apply -f https://github.com/open-telemetry/opentelemetry-operator/releases/latest/download/opentelemetry-operator.yaml
```

---

## Section 3: Dashboard Automation

Create and update dashboards programmatically using the Dashboards API.

### Creating a Dashboard

```python
from google.cloud import monitoring_dashboard_v1

def create_ml_training_dashboard(project_id):
    """Create dashboard for ML training metrics."""
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    project_name = f"projects/{project_id}"

    # Define dashboard
    dashboard = monitoring_dashboard_v1.Dashboard()
    dashboard.display_name = "ML Training - GPU Utilization"

    # Create line chart widget
    widget = monitoring_dashboard_v1.Widget()
    widget.title = "GPU Utilization by Job"

    # Define MQL query
    xy_chart = monitoring_dashboard_v1.XyChart()
    data_set = monitoring_dashboard_v1.XyChart.DataSet()
    data_set.time_series_query.time_series_filter.filter = (
        'metric.type="custom.googleapis.com/ml_training/gpu_utilization"'
    )
    data_set.time_series_query.time_series_filter.aggregation.alignment_period.seconds = 60
    data_set.time_series_query.time_series_filter.aggregation.per_series_aligner = (
        monitoring_dashboard_v1.Aggregation.Aligner.ALIGN_MEAN
    )
    data_set.plot_type = monitoring_dashboard_v1.XyChart.DataSet.PlotType.LINE

    xy_chart.data_sets.append(data_set)
    xy_chart.y_axis.label = "GPU Utilization (%)"
    xy_chart.y_axis.scale = monitoring_dashboard_v1.XyChart.Axis.Scale.LINEAR

    widget.xy_chart.CopyFrom(xy_chart)

    # Add widget to grid layout
    grid_layout = monitoring_dashboard_v1.GridLayout()
    grid_layout.widgets.append(widget)
    dashboard.grid_layout.CopyFrom(grid_layout)

    # Create dashboard
    dashboard = client.create_dashboard(parent=project_name, dashboard=dashboard)
    print(f"Created dashboard: {dashboard.name}")
    return dashboard

# Usage
dashboard = create_ml_training_dashboard("my-project")
```

### Dashboard with Multiple Widgets

```python
def create_comprehensive_dashboard(project_id):
    """Create multi-widget dashboard."""
    client = monitoring_dashboard_v1.DashboardsServiceClient()
    project_name = f"projects/{project_id}"

    dashboard = monitoring_dashboard_v1.Dashboard()
    dashboard.display_name = "ML Training Overview"

    grid_layout = monitoring_dashboard_v1.GridLayout()

    # Widget 1: GPU Utilization (line chart)
    widget1 = create_line_chart_widget(
        title="GPU Utilization",
        metric_type="custom.googleapis.com/ml_training/gpu_utilization",
        y_label="Utilization (%)"
    )
    widget1.width = 6
    widget1.height = 4
    widget1.x_pos = 0
    widget1.y_pos = 0
    grid_layout.widgets.append(widget1)

    # Widget 2: Training Loss (line chart)
    widget2 = create_line_chart_widget(
        title="Training Loss",
        metric_type="custom.googleapis.com/ml_training/training_loss",
        y_label="Loss"
    )
    widget2.width = 6
    widget2.height = 4
    widget2.x_pos = 6
    widget2.y_pos = 0
    grid_layout.widgets.append(widget2)

    # Widget 3: API Request Rate (scorecard)
    widget3 = create_scorecard_widget(
        title="API Requests/min",
        metric_type="custom.googleapis.com/api/request_latency"
    )
    widget3.width = 3
    widget3.height = 2
    widget3.x_pos = 0
    widget3.y_pos = 4
    grid_layout.widgets.append(widget3)

    # Widget 4: Error Rate (scorecard with threshold)
    widget4 = create_scorecard_widget(
        title="Error Rate",
        metric_type="custom.googleapis.com/api/errors_total",
        threshold=0.01  # Alert if >1% errors
    )
    widget4.width = 3
    widget4.height = 2
    widget4.x_pos = 3
    widget4.y_pos = 4
    grid_layout.widgets.append(widget4)

    dashboard.grid_layout.CopyFrom(grid_layout)
    dashboard = client.create_dashboard(parent=project_name, dashboard=dashboard)
    print(f"Created dashboard: {dashboard.name}")
    return dashboard

def create_line_chart_widget(title, metric_type, y_label):
    """Helper to create line chart widget."""
    widget = monitoring_dashboard_v1.Widget()
    widget.title = title

    xy_chart = monitoring_dashboard_v1.XyChart()
    data_set = monitoring_dashboard_v1.XyChart.DataSet()
    data_set.time_series_query.time_series_filter.filter = (
        f'metric.type="{metric_type}"'
    )
    data_set.time_series_query.time_series_filter.aggregation.alignment_period.seconds = 60
    data_set.time_series_query.time_series_filter.aggregation.per_series_aligner = (
        monitoring_dashboard_v1.Aggregation.Aligner.ALIGN_MEAN
    )
    data_set.plot_type = monitoring_dashboard_v1.XyChart.DataSet.PlotType.LINE

    xy_chart.data_sets.append(data_set)
    xy_chart.y_axis.label = y_label
    widget.xy_chart.CopyFrom(xy_chart)
    return widget

def create_scorecard_widget(title, metric_type, threshold=None):
    """Helper to create scorecard widget."""
    widget = monitoring_dashboard_v1.Widget()
    widget.title = title

    scorecard = monitoring_dashboard_v1.Scorecard()
    scorecard.time_series_query.time_series_filter.filter = (
        f'metric.type="{metric_type}"'
    )
    scorecard.time_series_query.time_series_filter.aggregation.alignment_period.seconds = 60
    scorecard.time_series_query.time_series_filter.aggregation.per_series_aligner = (
        monitoring_dashboard_v1.Aggregation.Aligner.ALIGN_RATE
    )

    if threshold:
        scorecard.thresholds.append(
            monitoring_dashboard_v1.Threshold(value=threshold, color="RED")
        )

    widget.scorecard.CopyFrom(scorecard)
    return widget
```

### Updating Dashboards via Terraform

Manage dashboards as code:

```hcl
# terraform/dashboards.tf
resource "google_monitoring_dashboard" "ml_training" {
  dashboard_json = jsonencode({
    displayName = "ML Training Metrics"
    gridLayout = {
      widgets = [
        {
          title = "GPU Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/ml_training/gpu_utilization\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              plotType = "LINE"
            }]
            yAxis = {
              label = "Utilization (%)"
              scale = "LINEAR"
            }
          }
        }
      ]
    }
  })
}
```

---

## Section 4: Alert Policies for Custom Metrics

Create alerts that trigger when custom metrics exceed thresholds.

### Basic Alert Policy

```python
from google.cloud import monitoring_v3

def create_gpu_utilization_alert(project_id):
    """Alert when GPU utilization drops below 50%."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    # Define alert condition
    condition = monitoring_v3.AlertPolicy.Condition()
    condition.display_name = "GPU utilization too low"

    # Threshold condition
    condition.condition_threshold.filter = (
        'metric.type="custom.googleapis.com/ml_training/gpu_utilization"'
    )
    condition.condition_threshold.comparison = (
        monitoring_v3.ComparisonType.COMPARISON_LT
    )
    condition.condition_threshold.threshold_value = 50.0
    condition.condition_threshold.duration.seconds = 300  # 5 minutes
    condition.condition_threshold.aggregations.append(
        monitoring_v3.Aggregation(
            alignment_period={"seconds": 60},
            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
        )
    )

    # Create alert policy
    policy = monitoring_v3.AlertPolicy()
    policy.display_name = "GPU Utilization Alert"
    policy.conditions.append(condition)
    policy.combiner = monitoring_v3.AlertPolicy.ConditionCombinerType.AND

    # Notification channels (email, Slack, PagerDuty)
    policy.notification_channels.append(
        f"{project_name}/notificationChannels/{notification_channel_id}"
    )

    # Documentation shown in alert
    policy.documentation.content = (
        "GPU utilization has dropped below 50% for 5 minutes. "
        "Check if training job is stuck or waiting for data."
    )
    policy.documentation.mime_type = "text/markdown"

    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"Created alert policy: {policy.name}")
    return policy
```

### Multi-Condition Alert

Alert when multiple metrics violate thresholds:

```python
def create_training_health_alert(project_id):
    """Alert when training shows signs of issues."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    policy = monitoring_v3.AlertPolicy()
    policy.display_name = "Training Health Alert"

    # Condition 1: GPU utilization too low
    cond1 = monitoring_v3.AlertPolicy.Condition()
    cond1.display_name = "Low GPU utilization"
    cond1.condition_threshold.filter = (
        'metric.type="custom.googleapis.com/ml_training/gpu_utilization"'
    )
    cond1.condition_threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_LT
    cond1.condition_threshold.threshold_value = 30.0
    cond1.condition_threshold.duration.seconds = 300
    policy.conditions.append(cond1)

    # Condition 2: Loss not decreasing
    cond2 = monitoring_v3.AlertPolicy.Condition()
    cond2.display_name = "Loss not improving"
    cond2.condition_threshold.filter = (
        'metric.type="custom.googleapis.com/ml_training/training_loss"'
    )
    cond2.condition_threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_GT
    cond2.condition_threshold.threshold_value = 1.0
    cond2.condition_threshold.duration.seconds = 600
    policy.conditions.append(cond2)

    # Trigger if ANY condition is met (OR combiner)
    policy.combiner = monitoring_v3.AlertPolicy.ConditionCombinerType.OR

    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"Created multi-condition alert: {policy.name}")
    return policy
```

### Rate-Based Alerts

Alert on rate of change (sudden spikes/drops):

```python
def create_error_rate_alert(project_id):
    """Alert on sudden error rate increase."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    condition = monitoring_v3.AlertPolicy.Condition()
    condition.display_name = "High error rate"

    # Rate aggregation
    condition.condition_threshold.filter = (
        'metric.type="custom.googleapis.com/api/errors_total"'
    )
    condition.condition_threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_GT
    condition.condition_threshold.threshold_value = 10.0  # 10 errors/sec
    condition.condition_threshold.duration.seconds = 120

    # ALIGN_RATE converts cumulative counter to rate
    condition.condition_threshold.aggregations.append(
        monitoring_v3.Aggregation(
            alignment_period={"seconds": 60},
            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
        )
    )

    policy = monitoring_v3.AlertPolicy()
    policy.display_name = "API Error Rate Alert"
    policy.conditions.append(condition)
    policy.combiner = monitoring_v3.AlertPolicy.ConditionCombinerType.AND

    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"Created rate alert: {policy.name}")
    return policy
```

---

## Section 5: SLO/SLI Tracking

Service Level Objectives (SLOs) define reliability targets. SLIs (Indicators) are the metrics used to measure SLOs.

### Creating an SLO

```python
from google.cloud import monitoring_v3

def create_api_latency_slo(project_id, service_id):
    """Create SLO for API latency (95% of requests < 500ms)."""
    client = monitoring_v3.ServiceMonitoringServiceClient()
    service_name = f"projects/{project_id}/services/{service_id}"

    # Define SLI (latency-based)
    slo = monitoring_v3.ServiceLevelObjective()
    slo.display_name = "API Latency SLO"
    slo.goal = 0.95  # 95% success rate

    # Rolling window (30 days)
    slo.rolling_period.seconds = 30 * 24 * 60 * 60

    # Request-based SLI
    sli = monitoring_v3.ServiceLevelIndicator()
    sli.request_based.good_total_ratio.good_service_filter = (
        'metric.type="custom.googleapis.com/api/request_latency" '
        'metric.latency < 500'
    )
    sli.request_based.good_total_ratio.total_service_filter = (
        'metric.type="custom.googleapis.com/api/request_latency"'
    )
    slo.service_level_indicator.CopyFrom(sli)

    slo = client.create_service_level_objective(
        parent=service_name,
        service_level_objective=slo
    )
    print(f"Created SLO: {slo.name}")
    return slo
```

### SLO for Availability

```python
def create_availability_slo(project_id, service_id):
    """Create SLO for 99.9% availability."""
    client = monitoring_v3.ServiceMonitoringServiceClient()
    service_name = f"projects/{project_id}/services/{service_id}"

    slo = monitoring_v3.ServiceLevelObjective()
    slo.display_name = "API Availability SLO"
    slo.goal = 0.999  # 99.9% availability

    # Calendar period (monthly)
    slo.calendar_period = monitoring_v3.ServiceLevelObjective.View.CALENDAR_MONTH

    # Availability SLI (good = non-5xx responses)
    sli = monitoring_v3.ServiceLevelIndicator()
    sli.request_based.good_total_ratio.good_service_filter = (
        'metric.type="custom.googleapis.com/api/requests_total" '
        'metric.status_code < 500'
    )
    sli.request_based.good_total_ratio.total_service_filter = (
        'metric.type="custom.googleapis.com/api/requests_total"'
    )
    slo.service_level_indicator.CopyFrom(sli)

    slo = client.create_service_level_objective(
        parent=service_name,
        service_level_objective=slo
    )
    print(f"Created availability SLO: {slo.name}")
    return slo
```

### Error Budget Alerts

Alert when error budget is depleting too fast:

```python
def create_error_budget_alert(project_id, slo_name):
    """Alert when error budget burn rate is high."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"

    condition = monitoring_v3.AlertPolicy.Condition()
    condition.display_name = "Error budget burn rate high"

    # SLO burn rate condition
    condition.condition_threshold.filter = (
        f'select_slo_burn_rate("{slo_name}", "3600s")'
    )
    condition.condition_threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_GT
    condition.condition_threshold.threshold_value = 10.0  # 10x burn rate
    condition.condition_threshold.duration.seconds = 300

    policy = monitoring_v3.AlertPolicy()
    policy.display_name = "Error Budget Alert"
    policy.conditions.append(condition)
    policy.combiner = monitoring_v3.AlertPolicy.ConditionCombinerType.AND

    policy.documentation.content = (
        "Error budget is burning at 10x normal rate. "
        "At this rate, monthly budget will be exhausted in <3 days. "
        "Investigate recent deployments or infrastructure changes."
    )

    policy = client.create_alert_policy(name=project_name, alert_policy=policy)
    print(f"Created error budget alert: {policy.name}")
    return policy
```

---

## Sources

**Official Documentation:**
- [Cloud Monitoring Custom Metrics](https://cloud.google.com/monitoring/custom-metrics) - Google Cloud Docs (accessed 2025-01-13)
- [OpenTelemetry Integration with Cloud Monitoring](https://cloud.google.com/monitoring/custom-metrics/open-telemetry) - Google Cloud Docs (accessed 2025-01-13)
- [Cloud Monitoring Dashboards API](https://cloud.google.com/monitoring/dashboards) - Google Cloud Docs (accessed 2025-01-13)
- [Alert Policies](https://cloud.google.com/monitoring/alerts) - Google Cloud Docs (accessed 2025-01-13)
- [Service Monitoring (SLOs)](https://cloud.google.com/monitoring/service-monitoring) - Google Cloud Docs (accessed 2025-01-13)

**Python Libraries:**
- [google-cloud-monitoring-dashboards 2.19.0](https://pypi.org/project/google-cloud-monitoring-dashboards/) - PyPI (accessed 2025-01-13)
- [GitHub - googleapis/python-monitoring-dashboards](https://github.com/googleapis/python-monitoring-dashboards) - GitHub (accessed 2025-01-13)

**OpenTelemetry Resources:**
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/) - OpenTelemetry.io (accessed 2025-01-13)
- [Open Telemetry (OTLP) metrics on GCP](https://medium.com/google-cloud/open-telemetry-otlp-metrics-on-gcp-48b1a5fda752) - Medium (Nick Brandaleone, 2024-10)

**Web Research:**
- Search: "Cloud Monitoring custom metrics API 2024" (Google Search, 2025-01-13)
- Search: "GCP OpenTelemetry integration custom metrics" (Google Search, 2025-01-13)
- Search: "Cloud Monitoring dashboard automation Python API 2024" (Google Search, 2025-01-13)

**Additional References:**
- [Cloud Run OTLP Metrics Tutorial](https://docs.cloud.google.com/run/docs/tutorials/custom-metrics-opentelemetry-sidecar) - Google Cloud Docs (accessed 2025-01-13)
- [Application Observability in GCP with OpenTelemetry](https://www.retit.de/application-observability-in-gcp-with-opentelemetry-and-the-google-cloud-operations-suite-formerly-stackdriver-en/) - RETIT (2022-03-29)
