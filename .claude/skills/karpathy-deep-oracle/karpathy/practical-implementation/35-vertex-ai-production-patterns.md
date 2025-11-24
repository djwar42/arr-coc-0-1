# Vertex AI Production Patterns

## Overview

Production ML training on Vertex AI requires careful architecture for high availability, cost optimization, and operational resilience. This guide covers enterprise patterns for running reliable, cost-effective training workloads at scale.

From [Vertex AI SLA](https://cloud.google.com/vertex-ai/sla) (accessed 2025-01-31):
- Vertex AI provides 99.5% Monthly Uptime Percentage SLO for training services
- Multi-region deployments increase availability and disaster recovery capabilities
- Production workloads benefit from preemptible VMs (60-91% cost savings)

## Section 1: High Availability Patterns (~200 lines)

### Multi-Region Training Setup

**Why Multi-Region:**
- Protects against regional outages
- Reduces latency for distributed teams
- Enables compliance with data residency requirements
- Improves disaster recovery posture

From [GCP Multi-Region Architecture](https://docs.cloud.google.com/architecture/multiregional-vms) (accessed 2025-01-31):
- Deploy training infrastructure across multiple GCP regions (us-central1, us-east1, europe-west1)
- Use regional Cloud Storage buckets with cross-region replication
- Configure regional Artifact Registry repositories for container images

**Multi-Region Training Architecture:**

```python
# Multi-region training configuration
regions = {
    "primary": "us-central1",
    "secondary": "us-east1",
    "tertiary": "europe-west1"
}

# Launch training job with regional failover
def launch_with_failover(training_config):
    for region in regions.values():
        try:
            job = aiplatform.CustomJob(
                display_name=f"training-{region}",
                worker_pool_specs=training_config,
                location=region
            )
            job.run(sync=False)
            return job
        except Exception as e:
            print(f"Region {region} failed: {e}")
            continue
    raise Exception("All regions failed")
```

**Regional Data Replication:**

```bash
# Create multi-region GCS bucket
gsutil mb -c MULTI_REGIONAL -l US gs://training-data-multi-region/

# Enable versioning for disaster recovery
gsutil versioning set on gs://training-data-multi-region/

# Setup cross-region replication
gsutil rewrite -r gs://training-data-us-central1/** \
    gs://training-data-multi-region/
```

**Container Registry Multi-Region:**

```bash
# Push images to multiple regional registries
regions=("us-central1" "us-east1" "europe-west1")

for region in "${regions[@]}"; do
    docker tag my-training-image:latest \
        ${region}-docker.pkg.dev/project/repo/my-training-image:latest
    docker push ${region}-docker.pkg.dev/project/repo/my-training-image:latest
done
```

### Network Topology Optimization

**VPC Configuration for Multi-Region:**

```python
# Vertex AI custom job with VPC peering
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="multi-region-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-16",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 4
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": f"{region}-docker.pkg.dev/project/repo/image:latest"
        },
        # VPC network configuration
        "network": f"projects/{project_id}/global/networks/ml-training-vpc",
        "reserved_ip_ranges": ["10.0.0.0/16"]
    }],
    location=region
)
```

From [GCP Disaster Recovery Guide](https://docs.cloud.google.com/architecture/disaster-recovery) (accessed 2025-01-31):
- Configure VPC peering between regions for private connectivity
- Use Cloud VPN or Dedicated Interconnect for on-premises connectivity
- Enable Private Service Connect for Vertex AI API access

### Disaster Recovery Planning

**RPO/RTO Targets:**

| Workload Type | RPO | RTO | Strategy |
|--------------|-----|-----|----------|
| Research Training | 24 hours | 4 hours | Daily checkpoints |
| Production Training | 1 hour | 30 minutes | Continuous checkpoints |
| Critical Training | 15 minutes | 10 minutes | Multi-region active |

**Checkpoint Strategy for DR:**

```python
# Disaster recovery checkpoint configuration
checkpoint_config = {
    "checkpoint_dir": "gs://checkpoints-multi-region/job-{job_id}/",
    "checkpoint_frequency": "300s",  # Every 5 minutes
    "max_checkpoints": 10,
    "replicate_to": [
        "gs://checkpoints-us-east1/",
        "gs://checkpoints-europe-west1/"
    ]
}

# Automatic checkpoint replication
def replicate_checkpoint(checkpoint_path):
    for backup_bucket in checkpoint_config["replicate_to"]:
        gsutil_copy = f"gsutil -m cp -r {checkpoint_path} {backup_bucket}"
        subprocess.run(gsutil_copy, shell=True, check=True)
```

### SLA Considerations

From [Vertex AI Platform SLA](https://cloud.google.com/vertex-ai/sla) (accessed 2025-01-31):
- Training service: 99.5% uptime
- Prediction service: 99.9% uptime for online endpoints
- Regional endpoints: Single region SLA
- Multi-regional endpoints: Automatic failover, higher availability

**Monitoring SLA Compliance:**

```python
# Cloud Monitoring alert for SLA tracking
from google.cloud import monitoring_v3

def create_sla_alert(project_id):
    client = monitoring_v3.AlertPolicyServiceClient()

    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Vertex AI Training SLA Violation",
        conditions=[monitoring_v3.AlertPolicy.Condition(
            display_name="Job failure rate > 0.5%",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter='resource.type="aiplatform.googleapis.com/CustomJob" '
                       'metric.type="aiplatform.googleapis.com/job/error_count"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=0.005,  # 0.5% failure rate
                duration={"seconds": 300}
            )
        )],
        notification_channels=[],  # Add notification channels
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            auto_close={"seconds": 604800}  # Auto-close after 7 days
        )
    )

    project_name = f"projects/{project_id}"
    client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

### Health Checks and Monitoring

**Training Job Health Checks:**

```python
# Custom health check for training jobs
def monitor_training_health(job_id):
    """Monitor training job health metrics"""

    metrics = {
        "gpu_utilization": get_gpu_utilization(job_id),
        "memory_usage": get_memory_usage(job_id),
        "loss_divergence": check_loss_divergence(job_id),
        "checkpoint_frequency": check_checkpoint_timing(job_id)
    }

    # Alert if anomalies detected
    if metrics["gpu_utilization"] < 50:
        alert("Low GPU utilization", severity="warning")

    if metrics["loss_divergence"]:
        alert("Training loss diverging", severity="critical")
        trigger_job_termination(job_id)

    return metrics
```

## Section 2: Cost Optimization (~200 lines)

### Preemptible VM Patterns

From [GCP Preemptible VM Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):
- Preemptible VMs: 60-91% discount vs on-demand
- Spot VMs: Similar pricing, longer runtime (no 24-hour limit)
- Available for most machine types including GPU/TPU instances

**Preemptible Training Configuration:**

```python
# Vertex AI job with preemptible VMs
job_spec = {
    "worker_pool_specs": [{
        "machine_spec": {
            "machine_type": "n1-standard-32",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 8
        },
        "replica_count": 4,
        # Enable preemptible VMs
        "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 100
        },
        "container_spec": {
            "image_uri": "gcr.io/project/training-image:latest",
            "command": ["python", "train.py"],
            "args": [
                "--checkpoint-dir", "gs://checkpoints/",
                "--resume-from-checkpoint", "auto"
            ]
        },
        # Spot/Preemptible configuration
        "spot": True  # Use spot instances (preemptible)
    }]
}
```

**Cost Savings Analysis:**

From [GCP Cost Optimization for ML](https://cloud.google.com/blog/products/ai-machine-learning/machine-learning-performance-and-cost-optimization-best-practices) (accessed 2025-01-31):

| Configuration | Cost/Hour | 100-Hour Job | Savings |
|--------------|-----------|--------------|---------|
| 8x V100 On-Demand | $24.80 | $2,480 | - |
| 8x V100 Preemptible | $5.94 | $594 | 76% |
| 8x V100 + Spot | $2.48 | $248 | 90% |

### Checkpoint-Resume for Preemptible Training

From [Reduce ML Costs with Preemptible VMs](https://cloud.google.com/blog/products/ai-machine-learning/reduce-the-costs-of-ml-workflows-with-preemptible-vms-and-gpus) (accessed 2025-01-31):
- Preemptible VMs can be terminated with 30-second notice
- Implement checkpoint-resume to handle interruptions
- Use Cloud Storage for durable checkpoint storage

**Checkpoint-Resume Implementation:**

```python
# Robust checkpoint-resume for preemptible training
import torch
import os
from google.cloud import storage

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.gcs_client = storage.Client()

    def save_checkpoint(self, model, optimizer, epoch, step):
        """Save checkpoint with automatic GCS upload"""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": current_loss
        }

        # Save locally first (faster)
        local_path = f"/tmp/checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, local_path)

        # Upload to GCS (durable)
        gcs_path = f"{self.checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt"
        self.upload_to_gcs(local_path, gcs_path)

        # Keep only last 3 checkpoints
        self.cleanup_old_checkpoints(keep=3)

    def load_latest_checkpoint(self, model, optimizer):
        """Resume from latest checkpoint if exists"""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            print("No checkpoint found, starting from scratch")
            return 0, 0

        latest_checkpoint = checkpoints[-1]
        checkpoint = torch.load(latest_checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint["epoch"], checkpoint["step"]

    def upload_to_gcs(self, local_path, gcs_path):
        """Upload file to GCS"""
        bucket_name = gcs_path.split("/")[2]
        blob_name = "/".join(gcs_path.split("/")[3:])

        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

# Training loop with checkpoint-resume
def train_with_preemptible(model, train_loader, checkpoint_manager):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Resume from checkpoint if exists
    start_epoch, start_step = checkpoint_manager.load_latest_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue  # Skip already processed steps

            # Training step
            loss = train_step(model, batch, optimizer)

            # Save checkpoint every 100 steps
            if step % 100 == 0:
                checkpoint_manager.save_checkpoint(model, optimizer, epoch, step)
```

**Preemption Handler:**

```python
# Handle preemption signals gracefully
import signal
import sys

class PreemptionHandler:
    def __init__(self, checkpoint_manager, model, optimizer):
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.preempted = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_preemption)
        signal.signal(signal.SIGINT, self.handle_preemption)

    def handle_preemption(self, signum, frame):
        """Save checkpoint before VM termination"""
        print("Preemption signal received, saving checkpoint...")
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.optimizer,
            current_epoch,
            current_step
        )
        self.preempted = True
        sys.exit(0)
```

### Committed Use Discounts

From [GCP VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) (accessed 2025-01-31):
- 1-year commitment: 37% discount
- 3-year commitment: 55% discount
- Flexible CUDs: Apply to any machine type in region
- Resource-based CUDs: Specific to machine type

**CUD Planning for ML Workloads:**

```python
# Calculate optimal CUD commitment
def calculate_cud_savings(monthly_usage_hours, hourly_rate):
    """
    Compare on-demand vs CUD pricing

    monthly_usage_hours: Expected GPU hours per month
    hourly_rate: On-demand hourly rate
    """
    on_demand_monthly = monthly_usage_hours * hourly_rate

    # 1-year CUD: 37% discount
    cud_1yr_rate = hourly_rate * 0.63
    cud_1yr_monthly = monthly_usage_hours * cud_1yr_rate

    # 3-year CUD: 55% discount
    cud_3yr_rate = hourly_rate * 0.45
    cud_3yr_monthly = monthly_usage_hours * cud_3yr_rate

    return {
        "on_demand": on_demand_monthly,
        "1yr_cud": cud_1yr_monthly,
        "3yr_cud": cud_3yr_monthly,
        "1yr_savings": on_demand_monthly - cud_1yr_monthly,
        "3yr_savings": on_demand_monthly - cud_3yr_monthly
    }

# Example: 8x V100 GPUs, 720 hours/month (24/7)
savings = calculate_cud_savings(
    monthly_usage_hours=720,
    hourly_rate=24.80  # 8x V100 on-demand
)
# Result: $17,856/mo on-demand → $11,249/mo (1yr CUD) → $8,035/mo (3yr CUD)
```

### Sustained Use Discounts

From [GCP Sustained Use Discounts](https://cloud.google.com/compute/docs/sustained-use-discounts) (accessed 2025-01-31):
- Automatic discounts for workloads running >25% of month
- Up to 30% discount for continuous usage
- Applies automatically to on-demand instances

**Usage Pattern for Maximum Savings:**

```python
# Cost optimization strategy: Mix preemptible + CUD + sustained use
training_strategy = {
    "baseline_capacity": {
        "type": "committed_use_discount",
        "machines": "8x V100 (3-year CUD)",
        "cost_per_hour": 11.16,  # 55% discount
        "usage": "24/7 continuous"
    },
    "burst_capacity": {
        "type": "preemptible",
        "machines": "up to 32x V100",
        "cost_per_hour": 5.94,  # 76% discount
        "usage": "as needed with checkpoint-resume"
    },
    "peak_capacity": {
        "type": "on_demand_with_sustained_use",
        "machines": "additional 16x V100",
        "cost_per_hour": 17.36,  # 30% sustained use discount
        "usage": "high-priority jobs"
    }
}
```

### Resource Right-Sizing

**Machine Type Selection:**

```python
# Profile training job to determine optimal machine type
def profile_training_requirements(model, sample_batch):
    """Profile compute, memory, and I/O requirements"""

    import torch.profiler as profiler

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        # Run training step
        output = model(sample_batch)
        loss = criterion(output, labels)
        loss.backward()

    # Analyze resource usage
    stats = prof.key_averages()

    recommendations = {
        "cpu_cores": estimate_cpu_cores(stats),
        "memory_gb": estimate_memory_gb(stats),
        "gpu_type": recommend_gpu_type(stats),
        "disk_iops": estimate_disk_requirements(stats)
    }

    return recommendations

# Right-size machine configuration
def select_machine_type(requirements):
    """Select cost-optimal machine type"""

    if requirements["memory_gb"] < 32:
        machine_type = "n1-standard-8"
    elif requirements["memory_gb"] < 64:
        machine_type = "n1-standard-16"
    else:
        machine_type = "n1-highmem-32"

    # GPU selection based on workload
    if requirements["model_size_gb"] < 16:
        gpu_type = "NVIDIA_TESLA_V100"  # 16GB, $2.48/hr
    elif requirements["model_size_gb"] < 32:
        gpu_type = "NVIDIA_TESLA_A100"  # 40GB, $3.67/hr
    else:
        gpu_type = "NVIDIA_TESLA_A100_80GB"  # 80GB, $4.12/hr

    return machine_type, gpu_type
```

### Cost Anomaly Detection

```python
# Detect unexpected cost spikes
from google.cloud import monitoring_v3

def setup_cost_alerts(project_id, budget_threshold):
    """Configure alerts for cost anomalies"""

    client = monitoring_v3.AlertPolicyServiceClient()

    # Alert when daily spend exceeds threshold
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="ML Training Cost Anomaly",
        conditions=[monitoring_v3.AlertPolicy.Condition(
            display_name=f"Daily spend > ${budget_threshold}",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter='resource.type="global" '
                       'metric.type="serviceruntime.googleapis.com/api/request_count"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=budget_threshold,
                duration={"seconds": 86400}  # 24 hours
            )
        )]
    )

    project_name = f"projects/{project_id}"
    client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

### Budget Alerts and Quotas

```bash
# Set up Cloud Billing budget with alerts
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="ML Training Budget" \
    --budget-amount=50000 \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=75 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100 \
    --notification-pubsub-topic=projects/PROJECT_ID/topics/budget-alerts

# Set quota limits for cost protection
gcloud compute project-info add-metadata \
    --metadata=google-compute-default-region=us-central1

# Request quota increase for GPUs
gcloud compute regions describe us-central1 \
    --format="value(quotas.filter(metric:nvidia_v100_gpus))"
```

## Section 3: Production Monitoring (~200 lines)

### Cloud Logging Integration

From [Cloud Monitoring Metrics for Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/general/monitoring-metrics) (accessed 2025-01-31):
- Vertex AI exports training metrics to Cloud Monitoring automatically
- Custom metrics can be published from training containers
- Integration with Cloud Logging for centralized log management

**Structured Logging for Training:**

```python
# Structured logging to Cloud Logging
import logging
from google.cloud import logging as cloud_logging

class CloudLogger:
    def __init__(self, project_id, job_name):
        self.client = cloud_logging.Client(project=project_id)
        self.logger = self.client.logger(f"vertex-training-{job_name}")

    def log_training_metrics(self, epoch, step, metrics):
        """Log training metrics with structured data"""

        log_entry = {
            "severity": "INFO",
            "jsonPayload": {
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "job_type": "training"
            }
        }

        self.logger.log_struct(log_entry)

    def log_error(self, error_msg, exception=None):
        """Log errors with stack trace"""

        log_entry = {
            "severity": "ERROR",
            "jsonPayload": {
                "error_message": error_msg,
                "exception": str(exception) if exception else None,
                "stack_trace": traceback.format_exc() if exception else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        self.logger.log_struct(log_entry)

# Usage in training loop
logger = CloudLogger(project_id="my-project", job_name="llm-training-001")

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, batch)

        # Log metrics every 10 steps
        if step % 10 == 0:
            logger.log_training_metrics(
                epoch=epoch,
                step=step,
                metrics={
                    "loss": float(loss),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gpu_memory_mb": get_gpu_memory_usage(),
                    "throughput_samples_per_sec": calculate_throughput()
                }
            )
```

### Cloud Monitoring Dashboards

From [Monitor Agent with Cloud Monitoring](https://docs.cloud.google.com/agent-builder/agent-engine/manage/monitoring) (accessed 2025-01-31):
- Create custom dashboards for training job monitoring
- Configure alerting policies for anomalies
- Use log-based metrics for custom KPIs

**Custom Monitoring Dashboard:**

```python
# Create Cloud Monitoring dashboard for training
from google.cloud import monitoring_dashboard_v1

def create_training_dashboard(project_id):
    """Create comprehensive monitoring dashboard"""

    client = monitoring_dashboard_v1.DashboardsServiceClient()

    dashboard = monitoring_dashboard_v1.Dashboard(
        display_name="Vertex AI Training Dashboard",
        grid_layout=monitoring_dashboard_v1.GridLayout(
            widgets=[
                # GPU Utilization widget
                monitoring_dashboard_v1.Widget(
                    title="GPU Utilization",
                    xy_chart=monitoring_dashboard_v1.XyChart(
                        data_sets=[{
                            "time_series_query": {
                                "time_series_filter": {
                                    "filter": 'resource.type="aiplatform.googleapis.com/CustomJob" '
                                             'metric.type="aiplatform.googleapis.com/job/gpu_utilization"',
                                    "aggregation": {
                                        "alignment_period": {"seconds": 60},
                                        "per_series_aligner": "ALIGN_MEAN"
                                    }
                                }
                            }
                        }]
                    )
                ),
                # Training Loss widget
                monitoring_dashboard_v1.Widget(
                    title="Training Loss",
                    xy_chart=monitoring_dashboard_v1.XyChart(
                        data_sets=[{
                            "time_series_query": {
                                "time_series_filter": {
                                    "filter": 'resource.type="generic_task" '
                                             'metric.type="custom.googleapis.com/training/loss"',
                                    "aggregation": {
                                        "alignment_period": {"seconds": 300},
                                        "per_series_aligner": "ALIGN_MEAN"
                                    }
                                }
                            }
                        }]
                    )
                ),
                # Cost widget
                monitoring_dashboard_v1.Widget(
                    title="Training Cost (Last 24h)",
                    scorecard=monitoring_dashboard_v1.Scorecard(
                        time_series_query={
                            "time_series_filter": {
                                "filter": 'resource.type="global" '
                                         'metric.type="serviceruntime.googleapis.com/api/request_count"'
                            }
                        }
                    )
                )
            ]
        )
    )

    project_name = f"projects/{project_id}"
    response = client.create_dashboard(parent=project_name, dashboard=dashboard)
    return response
```

### Custom Metrics and Alerting

**Publishing Custom Metrics:**

```python
# Publish custom training metrics to Cloud Monitoring
from google.cloud import monitoring_v3
import time

class MetricsPublisher:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"

    def publish_metric(self, metric_type, value, labels=None):
        """Publish custom metric to Cloud Monitoring"""

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"

        if labels:
            for key, val in labels.items():
                series.metric.labels[key] = val

        series.resource.type = "generic_task"
        series.resource.labels["project_id"] = self.project_id
        series.resource.labels["location"] = "us-central1"
        series.resource.labels["namespace"] = "vertex-training"
        series.resource.labels["job"] = "training-job"
        series.resource.labels["task_id"] = os.environ.get("CLOUD_RUN_TASK_INDEX", "0")

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10 ** 9)

        interval = monitoring_v3.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )

        point = monitoring_v3.Point({
            "interval": interval,
            "value": {"double_value": value}
        })

        series.points = [point]
        self.client.create_time_series(name=self.project_name, time_series=[series])

# Usage in training
metrics_publisher = MetricsPublisher(project_id="my-project")

# Publish training metrics
metrics_publisher.publish_metric(
    metric_type="training/loss",
    value=current_loss,
    labels={"model": "llm-7b", "epoch": str(epoch)}
)

metrics_publisher.publish_metric(
    metric_type="training/throughput",
    value=samples_per_second,
    labels={"model": "llm-7b"}
)
```

### Error Reporting Integration

From [Cloud Error Reporting](https://cloud.google.com/error-reporting/docs) (accessed 2025-01-31):
- Automatic error grouping and analysis
- Integration with Cloud Logging
- Real-time error notifications

**Error Reporting Setup:**

```python
# Integrate Error Reporting with training code
from google.cloud import error_reporting

def setup_error_reporting(project_id):
    """Initialize error reporting client"""
    return error_reporting.Client(project=project_id)

# Report exceptions to Error Reporting
error_client = setup_error_reporting("my-project")

try:
    # Training code
    model.train()
except Exception as e:
    # Report error with context
    error_client.report_exception(
        http_context={
            "method": "TRAIN",
            "url": "vertex-ai/training-job",
            "userAgent": "vertex-custom-job"
        }
    )
    raise
```

### Performance Profiling

**Cloud Profiler Integration:**

```python
# Enable Cloud Profiler for performance analysis
import googlecloudprofiler

def enable_profiling(project_id, service_name):
    """Enable Cloud Profiler for training job"""

    try:
        googlecloudprofiler.start(
            service=service_name,
            service_version='1.0.0',
            project_id=project_id,
            verbose=3
        )
    except Exception as e:
        print(f"Failed to start profiler: {e}")

# Enable at training start
enable_profiling(
    project_id="my-project",
    service_name="vertex-training-llm"
)
```

### W&B + Cloud Monitoring Integration

**Dual Monitoring Strategy:**

```python
# Integrate W&B with Cloud Monitoring
import wandb
from google.cloud import monitoring_v3

class DualMonitoring:
    def __init__(self, project_id, wandb_project):
        # Initialize W&B
        wandb.init(project=wandb_project)

        # Initialize Cloud Monitoring
        self.metrics_publisher = MetricsPublisher(project_id)

    def log_metrics(self, metrics_dict):
        """Log to both W&B and Cloud Monitoring"""

        # Log to W&B (for experiment tracking)
        wandb.log(metrics_dict)

        # Log to Cloud Monitoring (for production alerting)
        for metric_name, value in metrics_dict.items():
            self.metrics_publisher.publish_metric(
                metric_type=f"training/{metric_name}",
                value=value
            )

# Usage
monitor = DualMonitoring(
    project_id="my-project",
    wandb_project="llm-training"
)

monitor.log_metrics({
    "loss": current_loss,
    "accuracy": current_accuracy,
    "learning_rate": current_lr
})
```

### Incident Response Patterns

**Automated Incident Response:**

```python
# Automated response to training anomalies
def setup_incident_response(project_id):
    """Configure automated responses to alerts"""

    # Example: Auto-terminate jobs with divergent loss
    alert_policy = {
        "display_name": "Training Loss Divergence",
        "conditions": [{
            "display_name": "Loss > 10x moving average",
            "condition_threshold": {
                "filter": 'metric.type="custom.googleapis.com/training/loss"',
                "comparison": "COMPARISON_GT",
                "threshold_value": 10.0,
                "duration": {"seconds": 600}
            }
        }],
        "notification_channels": [],
        "alert_strategy": {
            "notification_rate_limit": {"period": {"seconds": 300}}
        },
        # Trigger Cloud Function for automated response
        "user_labels": {
            "automated_response": "terminate_job",
            "severity": "critical"
        }
    }

    # Cloud Function triggered by alert
    # def respond_to_alert(event, context):
    #     job_id = event["incident"]["resource"]["labels"]["job_id"]
    #     aiplatform.CustomJob(job_id).cancel()
```

### Complete Monitoring Setup

```python
# End-to-end monitoring setup for production training
def setup_production_monitoring(project_id, job_name):
    """Initialize all monitoring components"""

    # 1. Cloud Logging
    logger = CloudLogger(project_id, job_name)

    # 2. Custom Metrics
    metrics_publisher = MetricsPublisher(project_id)

    # 3. Error Reporting
    error_client = setup_error_reporting(project_id)

    # 4. Cloud Profiler
    enable_profiling(project_id, f"vertex-training-{job_name}")

    # 5. W&B Integration
    wandb.init(project=f"vertex-{job_name}")

    # 6. Create Dashboard
    create_training_dashboard(project_id)

    # 7. Setup Alerts
    setup_cost_alerts(project_id, budget_threshold=10000)
    setup_sla_alert(project_id)

    return {
        "logger": logger,
        "metrics": metrics_publisher,
        "errors": error_client
    }
```

## Sources

**Web Research (accessed 2025-01-31):**
- [Vertex AI Platform SLA](https://cloud.google.com/vertex-ai/sla) - Service level agreements and uptime guarantees
- [GCP Disaster Recovery Architecture](https://docs.cloud.google.com/architecture/disaster-recovery) - Multi-region deployment patterns
- [GCP Multi-Regional VMs](https://docs.cloud.google.com/architecture/multiregional-vms) - Multi-region architecture guide
- [GCP Preemptible VM Pricing](https://cloud.google.com/spot-vms/pricing) - Spot and preemptible instance pricing
- [Preemptible VM Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) - Preemptible instance details
- [GCP VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) - Comprehensive VM pricing guide
- [Reduce ML Costs with Preemptible VMs](https://cloud.google.com/blog/products/ai-machine-learning/reduce-the-costs-of-ml-workflows-with-preemptible-vms-and-gpus) - Cost optimization strategies
- [ML Performance and Cost Optimization](https://cloud.google.com/blog/products/ai-machine-learning/machine-learning-performance-and-cost-optimization-best-practices) - Best practices guide
- [Cloud Monitoring Metrics for Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/general/monitoring-metrics) - Monitoring integration
- [Monitor Vertex AI Agent](https://docs.cloud.google.com/agent-builder/agent-engine/manage/monitoring) - Custom metrics and alerts

**Related Documentation:**
- [35-vertex-ai-production-patterns.md](35-vertex-ai-production-patterns.md) - This document
- [30-vertex-ai-fundamentals.md](30-vertex-ai-fundamentals.md) - Vertex AI basics
- [31-wandb-launch-vertex-agent.md](31-wandb-launch-vertex-agent.md) - Launch agent setup
- [32-vertex-ai-gpu-tpu.md](32-vertex-ai-gpu-tpu.md) - Accelerator management
