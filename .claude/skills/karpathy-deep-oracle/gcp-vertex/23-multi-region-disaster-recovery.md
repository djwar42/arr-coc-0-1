# Multi-Region Disaster Recovery for Vertex AI

## Overview

Production ML systems require robust disaster recovery (DR) strategies to ensure high availability, data protection, and business continuity. Multi-region architectures on Vertex AI provide resilience against regional outages, reduce latency for distributed teams, and meet compliance requirements for data residency.

From [Vertex AI SLA](https://cloud.google.com/vertex-ai/sla) (accessed 2025-11-16):
- Vertex AI provides 99.5% Monthly Uptime Percentage SLO for training services
- Prediction service endpoints offer 99.9% uptime for online inference
- Multi-region deployments significantly increase availability beyond single-region SLAs
- Regional outages can affect training jobs, model serving, and data access

From [GCP Multi-Region Architecture](https://docs.cloud.google.com/architecture/multiregional-vms) (accessed 2025-11-16):
- Deploy infrastructure across multiple GCP regions (us-central1, us-east1, europe-west1, asia-northeast1)
- Use regional Cloud Storage buckets with cross-region replication for data durability
- Configure regional Artifact Registry repositories for container image availability
- Implement global load balancing for automatic failover and traffic distribution

## Section 1: Multi-Region Architecture Patterns (~120 lines)

### Active-Active Architecture

**Pattern Overview:**

Active-active deployments run workloads simultaneously in multiple regions, distributing traffic across all regions for maximum availability and performance.

**Benefits:**
- Zero downtime during regional failures (automatic failover)
- Load distribution across multiple regions (reduces latency globally)
- Higher aggregate capacity (combined resources from all regions)
- Continuous testing of failover mechanisms (all regions always active)

**Trade-offs:**
- Higher cost (2x-3x resources running simultaneously)
- Data consistency challenges (multi-region writes)
- Complex orchestration (coordinating deployments across regions)
- Increased network costs (cross-region data transfer)

From [Google Cloud Multi-Regional Deployment](https://docs.cloud.google.com/architecture/deployment-archetypes/multiregional) (accessed 2025-11-16):
- Active-active requires global load balancing with health checks
- Data must be replicated with eventual or strong consistency
- Application must handle split-brain scenarios (network partitions)

**Active-Active Vertex AI Implementation:**

```python
# Active-active model serving across regions
from google.cloud import aiplatform
from google.cloud import storage

class MultiRegionModelDeployment:
    def __init__(self, project_id, model_id):
        self.project_id = project_id
        self.model_id = model_id
        self.regions = ["us-central1", "europe-west1", "asia-northeast1"]

    def deploy_active_active(self):
        """Deploy model to multiple regions simultaneously"""

        endpoints = {}

        for region in self.regions:
            # Initialize Vertex AI client for each region
            aiplatform.init(project=self.project_id, location=region)

            # Create regional endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=f"arr-coc-model-{region}",
                description=f"Active-active endpoint in {region}"
            )

            # Deploy model to regional endpoint
            model = aiplatform.Model(model_name=self.model_id)
            endpoint.deploy(
                model=model,
                deployed_model_display_name=f"arr-coc-v1-{region}",
                machine_type="n1-standard-8",
                min_replica_count=2,  # HA within region
                max_replica_count=10,
                traffic_percentage=100,
                # Enable autoscaling
                autoscaling_target_cpu_utilization=70,
                autoscaling_target_accelerator_duty_cycle=70
            )

            endpoints[region] = endpoint

        return endpoints

    def configure_global_load_balancer(self, endpoints):
        """Configure Cloud Load Balancing for global traffic distribution"""

        # Global load balancer distributes traffic based on:
        # 1. User proximity (latency-based routing)
        # 2. Endpoint health (automatic failover)
        # 3. Endpoint capacity (avoid overload)

        backend_config = {
            "backends": [
                {
                    "group": f"vertex-ai-{region}",
                    "balancing_mode": "UTILIZATION",
                    "max_utilization": 0.8,
                    "capacity_scaler": 1.0
                }
                for region in self.regions
            ],
            "health_checks": [{
                "check_interval_sec": 10,
                "timeout_sec": 5,
                "healthy_threshold": 2,
                "unhealthy_threshold": 3
            }],
            "timeout_sec": 30,
            "enable_cdn": False  # ML predictions not cacheable
        }

        return backend_config
```

### Active-Passive Architecture

**Pattern Overview:**

Active-passive deployments maintain a primary region for all traffic, with standby regions ready to take over during failures.

**Benefits:**
- Lower cost (standby resources can be minimal or cold)
- Simpler data consistency (single write location)
- Easier to reason about (clear primary region)
- Predictable costs (only pay for standby when needed)

**Trade-offs:**
- Recovery time objective (RTO) higher than active-active (requires failover)
- Standby region may be untested (failover surprises)
- Geographic concentration of traffic (single active region)
- Manual or automated failover required (complexity)

**Active-Passive Vertex AI Implementation:**

```python
# Active-passive model serving with manual failover
class ActivePassiveDeployment:
    def __init__(self, project_id, model_id):
        self.project_id = project_id
        self.model_id = model_id
        self.primary_region = "us-central1"
        self.standby_regions = ["us-east1", "europe-west1"]

    def deploy_active_passive(self):
        """Deploy model with primary and standby regions"""

        # Primary region: Full deployment
        aiplatform.init(project=self.project_id, location=self.primary_region)

        primary_endpoint = aiplatform.Endpoint.create(
            display_name=f"arr-coc-primary-{self.primary_region}"
        )

        model = aiplatform.Model(model_name=self.model_id)
        primary_endpoint.deploy(
            model=model,
            machine_type="n1-standard-8",
            min_replica_count=3,
            max_replica_count=20,
            traffic_percentage=100
        )

        # Standby regions: Minimal deployment (warm standby)
        standby_endpoints = {}
        for region in self.standby_regions:
            aiplatform.init(project=self.project_id, location=region)

            standby_endpoint = aiplatform.Endpoint.create(
                display_name=f"arr-coc-standby-{region}"
            )

            # Deploy with minimal replicas (warm standby)
            standby_endpoint.deploy(
                model=model,
                machine_type="n1-standard-8",
                min_replica_count=1,  # Minimal for fast scale-up
                max_replica_count=20,
                traffic_percentage=0  # No traffic until failover
            )

            standby_endpoints[region] = standby_endpoint

        return primary_endpoint, standby_endpoints

    def failover_to_standby(self, standby_region):
        """Execute failover to standby region"""

        # 1. Scale up standby endpoint
        aiplatform.init(project=self.project_id, location=standby_region)

        standby_endpoint = aiplatform.Endpoint(
            endpoint_name=f"arr-coc-standby-{standby_region}"
        )

        # Update to full production capacity
        standby_endpoint.update(
            min_replica_count=3,
            traffic_percentage=100
        )

        # 2. Update DNS/Load Balancer to route to standby
        # (Implementation depends on DNS provider)

        # 3. Monitor health and confirm failover
        print(f"Failover complete to {standby_region}")
```

### Pilot Light Architecture

**Pattern Overview:**

Pilot light maintains minimal infrastructure in standby regions (just data replication), with ability to rapidly spin up compute resources when needed.

**Benefits:**
- Lowest cost (data replication only, no compute in standby)
- Data always ready for recovery (RPO near-zero for data)
- Can scale standby to any size (not constrained by pre-allocated resources)

**Trade-offs:**
- Highest RTO (15-30 minutes to provision and deploy)
- Requires automation for rapid scale-up
- Untested standby resources (first real test is during disaster)

From [GCP Disaster Recovery Planning](https://docs.cloud.google.com/architecture/disaster-recovery) (accessed 2025-11-16):
- Pilot light suitable for RPO < 1 hour, RTO < 4 hours
- Data replication must be continuous (GCS multi-region, BigQuery cross-region)
- Infrastructure-as-Code essential for rapid provisioning

## Section 2: Model Registry Replication (~120 lines)

### Cross-Region Model Copy

From [Vertex AI Model Registry Copy](https://docs.cloud.google.com/vertex-ai/docs/model-registry/copy-model) (accessed 2025-11-16):
- Copy models between regions within same project
- Copy models as new model or new version of existing model
- Source and destination models can have different names
- Model artifacts (SavedModel, ONNX, etc.) are copied to destination GCS bucket

**Model Copy Workflow:**

```python
# Cross-region model replication
from google.cloud import aiplatform
from google.cloud import storage

class ModelReplication:
    def __init__(self, project_id, source_region, dest_regions):
        self.project_id = project_id
        self.source_region = source_region
        self.dest_regions = dest_regions

    def replicate_model_registry(self, model_id):
        """Replicate model to multiple regions"""

        # 1. Get source model metadata
        aiplatform.init(project=self.project_id, location=self.source_region)
        source_model = aiplatform.Model(model_name=model_id)

        replicated_models = {}

        for dest_region in self.dest_regions:
            # 2. Copy model to destination region
            aiplatform.init(project=self.project_id, location=dest_region)

            dest_model = source_model.copy(
                destination_location=dest_region,
                destination_model_id=f"{model_id}-{dest_region}",
                copy_ancestry=True  # Preserve lineage
            )

            replicated_models[dest_region] = dest_model

        return replicated_models
```

### GCS Bucket Synchronization

**Multi-Region Bucket Strategy:**

```python
# Continuous GCS bucket synchronization
from google.cloud import storage
import subprocess

class GCSReplication:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = storage.Client(project=project_id)

    def create_multi_region_bucket(self, bucket_name, location="US"):
        """Create multi-region bucket for automatic replication"""

        # Multi-region buckets replicate automatically across regions
        bucket = self.client.bucket(bucket_name)
        bucket.location = location  # US, EU, ASIA
        bucket.storage_class = "STANDARD"

        # Enable versioning for disaster recovery
        bucket.versioning_enabled = True

        # Create bucket
        bucket = self.client.create_bucket(bucket, location=location)

        return bucket

    def setup_cross_region_sync(self, source_bucket, dest_bucket):
        """Set up cross-region bucket synchronization"""

        # Use gsutil for efficient cross-region sync
        sync_command = [
            "gsutil", "-m", "rsync", "-r", "-d",
            f"gs://{source_bucket}/",
            f"gs://{dest_bucket}/"
        ]

        # Run sync (can be scheduled via Cloud Scheduler)
        result = subprocess.run(sync_command, capture_output=True, text=True)

        return result.returncode == 0

    def configure_lifecycle_policy(self, bucket_name):
        """Configure lifecycle management for cost optimization"""

        bucket = self.client.bucket(bucket_name)

        # Move old checkpoints to cheaper storage classes
        lifecycle_rules = [
            {
                "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
                "condition": {"age": 30}  # After 30 days
            },
            {
                "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
                "condition": {"age": 90}  # After 90 days
            },
            {
                "action": {"type": "Delete"},
                "condition": {"age": 365}  # Delete after 1 year
            }
        ]

        bucket.lifecycle_rules = lifecycle_rules
        bucket.patch()
```

### Container Image Replication

**Artifact Registry Multi-Region:**

```bash
# Push images to multiple regional registries
PROJECT_ID="my-project"
IMAGE_NAME="arr-coc-training"
VERSION="v1.0"

# Build image once
docker build -t ${IMAGE_NAME}:${VERSION} .

# Push to multiple regions
for region in us-central1 us-east1 europe-west1 asia-northeast1; do
    # Tag for regional registry
    docker tag ${IMAGE_NAME}:${VERSION} \
        ${region}-docker.pkg.dev/${PROJECT_ID}/ml-images/${IMAGE_NAME}:${VERSION}

    # Push to regional Artifact Registry
    docker push ${region}-docker.pkg.dev/${PROJECT_ID}/ml-images/${IMAGE_NAME}:${VERSION}
done
```

## Section 3: Endpoint Failover with Global Load Balancing (~120 lines)

### Global Load Balancer Configuration

From [GCP Load Balancing](https://cloud.google.com/load-balancing/docs/https) (accessed 2025-11-16):
- Global Application Load Balancer distributes traffic across regions
- Health checks monitor backend endpoint availability
- Automatic failover when backends become unhealthy
- SSL/TLS termination at load balancer edge

**Health Check Configuration:**

```python
# Configure health checks for Vertex AI endpoints
from google.cloud import compute_v1

class VertexAIHealthCheck:
    def __init__(self, project_id):
        self.project_id = project_id
        self.health_check_client = compute_v1.HealthChecksClient()

    def create_health_check(self):
        """Create health check for Vertex AI endpoints"""

        health_check = compute_v1.HealthCheck()
        health_check.name = "vertex-ai-endpoint-health"
        health_check.type_ = "HTTPS"

        # HTTPS health check configuration
        https_health_check = compute_v1.HTTPSHealthCheck()
        https_health_check.port = 443
        https_health_check.request_path = "/v1/health"  # Endpoint health path
        https_health_check.check_interval_sec = 10
        https_health_check.timeout_sec = 5
        https_health_check.healthy_threshold = 2
        https_health_check.unhealthy_threshold = 3

        health_check.https_health_check = https_health_check

        # Create health check
        operation = self.health_check_client.insert(
            project=self.project_id,
            health_check_resource=health_check
        )

        return operation
```

**Automatic Failover Logic:**

```python
# Monitoring and automatic failover
from google.cloud import monitoring_v3
import time

class AutomaticFailover:
    def __init__(self, project_id, primary_region, standby_region):
        self.project_id = project_id
        self.primary_region = primary_region
        self.standby_region = standby_region
        self.monitoring_client = monitoring_v3.MetricServiceClient()

    def monitor_endpoint_health(self, endpoint_id):
        """Monitor endpoint health and trigger failover if needed"""

        # Query Cloud Monitoring for endpoint health
        project_name = f"projects/{self.project_id}"

        # Health metric filter
        filter_str = (
            f'resource.type="aiplatform.googleapis.com/Endpoint" '
            f'resource.labels.endpoint_id="{endpoint_id}" '
            f'metric.type="aiplatform.googleapis.com/prediction/error_count"'
        )

        # Check health every minute
        while True:
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": int(time.time())},
                    "start_time": {"seconds": int(time.time()) - 300}  # Last 5 minutes
                }
            )

            results = self.monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": filter_str,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                }
            )

            # Calculate error rate
            total_errors = sum(
                point.value.int64_value
                for result in results
                for point in result.points
            )

            # Trigger failover if error rate exceeds threshold
            if total_errors > 100:  # 100 errors in 5 minutes
                print(f"Error threshold exceeded: {total_errors} errors")
                self.execute_failover()
                break

            time.sleep(60)  # Check every minute

    def execute_failover(self):
        """Execute automatic failover to standby region"""

        print(f"Executing failover from {self.primary_region} to {self.standby_region}")

        # 1. Update load balancer to route traffic to standby
        # 2. Scale up standby endpoint
        # 3. Send alerts to operators
        # 4. Update monitoring dashboards
```

### Traffic Splitting and Gradual Migration

**Blue-Green Deployment Across Regions:**

```python
# Gradual traffic migration between regions
class TrafficMigration:
    def __init__(self, project_id):
        self.project_id = project_id

    def gradual_traffic_shift(self, source_endpoint, dest_endpoint, duration_minutes=30):
        """Gradually shift traffic from source to destination"""

        # Traffic shift schedule: 0% -> 10% -> 25% -> 50% -> 75% -> 100%
        traffic_steps = [0, 10, 25, 50, 75, 100]
        step_duration = duration_minutes // len(traffic_steps)

        for traffic_percentage in traffic_steps:
            # Update traffic split
            source_endpoint.update(
                traffic_percentage=100 - traffic_percentage
            )
            dest_endpoint.update(
                traffic_percentage=traffic_percentage
            )

            print(f"Traffic split: {100 - traffic_percentage}% source, {traffic_percentage}% dest")

            # Monitor for errors during shift
            time.sleep(step_duration * 60)

            # Check error rates, rollback if needed
            if self.check_error_rate_spike(dest_endpoint):
                print("Error rate spike detected, rolling back")
                self.rollback_traffic(source_endpoint, dest_endpoint)
                return False

        return True

    def check_error_rate_spike(self, endpoint):
        """Check if error rate has spiked during migration"""
        # Query monitoring metrics
        # Return True if error rate > baseline * 2
        return False

    def rollback_traffic(self, source_endpoint, dest_endpoint):
        """Immediately rollback traffic to source"""
        source_endpoint.update(traffic_percentage=100)
        dest_endpoint.update(traffic_percentage=0)
```

## Section 4: Data Replication Strategies (~120 lines)

### Multi-Region Cloud Storage

**Dual-Region and Multi-Region Buckets:**

From [GCS Multi-Region Storage](https://cloud.google.com/storage/docs/locations#location-mr) (accessed 2025-11-16):
- Multi-region buckets: Data replicated across >= 2 regions within geo (US, EU, ASIA)
- Dual-region buckets: Data replicated across exactly 2 regions (user-specified)
- 99.95% availability SLA for multi-region (vs 99.9% for regional)
- Automatic replication, no configuration needed

**Bucket Configuration for ML Workloads:**

```python
# Configure multi-region buckets for ML data
from google.cloud import storage

class MLDataReplication:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = storage.Client(project=project_id)

    def create_training_data_bucket(self):
        """Create multi-region bucket for training data"""

        # Multi-region for maximum availability
        bucket = self.client.bucket("arr-coc-training-data")
        bucket.location = "US"  # Covers us-central1, us-east1, us-west1
        bucket.storage_class = "STANDARD"
        bucket.versioning_enabled = True

        # Create bucket
        bucket = self.client.create_bucket(bucket, location="US")

        return bucket

    def create_checkpoint_bucket(self):
        """Create dual-region bucket for model checkpoints"""

        # Dual-region for cost-optimized replication
        bucket = self.client.bucket("arr-coc-checkpoints")
        bucket.location = "US"
        bucket.storage_class = "NEARLINE"  # Cheaper for infrequent access
        bucket.versioning_enabled = True

        # Custom placement policy (dual-region)
        bucket.location_type = "dual-region"
        bucket.data_locations = ["US-CENTRAL1", "US-EAST1"]

        bucket = self.client.create_bucket(bucket)

        return bucket
```

### BigQuery Cross-Region Dataset Replication

**Dataset Transfer Service:**

```python
# BigQuery cross-region dataset replication
from google.cloud import bigquery_datatransfer_v1

class BigQueryReplication:
    def __init__(self, project_id, source_region, dest_region):
        self.project_id = project_id
        self.source_region = source_region
        self.dest_region = dest_region
        self.transfer_client = bigquery_datatransfer_v1.DataTransferServiceClient()

    def create_cross_region_copy(self, source_dataset, dest_dataset):
        """Create scheduled dataset copy to destination region"""

        # Transfer config for cross-region copy
        transfer_config = bigquery_datatransfer_v1.TransferConfig(
            display_name=f"DR copy {source_dataset} to {dest_dataset}",
            data_source_id="cross_region_copy",
            destination_dataset_id=dest_dataset,
            schedule="every day 02:00",  # Daily at 2 AM
            params={
                "source_project_id": self.project_id,
                "source_dataset_id": source_dataset,
                "overwrite_destination_table": True
            }
        )

        # Create transfer
        parent = f"projects/{self.project_id}/locations/{self.dest_region}"
        transfer = self.transfer_client.create_transfer_config(
            parent=parent,
            transfer_config=transfer_config
        )

        return transfer
```

### Continuous Data Synchronization

**Pub/Sub for Real-Time Replication:**

```python
# Real-time data replication via Pub/Sub
from google.cloud import pubsub_v1

class RealtimeReplication:
    def __init__(self, project_id):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()

    def publish_data_change(self, data, regions):
        """Publish data changes to multiple regional topics"""

        futures = []

        for region in regions:
            topic_path = self.publisher.topic_path(
                self.project_id,
                f"data-sync-{region}"
            )

            # Publish to regional topic
            future = self.publisher.publish(
                topic_path,
                data=data.encode("utf-8"),
                region=region
            )

            futures.append(future)

        # Wait for all publishes to complete
        for future in futures:
            future.result()
```

## Section 5: Disaster Recovery Testing (~100 lines)

### Chaos Engineering for ML Systems

**Controlled Failure Injection:**

```python
# Chaos engineering for DR testing
import random
from google.cloud import aiplatform

class ChaosEngineering:
    def __init__(self, project_id):
        self.project_id = project_id

    def simulate_regional_outage(self, region):
        """Simulate regional outage by stopping all endpoints"""

        aiplatform.init(project=self.project_id, location=region)

        # List all endpoints in region
        endpoints = aiplatform.Endpoint.list()

        # Undeploy models from all endpoints (simulate outage)
        for endpoint in endpoints:
            deployed_models = endpoint.list_models()
            for model in deployed_models:
                endpoint.undeploy(deployed_model_id=model.id)

        print(f"Simulated outage in {region}")

    def test_failover_time(self, primary_region, standby_region):
        """Measure time to failover to standby region"""

        import time

        start_time = time.time()

        # 1. Simulate primary region failure
        self.simulate_regional_outage(primary_region)

        # 2. Trigger failover to standby
        # (Automatic via health checks or manual)

        # 3. Verify standby region serving traffic
        aiplatform.init(project=self.project_id, location=standby_region)
        standby_endpoint = aiplatform.Endpoint.list()[0]

        # Test prediction
        test_instance = {"text": "test"}
        prediction = standby_endpoint.predict(instances=[test_instance])

        end_time = time.time()
        rto = end_time - start_time

        print(f"Failover completed in {rto:.2f} seconds")

        return rto
```

### DR Drill Runbooks

**Monthly Failover Drills:**

```yaml
# DR drill runbook (executed monthly)
disaster_recovery_drill:
  schedule: "First Saturday of each month at 2 AM PST"

  steps:
    - name: "Pre-drill checks"
      tasks:
        - Verify all regions healthy
        - Confirm standby endpoints deployed
        - Alert on-call team of drill

    - name: "Execute failover"
      tasks:
        - Simulate primary region failure (us-central1)
        - Verify automatic failover to us-east1
        - Measure failover time (target: < 5 minutes)

    - name: "Validate recovery"
      tasks:
        - Send test predictions to standby endpoint
        - Verify model responses correct
        - Check monitoring dashboards

    - name: "Failback to primary"
      tasks:
        - Restore primary region
        - Gradually shift traffic back (10% increments)
        - Confirm primary region stable

    - name: "Post-drill analysis"
      tasks:
        - Document observed RTO/RPO
        - Identify issues encountered
        - Update runbooks with improvements
```

**Chaos Testing Schedule:**

```python
# Scheduled chaos tests
class ScheduledChaosTests:
    def __init__(self):
        self.tests = [
            {
                "name": "Regional Endpoint Failure",
                "frequency": "weekly",
                "blast_radius": "single endpoint",
                "expected_rto": 60  # seconds
            },
            {
                "name": "Regional GCS Outage",
                "frequency": "monthly",
                "blast_radius": "single region storage",
                "expected_rto": 120
            },
            {
                "name": "Cross-Region Network Partition",
                "frequency": "quarterly",
                "blast_radius": "inter-region connectivity",
                "expected_rto": 300
            }
        ]
```

## Section 6: RPO/RTO Targets for ML Systems (~120 lines)

### Defining Recovery Objectives

From [AWS Disaster Recovery Objectives](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/disaster-recovery-dr-objectives.html) (accessed 2025-11-16):
- RPO (Recovery Point Objective): Maximum acceptable data loss
- RTO (Recovery Time Objective): Maximum acceptable downtime
- Objectives vary by workload criticality and business impact

**ML Workload Classification:**

| Workload Type | RPO Target | RTO Target | DR Strategy | Example |
|--------------|------------|------------|-------------|---------|
| Research Training | 24 hours | 4 hours | Backup & Restore | Experimental model training |
| Production Training | 1 hour | 30 minutes | Pilot Light | Scheduled retraining jobs |
| Online Inference | 5 minutes | 2 minutes | Active-Passive | Production model serving |
| Critical Inference | 0 seconds | 0 seconds | Active-Active | Real-time fraud detection |

**Setting Realistic Targets:**

```python
# Calculate RPO/RTO based on business requirements
class RecoveryObjectives:
    def __init__(self, workload_type):
        self.workload_type = workload_type

    def calculate_rpo(self, data_generation_rate_gb_per_hour, data_value_per_gb):
        """Calculate acceptable RPO based on data value"""

        # Maximum data loss cost (business decision)
        max_acceptable_loss = 10000  # $10,000

        # Calculate RPO that keeps loss under limit
        rpo_hours = max_acceptable_loss / (data_generation_rate_gb_per_hour * data_value_per_gb)

        return rpo_hours

    def calculate_rto(self, revenue_per_hour, downtime_cost_multiplier=2):
        """Calculate acceptable RTO based on revenue impact"""

        # Maximum revenue loss (business decision)
        max_acceptable_downtime_cost = 50000  # $50,000

        # Calculate RTO (includes reputation cost multiplier)
        total_cost_per_hour = revenue_per_hour * downtime_cost_multiplier
        rto_hours = max_acceptable_downtime_cost / total_cost_per_hour

        return rto_hours

# Example: Production inference endpoint
objectives = RecoveryObjectives("production_inference")

# Predict dataset: 100 GB/hour, $50/GB value
rpo = objectives.calculate_rpo(
    data_generation_rate_gb_per_hour=100,
    data_value_per_gb=50
)
print(f"Recommended RPO: {rpo:.2f} hours")

# Revenue: $10,000/hour, 2x reputation multiplier
rto = objectives.calculate_rto(
    revenue_per_hour=10000,
    downtime_cost_multiplier=2
)
print(f"Recommended RTO: {rto:.2f} hours")
```

### Checkpoint Strategy for Training RPO

**Continuous Checkpointing:**

```python
# Checkpoint strategy to achieve RPO targets
import torch
from google.cloud import storage

class CheckpointStrategy:
    def __init__(self, rpo_minutes=5):
        self.rpo_minutes = rpo_minutes
        self.gcs_client = storage.Client()

    def calculate_checkpoint_frequency(self, avg_training_time_per_epoch_minutes):
        """Calculate optimal checkpoint frequency"""

        # Checkpoint at least as often as RPO requires
        checkpoints_per_epoch = max(
            1,
            int(avg_training_time_per_epoch_minutes / self.rpo_minutes)
        )

        steps_per_checkpoint = total_steps_per_epoch // checkpoints_per_epoch

        return steps_per_checkpoint

    def save_checkpoint_multi_region(self, model, optimizer, epoch, step):
        """Save checkpoint to multiple regions for DR"""

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        # Save locally first (fast)
        local_path = f"/tmp/checkpoint_e{epoch}_s{step}.pt"
        torch.save(checkpoint, local_path)

        # Upload to multiple regional buckets (parallel)
        regions = ["us-central1", "us-east1", "europe-west1"]

        for region in regions:
            bucket_name = f"arr-coc-checkpoints-{region}"
            blob_name = f"checkpoints/checkpoint_e{epoch}_s{step}.pt"

            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)

        print(f"Checkpoint saved to {len(regions)} regions (RPO: {self.rpo_minutes} min)")
```

### Monitoring RPO/RTO Compliance

```python
# Monitor actual RPO/RTO vs targets
from google.cloud import monitoring_v3

class RPOMonitoring:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()

    def log_rpo_metric(self, last_checkpoint_time):
        """Log current RPO (time since last checkpoint)"""

        import time
        current_time = time.time()
        rpo_seconds = current_time - last_checkpoint_time

        # Create custom metric
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/training/rpo_seconds"
        series.resource.type = "generic_task"

        point = monitoring_v3.Point()
        point.value.double_value = rpo_seconds
        point.interval.end_time.seconds = int(current_time)

        series.points = [point]

        self.client.create_time_series(
            name=f"projects/{self.project_id}",
            time_series=[series]
        )

    def alert_rpo_violation(self, rpo_target_minutes=5):
        """Create alert for RPO violations"""

        # Alert when RPO exceeds target
        alert_policy = {
            "display_name": "Training RPO Violation",
            "conditions": [{
                "display_name": f"RPO > {rpo_target_minutes} minutes",
                "condition_threshold": {
                    "filter": 'metric.type="custom.googleapis.com/training/rpo_seconds"',
                    "comparison": "COMPARISON_GT",
                    "threshold_value": rpo_target_minutes * 60,
                    "duration": {"seconds": 60}
                }
            }]
        }
```

## Section 7: Cost Analysis (~100 lines)

### Single-Region vs Multi-Region Cost Comparison

**Cost Components:**

From [GCP Pricing Calculator](https://cloud.google.com/products/calculator) (accessed 2025-11-16):
- Compute: VM instances for training/inference
- Storage: GCS buckets (regional vs multi-region)
- Network: Cross-region data transfer ($0.01-0.08/GB)
- Load Balancing: Global LB forwarding rules ($0.025/hour + $0.008/GB)

**Example Cost Analysis:**

```python
# Cost comparison for arr-coc-0-1 deployment
class DRCostAnalysis:
    def __init__(self):
        # Pricing (simplified, actual varies by region)
        self.vm_cost_per_hour = {
            "n1-standard-8": 0.38,
            "n1-standard-16": 0.76
        }
        self.gpu_cost_per_hour = {
            "nvidia-tesla-v100": 2.48
        }
        self.storage_cost_per_gb_month = {
            "regional": 0.020,
            "dual_region": 0.026,
            "multi_region": 0.026
        }
        self.network_egress_per_gb = 0.08  # Cross-region

    def calculate_single_region_cost(self, hours_per_month=730):
        """Calculate single-region deployment cost"""

        # Inference endpoints: 3 replicas
        compute_cost = (
            self.vm_cost_per_hour["n1-standard-8"] * 3 * hours_per_month
        )

        # Storage: 1 TB training data + 500 GB checkpoints
        storage_cost = (
            1000 * self.storage_cost_per_gb_month["regional"] +
            500 * self.storage_cost_per_gb_month["regional"]
        )

        total = compute_cost + storage_cost

        return {
            "compute": compute_cost,
            "storage": storage_cost,
            "network": 0,
            "load_balancing": 0,
            "total": total
        }

    def calculate_active_passive_cost(self, hours_per_month=730):
        """Calculate active-passive multi-region cost"""

        # Primary: 3 replicas, Standby: 1 replica
        compute_cost = (
            self.vm_cost_per_hour["n1-standard-8"] * 3 * hours_per_month +  # Primary
            self.vm_cost_per_hour["n1-standard-8"] * 1 * hours_per_month    # Standby
        )

        # Storage: Multi-region for DR
        storage_cost = (
            1000 * self.storage_cost_per_gb_month["multi_region"] +
            500 * self.storage_cost_per_gb_month["dual_region"]
        )

        # Cross-region sync: 100 GB/day
        network_cost = 100 * 30 * self.network_egress_per_gb

        # Load balancer
        lb_cost = 0.025 * hours_per_month  # Forwarding rule

        total = compute_cost + storage_cost + network_cost + lb_cost

        return {
            "compute": compute_cost,
            "storage": storage_cost,
            "network": network_cost,
            "load_balancing": lb_cost,
            "total": total
        }

    def calculate_active_active_cost(self, num_regions=3, hours_per_month=730):
        """Calculate active-active multi-region cost"""

        # All regions: 3 replicas each
        compute_cost = (
            self.vm_cost_per_hour["n1-standard-8"] * 3 * num_regions * hours_per_month
        )

        # Storage: Multi-region
        storage_cost = (
            1000 * self.storage_cost_per_gb_month["multi_region"] +
            500 * self.storage_cost_per_gb_month["multi_region"]
        )

        # Cross-region sync: 200 GB/day (more frequent)
        network_cost = 200 * 30 * self.network_egress_per_gb * num_regions

        # Global load balancer
        lb_cost = 0.025 * hours_per_month + (1000 * 0.008)  # Rule + data processed

        total = compute_cost + storage_cost + network_cost + lb_cost

        return {
            "compute": compute_cost,
            "storage": storage_cost,
            "network": network_cost,
            "load_balancing": lb_cost,
            "total": total
        }

# Cost comparison
analyzer = DRCostAnalysis()

single = analyzer.calculate_single_region_cost()
active_passive = analyzer.calculate_active_passive_cost()
active_active = analyzer.calculate_active_active_cost()

print("Monthly Cost Comparison:")
print(f"Single Region:    ${single['total']:.2f}")
print(f"Active-Passive:   ${active_passive['total']:.2f} ({active_passive['total']/single['total']:.1f}x)")
print(f"Active-Active:    ${active_active['total']:.2f} ({active_active['total']/single['total']:.1f}x)")
```

**Cost Optimization Strategies:**

```python
# Optimize DR costs
class DRCostOptimization:
    def __init__(self):
        pass

    def optimize_standby_resources(self):
        """Recommendations for standby resource optimization"""

        optimizations = [
            {
                "strategy": "Use Spot VMs for standby",
                "savings": "60-91%",
                "trade_off": "May need re-provisioning during failover"
            },
            {
                "strategy": "Preemptible training in standby region",
                "savings": "70%",
                "trade_off": "Training interrupted, but data replicated"
            },
            {
                "strategy": "Nearline storage for old checkpoints",
                "savings": "50% on storage",
                "trade_off": "Slower retrieval (not for active checkpoints)"
            },
            {
                "strategy": "Scheduled standby scale-down",
                "savings": "30-50%",
                "trade_off": "Longer RTO during off-hours"
            }
        ]

        return optimizations
```

## Section 8: arr-coc-0-1 High Availability Deployment (~120 lines)

### Multi-Region Architecture for arr-coc-0-1

**Deployment Strategy:**

```python
# arr-coc-0-1 multi-region deployment
class ArrCocHADeployment:
    def __init__(self, project_id):
        self.project_id = project_id
        self.model_name = "arr-coc-vision-v1"
        self.regions = {
            "primary": "us-central1",
            "secondary": "us-east1",
            "tertiary": "europe-west1"
        }

    def deploy_multi_region(self):
        """Deploy arr-coc-0-1 with HA across 3 regions"""

        endpoints = {}

        for role, region in self.regions.items():
            aiplatform.init(project=self.project_id, location=region)

            # Create regional endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=f"arr-coc-{role}-{region}"
            )

            # Deploy model
            model = aiplatform.Model.upload(
                display_name=f"arr-coc-model-{region}",
                artifact_uri=f"gs://arr-coc-models-{region}/v1/",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest"
            )

            # Configure based on role
            if role == "primary":
                # Primary: Full capacity
                min_replicas = 3
                max_replicas = 20
            else:
                # Secondary/Tertiary: Warm standby
                min_replicas = 1
                max_replicas = 20

            endpoint.deploy(
                model=model,
                deployed_model_display_name=f"arr-coc-{role}",
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_V100",
                accelerator_count=1,
                min_replica_count=min_replicas,
                max_replica_count=max_replicas,
                traffic_percentage=100 if role == "primary" else 0
            )

            endpoints[role] = endpoint

        return endpoints
```

**arr-coc-0-1 Specific Considerations:**

```python
# Model-specific HA configuration
class ArrCocDRConfig:
    def __init__(self):
        # arr-coc-0-1 model characteristics
        self.model_size_gb = 4.5  # Base Qwen3-VL + adapters
        self.avg_inference_time_ms = 250
        self.peak_qps = 100

    def calculate_required_capacity(self, target_rto_minutes=5):
        """Calculate standby capacity needed for RTO target"""

        # Time to scale from 1 -> full capacity
        scale_up_time_minutes = 3  # Vertex AI autoscaling

        # Remaining time for traffic migration
        migration_time_minutes = target_rto_minutes - scale_up_time_minutes

        if migration_time_minutes < 0:
            # Need warm standby with more initial replicas
            required_standby_replicas = self.peak_qps / (1000 / self.avg_inference_time_ms)
            return int(required_standby_replicas)
        else:
            # Can start with minimal standby
            return 1

    def configure_checkpoint_replication(self):
        """Configure checkpoint replication for arr-coc training"""

        checkpoint_config = {
            "frequency_steps": 100,  # Checkpoint every 100 steps
            "rpo_minutes": 5,  # Maximum 5 minutes data loss
            "regions": ["us-central1", "us-east1", "europe-west1"],
            "storage_class": "DUAL_REGION",  # Cost-optimized
            "retention_days": 30
        }

        return checkpoint_config
```

**Monitoring Dashboard for arr-coc-0-1:**

```python
# arr-coc-0-1 specific monitoring
class ArrCocMonitoring:
    def __init__(self, project_id):
        self.project_id = project_id

    def create_ha_dashboard(self):
        """Create monitoring dashboard for HA metrics"""

        dashboard_config = {
            "displayName": "arr-coc-0-1 HA Dashboard",
            "gridLayout": {
                "widgets": [
                    {
                        "title": "Regional Endpoint Health",
                        "metrics": [
                            "aiplatform.googleapis.com/prediction/error_count",
                            "aiplatform.googleapis.com/prediction/online/response_count"
                        ],
                        "groupBy": ["resource.location"]
                    },
                    {
                        "title": "Cross-Region Latency",
                        "metrics": [
                            "aiplatform.googleapis.com/prediction/online/prediction_latencies"
                        ],
                        "aggregation": "p99"
                    },
                    {
                        "title": "Checkpoint Replication Lag",
                        "metrics": [
                            "custom.googleapis.com/checkpoint/replication_lag_seconds"
                        ],
                        "threshold": 300  # 5 minutes RPO
                    },
                    {
                        "title": "Regional Traffic Distribution",
                        "metrics": [
                            "loadbalancing.googleapis.com/https/request_count"
                        ],
                        "groupBy": ["resource.backend_target_name"]
                    }
                ]
            }
        }

        return dashboard_config
```

**Automated DR Testing for arr-coc-0-1:**

```bash
#!/bin/bash
# arr-coc-0-1 DR test script

echo "=== arr-coc-0-1 Disaster Recovery Test ==="

# 1. Verify all regions healthy
echo "Checking regional endpoint health..."
for region in us-central1 us-east1 europe-west1; do
    gcloud ai endpoints list --region=$region --filter="displayName:arr-coc"
done

# 2. Simulate primary region failure
echo "Simulating primary region (us-central1) failure..."
PRIMARY_ENDPOINT=$(gcloud ai endpoints list --region=us-central1 --filter="displayName:arr-coc-primary" --format="value(name)")
gcloud ai endpoints undeploy-model $PRIMARY_ENDPOINT --deployed-model-id=arr-coc-primary --region=us-central1

# 3. Measure failover time
START_TIME=$(date +%s)

# 4. Scale up secondary endpoint
echo "Scaling up secondary endpoint (us-east1)..."
SECONDARY_ENDPOINT=$(gcloud ai endpoints list --region=us-east1 --filter="displayName:arr-coc-secondary" --format="value(name)")
gcloud ai endpoints update $SECONDARY_ENDPOINT --min-replica-count=3 --region=us-east1

# 5. Wait for healthy
echo "Waiting for secondary endpoint ready..."
while true; do
    STATUS=$(gcloud ai endpoints describe $SECONDARY_ENDPOINT --region=us-east1 --format="value(deployedModels[0].state)")
    if [ "$STATUS" == "DEPLOYED" ]; then
        break
    fi
    sleep 5
done

END_TIME=$(date +%s)
RTO=$((END_TIME - START_TIME))

echo "=== Failover completed in $RTO seconds ==="

# 6. Test prediction
echo "Testing prediction on secondary endpoint..."
echo '{"instances": [{"text": "What is in this image?", "image": "gs://arr-coc-test-images/sample.jpg"}]}' | \
    gcloud ai endpoints predict $SECONDARY_ENDPOINT --region=us-east1

# 7. Restore primary
echo "Restoring primary endpoint..."
gcloud ai endpoints deploy-model $PRIMARY_ENDPOINT --model=arr-coc-model --region=us-central1

echo "=== DR test complete ==="
echo "RTO achieved: $RTO seconds (target: 300 seconds)"
```

## Sources

**Existing Knowledge:**
- [karpathy/practical-implementation/35-vertex-ai-production-patterns.md](../karpathy/practical-implementation/35-vertex-ai-production-patterns.md) - Multi-region training setup, checkpoint strategies, RPO/RTO targets

**Web Research (accessed 2025-11-16):**
- [Google Cloud Multi-Regional Deployment](https://docs.cloud.google.com/architecture/deployment-archetypes/multiregional) - Multi-region architecture patterns
- [Vertex AI Model Registry Copy](https://docs.cloud.google.com/vertex-ai/docs/model-registry/copy-model) - Cross-region model replication
- [GCP Disaster Recovery Architecture](https://docs.cloud.google.com/architecture/disaster-recovery) - DR planning and RPO/RTO guidance
- [Vertex AI Platform SLA](https://cloud.google.com/vertex-ai/sla) - Service level agreements and uptime guarantees
- [GCS Multi-Region Storage](https://cloud.google.com/storage/docs/locations#location-mr) - Multi-region bucket configuration
- [GCP Load Balancing](https://cloud.google.com/load-balancing/docs/https) - Global load balancer setup
- [AWS Disaster Recovery Objectives](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/disaster-recovery-dr-objectives.html) - RPO/RTO definitions and best practices

**Additional References:**
- Search results: "multi-region ML deployment GCP 2024"
- Search results: "Vertex AI cross-region model replication"
- Search results: "disaster recovery RPO RTO ML systems"
- Search results: "Global Load Balancer Vertex AI endpoints"
