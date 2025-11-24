# Multi-Region Quota Management: Failover, Pooling, and Dynamic Allocation

## Overview

Advanced multi-region quota management enables organizations to maximize resource availability across Google Cloud regions through automated failover, quota pooling strategies, and dynamic allocation. This guide covers production-ready patterns for managing quotas at scale across multiple regions, with emphasis on GPU workloads and automated orchestration.

**Key Capabilities:**
- Automated region failover when quota exhausted
- Quota pooling across regions for efficient utilization
- Dynamic quota allocation based on demand patterns
- Monitoring and alerting for quota exhaustion events
- Cost optimization through intelligent region selection

From [gcloud-production/01-quotas-alpha.md](../gcloud-production/01-quotas-alpha.md) multi-region section (lines 252-334):
- Regional quota spreading across multiple regions
- Failover queue system for quota availability
- Dynamic quota rebalancing based on usage

## Section 1: Multi-Region Architecture (~150 lines)

### Regional Quota Distribution

From [Reddit: Cloud Run L4 GPU Scaling](https://www.reddit.com/r/googlecloud/comments/1n24xwe/best_approach_to_scale_cloud_run_l4_gpu_jobs_past/) (accessed 2025-02-03):

**Problem**: Single-region quota limits (e.g., 5 L4 GPUs per region for Cloud Run)

**Solution**: Distribute workloads across multiple regions with identical deployments

```bash
# Define primary and failover regions
PRIMARY_REGIONS=("us-west1" "us-central1")
FAILOVER_REGIONS=("us-east1" "europe-west1" "asia-southeast1")

# Deploy to all regions
for region in "${PRIMARY_REGIONS[@]}" "${FAILOVER_REGIONS[@]}"; do
  gcloud run deploy training-service \
    --region=$region \
    --image=gcr.io/project/training:latest \
    --gpu=1 \
    --gpu-type=nvidia-l4
done
```

### Quota Pooling Strategy

From [Dynamic Shared Quota (DSQ)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/dynamic-shared-quota) (accessed 2025-02-03):

**Concept**: Vertex AI Gemini models use Dynamic Shared Quota - no predefined limits, shared pool dynamically allocated

**Key Insight**: Not all GCP services support dynamic quota pooling, but we can implement similar patterns manually:

```python
class QuotaPool:
    """Aggregate quota across multiple regions for elastic allocation."""

    def __init__(self, project_id: str, regions: list[str], gpu_type: str):
        self.project_id = project_id
        self.regions = regions
        self.gpu_type = gpu_type
        self._quota_cache = {}

    def get_total_available_quota(self) -> int:
        """Sum available quota across all regions."""
        total = 0
        for region in self.regions:
            quota_info = self._get_quota_usage(region, self.gpu_type)
            available = quota_info['limit'] - quota_info['usage']
            total += available
            self._quota_cache[region] = available
        return total

    def find_best_region(self, required_gpus: int) -> str:
        """Select region with most available quota."""
        # Refresh quota cache
        self.get_total_available_quota()

        # Sort by available quota descending
        sorted_regions = sorted(
            self._quota_cache.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for region, available in sorted_regions:
            if available >= required_gpus:
                return region

        return None  # No region has capacity
```

### Region Selection Criteria

**Priority order for intelligent region selection:**

1. **Quota Availability**: Regions with most available quota
2. **Cost**: GPU pricing varies by region (up to 30% difference)
3. **Latency**: Proximity to data sources or end users
4. **Capacity**: Historical availability and stockout patterns

From [CloudZero: GCP Availability Zones](https://www.cloudzero.com/blog/gcp-availability-zones/) (accessed 2025-02-03):

```python
REGION_PRIORITIES = {
    # Region: (cost_multiplier, latency_zone, capacity_score)
    "us-west1": (1.0, "west", 0.9),     # Best for US West
    "us-central1": (0.95, "central", 0.95),  # Lowest cost, high capacity
    "us-east1": (1.0, "east", 0.85),
    "europe-west1": (1.1, "eu", 0.8),
    "asia-southeast1": (1.15, "asia", 0.75),
}

def select_optimal_region(
    available_regions: list[str],
    priority: str = "cost"  # "cost", "latency", "capacity"
) -> str:
    """Select region based on priority criteria."""
    if priority == "cost":
        key = lambda r: REGION_PRIORITIES[r][0]
    elif priority == "latency":
        # Assume user location for latency calculation
        key = lambda r: abs(hash(REGION_PRIORITIES[r][1]) - hash(user_zone))
    else:  # capacity
        key = lambda r: -REGION_PRIORITIES[r][2]  # Higher is better

    return sorted(available_regions, key=key)[0]
```

## Section 2: Automated Failover (~150 lines)

### Failover Queue System

From [gcloud-production/01-quotas-alpha.md](../gcloud-production/01-quotas-alpha.md) Section 4 (lines 287-308):

**Pattern**: Check quota availability across regions, automatically failover when primary exhausted

```python
from google.cloud import compute_v1
import time

class RegionFailoverManager:
    """Manage automatic failover across regions for GPU workloads."""

    def __init__(
        self,
        project_id: str,
        primary_regions: list[str],
        failover_regions: list[str],
        gpu_type: str,
        min_gpus: int
    ):
        self.project_id = project_id
        self.primary_regions = primary_regions
        self.failover_regions = failover_regions
        self.gpu_type = gpu_type
        self.min_gpus = min_gpus
        self.quota_client = compute_v1.RegionOperationsClient()

    def get_quota_usage(self, region: str, metric: str) -> dict:
        """Get quota usage for specific metric in region."""
        service = compute_v1.ProjectsClient()
        project = service.get(project=self.project_id)

        for quota in project.quotas:
            if quota.metric == metric and region in quota.usage:
                return {
                    'limit': quota.limit,
                    'usage': quota.usage.get(region, 0),
                    'available': quota.limit - quota.usage.get(region, 0)
                }

        return {'limit': 0, 'usage': 0, 'available': 0}

    def find_available_region(self) -> str | None:
        """Find region with available quota (primary first, then failover)."""
        # Try primary regions first
        for region in self.primary_regions:
            quota = self.get_quota_usage(region, self.gpu_type)
            if quota['available'] >= self.min_gpus:
                print(f"✓ Primary region available: {region}")
                return region

        print("⚠ All primary regions exhausted, checking failover...")

        # Try failover regions
        for region in self.failover_regions:
            quota = self.get_quota_usage(region, self.gpu_type)
            if quota['available'] >= self.min_gpus:
                print(f"✓ Failover region available: {region}")
                return region

        return None  # No regions available

    def launch_with_failover(
        self,
        instance_config: dict,
        max_retries: int = 3
    ) -> dict:
        """Launch instance with automatic region failover."""
        attempt = 0

        while attempt < max_retries:
            region = self.find_available_region()

            if not region:
                print(f"✗ No regions available (attempt {attempt+1}/{max_retries})")
                time.sleep(60)  # Wait 1 minute before retry
                attempt += 1
                continue

            try:
                # Launch instance in selected region
                instance = self._launch_instance(region, instance_config)
                print(f"✓ Instance launched in {region}")
                return {
                    'success': True,
                    'region': region,
                    'instance': instance
                }

            except Exception as e:
                if "QUOTA_EXCEEDED" in str(e):
                    print(f"✗ Quota exhausted in {region}, trying next...")
                    attempt += 1
                else:
                    raise  # Non-quota error, propagate

        return {
            'success': False,
            'error': 'No regions available after retries'
        }
```

### Vertex AI Training Failover

From [Vertex AI Custom Jobs](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job) (accessed 2025-02-03):

```python
from google.cloud import aiplatform

def launch_training_with_regional_fallback(
    display_name: str,
    container_uri: str,
    regions: list[str],
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1
):
    """Launch Vertex AI training with regional fallback."""

    for region in regions:
        try:
            aiplatform.init(project=PROJECT_ID, location=region)

            job = aiplatform.CustomContainerTrainingJob(
                display_name=display_name,
                container_uri=container_uri
            )

            job.run(
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                replica_count=1
            )

            print(f"✓ Training launched in {region}")
            return {'region': region, 'job': job}

        except Exception as e:
            if "quota" in str(e).lower() or "insufficient" in str(e).lower():
                print(f"⚠ {region} quota exhausted, trying next region...")
                continue
            else:
                raise  # Non-quota error

    raise RuntimeError("No regions available for training")
```

## Section 3: Dynamic Quota Allocation (~150 lines)

### Demand-Based Allocation

From [gcloud-production/01-quotas-alpha.md](../gcloud-production/01-quotas-alpha.md) Section 4 (lines 310-333):

**Strategy**: Monitor usage patterns, reallocate quota requests to underutilized regions

```python
from google.cloud import monitoring_v3
from datetime import datetime, timedelta

class DynamicQuotaAllocator:
    """Dynamically allocate quota requests based on usage patterns."""

    def __init__(self, project_id: str, regions: list[str]):
        self.project_id = project_id
        self.regions = regions
        self.monitoring_client = monitoring_v3.MetricServiceClient()

    def get_usage_pattern(self, region: str, days: int = 7) -> dict:
        """Analyze quota usage pattern over time."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Query quota usage metrics
        interval = monitoring_v3.TimeInterval({
            "start_time": start_time,
            "end_time": end_time
        })

        # Build query for quota usage
        query = f"""
        fetch consumer_quota
        | filter resource.location == '{region}'
        | metric 'serviceruntime.googleapis.com/quota/allocation/usage'
        | align rate(1h)
        | every 1h
        """

        results = self.monitoring_client.list_time_series(
            request={
                "name": f"projects/{self.project_id}",
                "filter": f'resource.location="{region}"',
                "interval": interval
            }
        )

        # Calculate pattern statistics
        usage_values = [point.value.double_value for ts in results for point in ts.points]

        return {
            'region': region,
            'avg_usage': sum(usage_values) / len(usage_values) if usage_values else 0,
            'peak_usage': max(usage_values) if usage_values else 0,
            'min_usage': min(usage_values) if usage_values else 0,
            'utilization': (sum(usage_values) / len(usage_values)) / 100 if usage_values else 0
        }

    def identify_reallocation_targets(self) -> dict:
        """Identify regions for quota reallocation."""
        patterns = {region: self.get_usage_pattern(region) for region in self.regions}

        # Sort by utilization
        sorted_regions = sorted(
            patterns.items(),
            key=lambda x: x[1]['utilization']
        )

        underutilized = [r for r, p in sorted_regions if p['utilization'] < 0.5]
        overutilized = [r for r, p in sorted_regions if p['utilization'] > 0.8]

        return {
            'underutilized': underutilized,
            'overutilized': overutilized,
            'recommendation': f"Consider requesting quota increases in: {overutilized}"
        }

    def auto_request_quota_increase(
        self,
        region: str,
        metric: str,
        increase_amount: int,
        justification: str
    ) -> str:
        """Automatically request quota increase using Cloud Quotas API."""
        from google.cloud import cloudquotas_v1

        client = cloudquotas_v1.CloudQuotasClient()

        preference = cloudquotas_v1.QuotaPreference(
            dimensions={"region": region},
            quota_config=cloudquotas_v1.QuotaConfig(
                preferred_value=increase_amount
            ),
            justification=justification,
            contact_email="infrastructure@company.com"
        )

        parent = f"projects/{self.project_id}/locations/{region}/services/compute.googleapis.com"

        request = cloudquotas_v1.CreateQuotaPreferenceRequest(
            parent=parent,
            quota_preference=preference,
            quota_preference_id=f"{metric}-auto-increase"
        )

        response = client.create_quota_preference(request=request)
        return f"Quota increase requested: {response.name}"
```

### Quota Adjuster Integration

From [Quota Adjuster](https://docs.cloud.google.com/docs/quotas/quota-adjuster) (accessed 2025-02-03):

**Pre-GA Feature**: Automatically increases quotas when usage approaches limits

```bash
# Enable quota adjuster (Pre-GA)
# Navigate to IAM & Admin > Quotas > Configurations tab > Enable

# Supported quotas (as of preview):
# - Compute Engine CPUs
# - N2, N2D, C2, C2D CPUs
# - Persistent Disk Standard (GB)

# Monitor adjuster activity
gcloud logging read \
  'protoPayload.methodName="QuotaAdjuster.AdjustQuota"' \
  --limit=50 \
  --format=json
```

**Limitations**:
- Only specific Compute Engine quotas supported in preview
- GPU quotas NOT yet supported
- Increases typically 10-20% of current limit
- Subject to Google Cloud approval

## Section 4: Monitoring and Alerting (~100 lines)

### Quota Usage Alerts

From [Medium: Quota Monitoring Options](https://medium.com/google-cloud/quota-monitoring-and-management-options-on-google-cloud-b94caf8a9671) (accessed 2025-02-03):

**Terraform Configuration for Multi-Region Alerts**:

```hcl
# Alert for quota usage across all regions
resource "google_monitoring_alert_policy" "quota_multi_region" {
  display_name = "Multi-Region Quota Alert (>80%)"
  enabled      = true
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "quota-usage-high"

    condition_monitoring_query_language {
      query = <<-EOQ
        fetch consumer_quota
        | filter resource.service =~ '.*'
        | filter resource.location =~ '.*'
        | {
            t_0:
              metric 'serviceruntime.googleapis.com/quota/allocation/usage'
              | align next_older(1d)
              | group_by [resource.project_id, metric.quota_metric, resource.location],
                [value_usage_max: max(value.usage)]
            ;
            t_1:
              metric 'serviceruntime.googleapis.com/quota/limit'
              | align next_older(1d)
              | group_by [resource.project_id, metric.quota_metric, resource.location],
                [value_limit_min: min(value.limit)]
        }
        | ratio
        | every 1m
        | condition gt(ratio, 0.8 '1')
      EOQ

      duration = "60s"
      trigger { count = 1 }
    }
  }

  alert_strategy {
    auto_close = "86400s"
  }

  notification_channels = [
    google_monitoring_notification_channel.quota_alerts.name
  ]
}

# Notification channel
resource "google_monitoring_notification_channel" "quota_alerts" {
  display_name = "Quota Alerts - Multi-Region"
  type         = "email"
  project      = var.project_id

  labels = {
    email_address = "infrastructure-alerts@company.com"
  }
}
```

### Regional Quota Dashboard

From [Google Quota Monitoring Solution](https://github.com/google/quota-monitoring-solution) (accessed 2025-02-03):

**Features**:
- Looker Studio dashboard with alerting
- Organization/folder-level aggregation
- Multi-region quota visibility
- Historical usage trends

```bash
# Deploy Quota Monitoring Solution
git clone https://github.com/google/quota-monitoring-solution.git
cd quota-monitoring-solution

# Configure for multi-region monitoring
cat > config.yaml <<EOF
project_id: ${PROJECT_ID}
regions:
  - us-west1
  - us-central1
  - us-east1
  - europe-west1
metrics:
  - compute.googleapis.com/nvidia_t4_gpus
  - compute.googleapis.com/nvidia_l4_gpus
alert_threshold: 0.8
EOF

# Deploy via Terraform
terraform init
terraform apply
```

### Log-Based Metrics for Quota Changes

```bash
# Create log-based metric for quota limit changes
gcloud logging metrics create quota_limit_change \
  --description="Alert when quota limits change in any region" \
  --log-filter='
    protoPayload.methodName="QuotaService.UpdateQuotaLimit"
    OR protoPayload.methodName="QuotaService.CreateQuotaPreference"
  '

# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Quota Limit Changes - All Regions" \
  --condition-display-name="Quota modified" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s
```

## Section 5: Production Automation Scripts (~50 lines)

### Complete Multi-Region Orchestration

```python
#!/usr/bin/env python3
"""
Multi-region quota management orchestrator.
Handles failover, pooling, and dynamic allocation.
"""

import argparse
from typing import List, Dict
from region_failover import RegionFailoverManager
from quota_pool import QuotaPool
from dynamic_allocator import DynamicQuotaAllocator

def main():
    parser = argparse.ArgumentParser(description="Multi-region quota orchestrator")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--gpu-type", default="nvidia_t4_gpus", help="GPU type")
    parser.add_argument("--min-gpus", type=int, default=1, help="Min GPUs required")
    parser.add_argument("--action", choices=["failover", "pool", "allocate"], required=True)

    args = parser.parse_args()

    PRIMARY_REGIONS = ["us-west1", "us-central1"]
    FAILOVER_REGIONS = ["us-east1", "europe-west1", "asia-southeast1"]

    if args.action == "failover":
        manager = RegionFailoverManager(
            project_id=args.project,
            primary_regions=PRIMARY_REGIONS,
            failover_regions=FAILOVER_REGIONS,
            gpu_type=args.gpu_type,
            min_gpus=args.min_gpus
        )

        result = manager.launch_with_failover(
            instance_config={'machine_type': 'n1-standard-4'}
        )
        print(f"Result: {result}")

    elif args.action == "pool":
        pool = QuotaPool(
            project_id=args.project,
            regions=PRIMARY_REGIONS + FAILOVER_REGIONS,
            gpu_type=args.gpu_type
        )

        total = pool.get_total_available_quota()
        best_region = pool.find_best_region(args.min_gpus)

        print(f"Total available quota: {total}")
        print(f"Best region: {best_region}")

    elif args.action == "allocate":
        allocator = DynamicQuotaAllocator(
            project_id=args.project,
            regions=PRIMARY_REGIONS + FAILOVER_REGIONS
        )

        recommendations = allocator.identify_reallocation_targets()
        print(f"Recommendations: {recommendations}")

if __name__ == "__main__":
    main()
```

**Usage**:

```bash
# Check quota pool availability
python orchestrator.py \
  --project=my-project \
  --gpu-type=nvidia_l4_gpus \
  --min-gpus=4 \
  --action=pool

# Launch with failover
python orchestrator.py \
  --project=my-project \
  --gpu-type=nvidia_t4_gpus \
  --min-gpus=2 \
  --action=failover

# Get dynamic allocation recommendations
python orchestrator.py \
  --project=my-project \
  --action=allocate
```

## Sources

**Source Documents:**
- [gcloud-production/01-quotas-alpha.md](../gcloud-production/01-quotas-alpha.md) - Multi-region strategies section (lines 252-334)

**Google Cloud Official Documentation:**
- [Dynamic Shared Quota (DSQ)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/dynamic-shared-quota) - Vertex AI dynamic quota pooling (accessed 2025-02-03)
- [Quota Adjuster](https://docs.cloud.google.com/docs/quotas/quota-adjuster) - Automated quota increases (accessed 2025-02-03)
- [Set Up Quota Alerts](https://docs.cloud.google.com/docs/quotas/set-up-quota-alerts) - Monitoring setup (accessed 2025-02-03)
- [Cloud Quotas API Overview](https://cloud.google.com/docs/quotas/api-overview) - Programmatic quota management (accessed 2025-02-03)

**Community Resources:**
- [Medium: Quota Monitoring and Management Options](https://medium.com/google-cloud/quota-monitoring-and-management-options-on-google-cloud-b94caf8a9671) - Vipul Raja, comprehensive quota monitoring guide (accessed 2025-02-03)
- [CloudZero: GCP Availability Zones](https://www.cloudzero.com/blog/gcp-availability-zones/) - Multi-region best practices (accessed 2025-02-03)
- [Google Quota Monitoring Solution](https://github.com/google/quota-monitoring-solution) - Official open source quota dashboard (accessed 2025-02-03)

**Reddit Discussions:**
- [Reddit: Cloud Run L4 GPU Scaling](https://www.reddit.com/r/googlecloud/comments/1n24xwe/best_approach_to_scale_cloud_run_l4_gpu_jobs_past/) - Multi-region GPU strategies (accessed 2025-02-03)

**Additional References:**
- [Compute Engine Allocation Quotas](https://docs.cloud.google.com/compute/resource-usage) - GPU quota structure (accessed 2025-02-03)
- [Vertex AI Custom Jobs](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job) - Training job deployment (accessed 2025-02-03)
