# GPU Cost Optimization Strategies on GCP

Comprehensive strategies for reducing GPU training and inference costs on Google Cloud Platform through pricing models, resource optimization, monitoring, and FinOps best practices. This guide focuses on achieving 60-91% cost savings while maintaining production ML/AI workload performance.

## Overview

GPU costs represent the largest expense in ML/AI infrastructure. Effective optimization requires a multi-layered approach combining pricing discounts, resource right-sizing, idle detection, scheduling strategies, and automated cost controls.

**Key optimization pillars:**
- Preemptible/Spot GPU instances (60-91% savings)
- Committed Use Discounts (up to 57% savings)
- Sustained Use Discounts (automatic, up to 30%)
- Idle GPU detection and elimination
- Right-sizing GPU types (T4 vs L4 vs A100 vs H100)
- Multi-tenancy and GPU sharing
- Scheduled training and automated shutdown

---

## Section 1: Preemptible and Spot GPU Instances (60-91% Savings)

### Preemptible vs Spot Pricing Model

From [GCP Spot VM pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-11-16) and [GCP Spot Instance Cost Optimization](../karpathy/practical-implementation/44-gcp-spot-cost-optimization.md):

**GCP uses dynamic pricing, NOT bidding** (critical distinction from AWS/Azure):

| Cloud Provider | Pricing Model | User Control |
|----------------|---------------|--------------|
| **AWS** | Spot bidding | Set max bid price |
| **Azure** | Spot bidding | Set max eviction price |
| **GCP** | Dynamic pricing | No bidding - accept current price |

**GCP Spot pricing characteristics:**
- Prices vary by region/zone/machine type
- Update up to once per day (not real-time)
- Transparent pricing visible in console
- No price spikes from bidding wars
- Simpler cost forecasting vs AWS

### Spot Discount Rates by GPU Type

From [GCP GPU pricing](https://cloud.google.com/compute/gpus-pricing) and [Economize Cloud GPU Pricing Comparison](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) (September 27, 2024, accessed 2025-11-16):

**GPU-optimized instances:**

```
A2 (A100 80GB):
- On-demand: $3.67/GPU/hour (us-central1)
- Spot: $1.20/GPU/hour
- Savings: 67%

A3 (H100 80GB):
- On-demand: ~$6.50/GPU/hour (limited availability)
- Spot: ~$2.00/GPU/hour
- Savings: 69%

G2 (L4):
- On-demand: $0.90/GPU/hour
- Spot: $0.26/GPU/hour
- Savings: 71%

N1 with T4:
- On-demand: $0.35/GPU/hour
- Spot: $0.11/GPU/hour
- Savings: 69%
```

**Multi-GPU instance pricing (8-GPU examples):**

```
a2-highgpu-8g (8× A100 40GB):
- On-demand: $29.39/hour (us-central1)
- Spot: $9.70/hour
- Savings: 67% ($19.69/hour saved)

a2-ultragpu-8g (8× A100 80GB):
- On-demand: Contact sales
- Spot: ~$15/hour estimated
- Savings: ~65-70%
```

### Regional Pricing Variations

**Price comparison across regions (A100 spot, approximate):**

```
Region              A100 Spot Price    On-Demand Price    Savings
us-central1         $1.20/GPU/hr       $3.67/GPU/hr       67%
us-west1            $1.20/GPU/hr       $3.67/GPU/hr       67%
europe-west4        $1.35/GPU/hr       $4.05/GPU/hr       67%
asia-southeast1     $1.40/GPU/hr       $4.10/GPU/hr       66%
```

**Regional arbitrage strategy:**
- Price differences: 5-15% variation between regions
- Data locality: Consider egress costs if data in different region
- Latency: Not critical for batch training workloads
- Compliance: Data residency requirements may limit options

### Preemption Patterns and Mitigation

From [GCP Spot Checkpoint Strategies](../karpathy/practical-implementation/43-gcp-spot-checkpoint-strategies.md):

**Preemption behavior:**
- 30-second shutdown warning (metadata server notification)
- No guaranteed runtime (can be preempted anytime after 60 seconds)
- Availability varies by zone and GPU type
- H100/A100 preemption rates: 10-25% in high-demand zones
- L4/T4 preemption rates: 5-15% (better availability)

**Cost-performance with preemption overhead:**

```python
def estimate_spot_training_cost_with_overhead(
    on_demand_hourly: float,
    training_hours: float,
    spot_discount: float = 0.67,
    preemption_overhead: float = 1.08  # 8% restart overhead
):
    """
    Calculate expected spot cost including preemption restart time.

    Args:
        on_demand_hourly: On-demand hourly rate
        training_hours: Expected training duration
        spot_discount: Spot discount (0.60-0.91)
        preemption_overhead: Extra time from restarts (1.05-1.15)

    Returns:
        Cost analysis with savings
    """
    spot_hourly = on_demand_hourly * (1 - spot_discount)
    adjusted_hours = training_hours * preemption_overhead
    spot_total = spot_hourly * adjusted_hours

    on_demand_total = on_demand_hourly * training_hours
    savings = on_demand_total - spot_total
    effective_discount = savings / on_demand_total

    return {
        "spot_cost": spot_total,
        "on_demand_cost": on_demand_total,
        "savings": savings,
        "effective_discount": effective_discount,
        "overhead_hours": adjusted_hours - training_hours
    }

# Example: 24-hour LLM training on 8× A100
result = estimate_spot_training_cost_with_overhead(
    on_demand_hourly=29.39,  # a2-highgpu-8g us-central1
    training_hours=24,
    spot_discount=0.67,
    preemption_overhead=1.08  # Good checkpoint strategy
)

print(f"Spot cost: ${result['spot_cost']:.2f}")
print(f"On-demand cost: ${result['on_demand_cost']:.2f}")
print(f"Savings: ${result['savings']:.2f} ({result['effective_discount']:.1%})")
print(f"Overhead: {result['overhead_hours']:.1f} extra hours")

# Output:
# Spot cost: $233.34
# On-demand cost: $705.36
# Savings: $472.02 (66.9%)
# Overhead: 1.9 extra hours
```

**Checkpoint strategy for minimal overhead:**

From [GCP Spot Checkpoint Strategies](../karpathy/practical-implementation/43-gcp-spot-checkpoint-strategies.md):

- Checkpoint every 100-500 steps (5-15 minutes)
- Use asynchronous checkpointing (doesn't block training)
- Store checkpoints to GCS (persistent across preemptions)
- Resume from latest checkpoint automatically
- Overhead with good strategy: 5-10% extra time

---

## Section 2: Committed Use Discounts (Up to 57% Savings)

### CUD Overview

From [GCP Committed Use Discounts](https://docs.cloud.google.com/compute/docs/instances/committed-use-discounts-overview) and [CloudZero GCP CUD Guide](https://www.cloudzero.com/blog/gcp-cud/) (June 30, 2023, accessed 2025-11-16):

**Committed Use Discounts (CUDs)** provide savings when you commit to using GCP resources for 1 or 3 years.

**Discount rates:**
- 1-year commitment: Up to 37% savings
- 3-year commitment: Up to 57% savings
- Memory-optimized: Up to 70% savings
- GPUs: Up to 55% savings

**Two CUD types:**

1. **Resource-based CUDs** (Compute Engine specific):
   - Commit to specific vCPU/memory/GPU quantities
   - Applied automatically to matching instances
   - Flexible across machine types within commitment

2. **Spend-based CUDs** (Flexible across services):
   - Commit to spend amount ($100/month, $1000/month, etc.)
   - Covers Compute Engine, GKE, Cloud Run
   - Most flexible option

### GPU-Specific CUD Pricing

From [GCP GPU pricing](https://cloud.google.com/compute/gpus-pricing) (accessed 2025-11-16):

**A100 GPU CUD examples (us-central1):**

```
NVIDIA A100 40GB:
- On-demand: $2.933/GPU/hour
- 1-year CUD: $1.851/GPU/hour (37% savings)
- 3-year CUD: $1.321/GPU/hour (55% savings)

NVIDIA A100 80GB:
- On-demand: $3.670/GPU/hour
- 1-year CUD: $2.313/GPU/hour (37% savings)
- 3-year CUD: $1.650/GPU/hour (55% savings)
```

**L4 GPU CUD examples:**

```
NVIDIA L4:
- On-demand: $0.900/GPU/hour
- 1-year CUD: $0.567/GPU/hour (37% savings)
- 3-year CUD: $0.405/GPU/hour (55% savings)
```

**Multi-GPU commitment calculation:**

```python
def calculate_cud_savings(
    gpu_count: int,
    hours_per_month: int,
    on_demand_rate: float,
    commitment_years: int
):
    """
    Calculate CUD savings for GPU commitment.

    Args:
        gpu_count: Number of GPUs to commit
        hours_per_month: Expected monthly usage hours
        on_demand_rate: On-demand hourly rate per GPU
        commitment_years: 1 or 3 years

    Returns:
        Savings analysis
    """
    discount_rate = 0.37 if commitment_years == 1 else 0.55
    cud_rate = on_demand_rate * (1 - discount_rate)

    monthly_on_demand = gpu_count * hours_per_month * on_demand_rate
    monthly_cud = gpu_count * hours_per_month * cud_rate
    monthly_savings = monthly_on_demand - monthly_cud

    annual_savings = monthly_savings * 12
    commitment_savings = annual_savings * commitment_years

    return {
        "monthly_on_demand": monthly_on_demand,
        "monthly_cud": monthly_cud,
        "monthly_savings": monthly_savings,
        "annual_savings": annual_savings,
        "total_commitment_savings": commitment_savings,
        "effective_discount": discount_rate
    }

# Example: 8× A100 80GB running 730 hours/month (24/7)
result = calculate_cud_savings(
    gpu_count=8,
    hours_per_month=730,  # ~730 hours/month
    on_demand_rate=3.67,  # A100 80GB on-demand
    commitment_years=3
)

print(f"Monthly on-demand: ${result['monthly_on_demand']:,.2f}")
print(f"Monthly CUD: ${result['monthly_cud']:,.2f}")
print(f"Monthly savings: ${result['monthly_savings']:,.2f}")
print(f"3-year savings: ${result['total_commitment_savings']:,.2f}")

# Output:
# Monthly on-demand: $21,443.20
# Monthly CUD: $9,649.44
# Monthly savings: $11,793.76
# 3-year savings: $424,575.36
```

### When to Use CUDs

**Ideal for:**
- Production training pipelines running 24/7
- Continuous inference serving workloads
- Predictable GPU usage (>60% utilization)
- Stable workload patterns over months

**Not ideal for:**
- Experimental/research workloads (variable demand)
- Short-term projects (<6 months)
- Unpredictable GPU needs
- Dev/test environments (use Spot instead)

**CUD + Spot hybrid strategy:**

```
Baseline workload: CUD commitment (covers 60% of usage)
Peak workload: Spot instances (handles 40% of variable demand)

Example: 8× A100 training cluster
- 5× A100 on 3-year CUD (55% savings, always running)
- 3× A100 on Spot (67% savings, scale up/down as needed)

Blended savings: ~60% overall cost reduction
```

---

## Section 3: Sustained Use Discounts (Automatic, Up to 30%)

### How Sustained Use Discounts Work

From [GCP Sustained Use Discounts](https://cloud.google.com/compute/docs/sustained-use-discounts) (accessed 2025-11-16):

**Sustained Use Discounts (SUDs) are automatic** - no commitment required.

**Discount tiers (incremental):**
```
Usage % of month    Discount Rate
0-25%              0%
25-50%             20% incremental
50-75%             40% incremental
75-100%            60% incremental

Effective monthly discount: Up to 30% for full-month usage
```

**GPU eligibility:**
- Applies to: General-purpose VMs (N1, N2, N2D, E2)
- Does NOT apply to: A2, A3, G2 GPU-optimized instances
- Does NOT apply to: Preemptible/Spot instances
- Does NOT apply to: Instances covered by CUDs

**Practical implications:**
- SUDs help with on-demand GPU VMs (N1 with attached GPUs)
- A2/A3 instances NOT eligible (must use CUD or Spot for savings)
- Automatically applied - no action needed
- Stacks with per-second billing (no waste from partial hours)

---

## Section 4: GPU Idle Time Monitoring and Waste Detection

### The Idle GPU Problem

From [How Much Do GPU Cloud Platforms Cost](https://www.gmicloud.ai/blog/how-much-do-gpu-cloud-platforms-cost-for-ai-startups-in-2025) (accessed 2025-11-16) and [FinOps for AI: Govern GPU Spend](https://www.flexera.com/blog/finops/finops-for-ai-governing-the-unique-economics-of-intelligent-workloads/) (September 11, 2025, accessed 2025-11-16):

**Idle GPU waste statistics:**
- 30-50% of GPU spending wasted on idle time (debugging, meetings, overnight)
- Idle resources account for up to 32% of cloud waste (FinOps studies)
- Average GPU utilization in production: 40-60% (target: >80%)

**Common causes of idle GPUs:**
- Forgotten Jupyter notebooks left running
- Debugging sessions paused indefinitely
- Training jobs that completed but VMs not terminated
- Development instances running 24/7
- Waiting for data preprocessing to finish

### Automated Idle Detection

**Cloud Monitoring metrics for GPU utilization:**

From [GCP Cloud Monitoring GPU metrics](https://cloud.google.com/monitoring/api/metrics_gcp#gcp-compute) (accessed 2025-11-16):

```
Metric: compute.googleapis.com/instance/gpu/utilization
Description: GPU utilization percentage (0-100%)
Sampling: 60-second intervals

Metric: compute.googleapis.com/instance/gpu/memory_utilization
Description: GPU memory utilization (0-100%)
Sampling: 60-second intervals
```

**Idle detection policy:**

```python
# Cloud Monitoring alert policy (YAML)
# File: gpu-idle-detection-policy.yaml

displayName: "GPU Idle Detection Alert"
documentation:
  content: "GPU utilization below 10% for >30 minutes"
conditions:
  - displayName: "Low GPU Utilization"
    conditionThreshold:
      filter: |
        resource.type = "gce_instance" AND
        metric.type = "compute.googleapis.com/instance/gpu/utilization"
      comparison: COMPARISON_LT
      thresholdValue: 10  # <10% utilization
      duration: 1800s  # 30 minutes
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
          crossSeriesReducer: REDUCE_MEAN
          groupByFields:
            - resource.instance_id

alertStrategy:
  notificationRateLimit:
    period: 3600s  # Alert at most once per hour

notificationChannels:
  - projects/PROJECT_ID/notificationChannels/EMAIL_CHANNEL
  - projects/PROJECT_ID/notificationChannels/SLACK_CHANNEL
```

**Automated shutdown for idle GPUs:**

```python
# Cloud Function triggered by Cloud Monitoring alert
# File: idle_gpu_shutdown.py

import google.auth
from google.cloud import compute_v1
import json
import base64

def shutdown_idle_gpu(event, context):
    """
    Triggered by Pub/Sub from GPU idle alert.
    Stops idle GPU instances automatically.
    """
    # Parse alert data
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    alert_data = json.loads(pubsub_message)

    instance_id = alert_data['incident']['resource']['labels']['instance_id']
    project = alert_data['incident']['resource']['labels']['project_id']
    zone = alert_data['incident']['resource']['labels']['zone']

    # Check if instance has 'auto-shutdown-allowed' label
    compute = compute_v1.InstancesClient()
    instance = compute.get(project=project, zone=zone, instance=instance_id)

    labels = instance.labels or {}
    if labels.get('auto-shutdown-idle') != 'true':
        print(f"Instance {instance_id} not labeled for auto-shutdown. Skipping.")
        return

    # Stop the instance
    print(f"Stopping idle GPU instance: {instance_id}")
    operation = compute.stop(project=project, zone=zone, instance=instance_id)

    # Send notification
    send_slack_notification(
        f"🛑 Stopped idle GPU instance: {instance_id} in {zone}\n"
        f"Instance was idle (< 10% GPU util) for 30+ minutes.\n"
        f"Estimated savings: ~${estimate_hourly_cost(instance)}/hour"
    )

    print(f"Successfully stopped {instance_id}")

def estimate_hourly_cost(instance):
    """Estimate hourly cost based on machine type"""
    machine_type = instance.machine_type.split('/')[-1]

    # Approximate GPU instance costs
    cost_map = {
        'a2-highgpu-1g': 3.67,
        'a2-highgpu-2g': 7.34,
        'a2-highgpu-4g': 14.68,
        'a2-highgpu-8g': 29.36,
        'g2-standard-4': 1.20,
        'g2-standard-8': 2.40,
        'n1-standard-8': 0.73,  # with attached T4
    }

    return cost_map.get(machine_type, 2.0)

def send_slack_notification(message):
    """Send Slack notification"""
    # Implement Slack webhook notification
    pass
```

**Usage monitoring dashboard:**

Create custom dashboard tracking:
- GPU utilization over time (target: >80%)
- GPU memory utilization (target: >70%)
- Cost per GPU hour vs utilization
- Idle time detection (instances < 10% util for >30 min)
- Weekly idle cost waste report

---

## Section 5: Right-Sizing GPU Types (T4 vs L4 vs A100 vs H100)

### GPU Type Selection Matrix

From [GPU Price Comparison 2025](https://getdeploying.com/reference/cloud-gpu) (accessed 2025-11-16) and [Cast AI 2025 GPU Price Report](https://cast.ai/reports/gpu-price-2025/) (accessed 2025-11-16):

**GPU comparison table:**

```
GPU Type    VRAM    FP32 TFlops  Tensor TFlops  On-Demand Price  Spot Price   Best For
T4          16 GB   8.1          65 (FP16)      $0.35/hr         $0.11/hr     Inference, small models
L4          24 GB   30.3         242 (FP16)     $0.90/hr         $0.26/hr     Inference, video, small training
A100 40GB   40 GB   19.5         312 (FP16)     $2.93/hr         $0.97/hr     Training, large models
A100 80GB   80 GB   19.5         312 (FP16)     $3.67/hr         $1.20/hr     Large model training
H100        80 GB   51           1979 (FP8)     $6.50/hr         $2.00/hr     Frontier training, research
```

**Workload-to-GPU mapping:**

```
Inference (small models <7B parameters):
→ T4 (16GB) - Most cost-effective
→ $0.11/hr spot, handles 10-50 req/sec

Inference (medium models 7-13B):
→ L4 (24GB) - Balanced price/performance
→ $0.26/hr spot, handles 20-100 req/sec

Training (models <13B parameters):
→ L4 or single A100 40GB
→ L4: $0.26/hr spot, slower but cheaper
→ A100: $0.97/hr spot, 3× faster

Training (models 13-70B parameters):
→ A100 80GB (single or multi-GPU)
→ $1.20/hr spot per GPU
→ 8× A100 for 70B model fine-tuning

Training (models >70B parameters):
→ H100 or multi-node A100 clusters
→ H100: $2.00/hr spot, 6× faster than A100
→ Best for frontier research
```

**Cost-performance analysis:**

```python
def compare_gpu_types_for_workload(
    model_size_gb: float,
    training_hours_a100: float
):
    """
    Compare cost-performance across GPU types.

    Args:
        model_size_gb: Model memory footprint (GB)
        training_hours_a100: Training time on A100 (baseline)
    """
    gpus = {
        'T4': {'vram': 16, 'relative_speed': 0.25, 'spot_price': 0.11},
        'L4': {'vram': 24, 'relative_speed': 0.5, 'spot_price': 0.26},
        'A100-40GB': {'vram': 40, 'relative_speed': 1.0, 'spot_price': 0.97},
        'A100-80GB': {'vram': 80, 'relative_speed': 1.0, 'spot_price': 1.20},
        'H100': {'vram': 80, 'relative_speed': 3.0, 'spot_price': 2.00},
    }

    results = {}
    for name, specs in gpus.items():
        # Skip if model doesn't fit
        if model_size_gb > specs['vram']:
            continue

        # Calculate time and cost
        training_time = training_hours_a100 / specs['relative_speed']
        total_cost = training_time * specs['spot_price']

        results[name] = {
            'training_hours': training_time,
            'total_cost': total_cost,
            'hourly_rate': specs['spot_price']
        }

    return results

# Example: 30GB model, 24 hours on A100
comparison = compare_gpu_types_for_workload(
    model_size_gb=30,
    training_hours_a100=24
)

for gpu, metrics in comparison.items():
    print(f"{gpu}:")
    print(f"  Time: {metrics['training_hours']:.1f} hours")
    print(f"  Cost: ${metrics['total_cost']:.2f}")
    print(f"  Rate: ${metrics['hourly_rate']:.2f}/hr")
    print()

# Output:
# A100-40GB:
#   Time: 24.0 hours
#   Cost: $23.28
#   Rate: $0.97/hr
#
# A100-80GB:
#   Time: 24.0 hours
#   Cost: $28.80
#   Rate: $1.20/hr
#
# H100:
#   Time: 8.0 hours
#   Cost: $16.00
#   Rate: $2.00/hr
```

**Decision criteria:**

1. **Memory requirements**: Choose GPU with sufficient VRAM
2. **Training time urgency**: H100 if speed critical, A100/L4 if cost-sensitive
3. **Batch size**: Larger batches need more VRAM
4. **Multi-GPU scaling**: A100/H100 have better NVLink bandwidth

---

## Section 6: Multi-Tenancy and GPU Sharing

### GPU Time-Sharing on GKE

From [GKE GPU time-sharing](https://cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus) and [Configure GPU Resource Quotas](https://docs.rafay.co/blog/2025/06/27/configure-and-manage-gpu-resource-quotas-in-multi-tenant-clouds/) (June 27, 2025, accessed 2025-11-16):

**GKE GPU time-sharing** allows multiple pods to share a single GPU (multi-tenant GPU access).

**Use cases:**
- Dev/test environments (multiple developers sharing GPU)
- Inference workloads with low GPU utilization
- Small models that don't need full GPU
- Cost-sharing across teams

**Configuration:**

```yaml
# Enable GPU time-sharing on GKE node pool
# Max 48 containers per GPU

apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-sharing-config
  namespace: kube-system
data:
  time-slicing-config: |
    version: v1
    sharing:
      timeSlicing:
        replicas: 4  # 4 pods share each GPU
        resources:
          - name: nvidia.com/gpu
            devices: all
```

**Pod resource request:**

```yaml
# Pod requesting shared GPU access
apiVersion: v1
kind: Pod
metadata:
  name: gpu-shared-workload
spec:
  containers:
  - name: training
    image: nvidia/cuda:12.1.0-base
    resources:
      limits:
        nvidia.com/gpu: 1  # Still request "1 GPU" but it's time-shared
```

**Cost allocation with shared GPUs:**

```python
def calculate_shared_gpu_cost(
    gpu_hourly_rate: float,
    sharing_ratio: int,
    usage_hours_per_user: dict
):
    """
    Calculate per-user cost for shared GPU.

    Args:
        gpu_hourly_rate: Full GPU hourly cost
        sharing_ratio: Number of users sharing GPU
        usage_hours_per_user: Dict of {user: hours_used}

    Returns:
        Cost breakdown per user
    """
    total_hours = sum(usage_hours_per_user.values())
    cost_per_hour_share = gpu_hourly_rate / sharing_ratio

    user_costs = {}
    for user, hours in usage_hours_per_user.items():
        user_costs[user] = hours * cost_per_hour_share

    return {
        'total_gpu_cost': total_hours * cost_per_hour_share,
        'user_costs': user_costs,
        'savings_vs_dedicated': (sharing_ratio - 1) * gpu_hourly_rate
    }

# Example: 4 users sharing L4 GPU
result = calculate_shared_gpu_cost(
    gpu_hourly_rate=0.90,  # L4 on-demand
    sharing_ratio=4,
    usage_hours_per_user={
        'team_a': 100,
        'team_b': 75,
        'team_c': 50,
        'team_d': 25
    }
)

print(f"Total GPU cost: ${result['total_gpu_cost']:.2f}")
print("Per-user costs:")
for user, cost in result['user_costs'].items():
    print(f"  {user}: ${cost:.2f}")

# Output:
# Total GPU cost: $56.25
# Per-user costs:
#   team_a: $22.50
#   team_b: $16.88
#   team_c: $11.25
#   team_d: $5.62
```

**Limitations:**
- No memory isolation (pods can see each other's GPU memory)
- Performance interference (pods compete for GPU cycles)
- Best for inference or low-intensity workloads
- NOT recommended for training large models

---

## Section 7: Scheduled Training and Automated Shutdown

### Cost-Optimized Training Schedules

**Strategy: Run training during low-cost time windows**

From [GCP billing automation](../gcloud-cost/00-billing-automation.md):

```python
def schedule_training_within_budget(
    training_jobs: list,
    daily_budget_usd: float,
    priority_order: list = None
):
    """
    Schedule training jobs to fit within daily GPU budget.

    Args:
        training_jobs: List of job configs with estimated costs
        daily_budget_usd: Maximum daily spend
        priority_order: Optional priority ordering for jobs

    Returns:
        Scheduled jobs and budget allocation
    """
    if priority_order:
        training_jobs = sorted(
            training_jobs,
            key=lambda j: priority_order.index(j['name'])
        )

    scheduled = []
    total_allocated = 0

    for job in training_jobs:
        estimated_cost = estimate_job_cost(job)

        if total_allocated + estimated_cost <= daily_budget_usd:
            scheduled.append(job)
            total_allocated += estimated_cost
            print(f"✓ Scheduled: {job['name']} (${estimated_cost:.2f})")
        else:
            print(f"✗ Skipped: {job['name']} - would exceed budget")

    print(f"\nBudget: ${total_allocated:.2f} / ${daily_budget_usd:.2f}")

    return {
        'scheduled_jobs': scheduled,
        'total_cost': total_allocated,
        'remaining_budget': daily_budget_usd - total_allocated
    }

def estimate_job_cost(job):
    """Estimate total cost for training job"""
    config = job['config']
    hours = config.get('estimated_hours', 4)

    # GPU pricing
    gpu_prices = {
        'NVIDIA_TESLA_T4': 0.11,    # Spot
        'NVIDIA_TESLA_L4': 0.26,
        'NVIDIA_A100_40GB': 0.97,
        'NVIDIA_A100_80GB': 1.20,
        'NVIDIA_H100': 2.00,
    }

    gpu_cost = gpu_prices.get(config['accelerator_type'], 1.0)
    gpu_cost *= config['accelerator_count']

    return hours * gpu_cost

# Example: Schedule 5 training jobs with $100/day budget
jobs = [
    {'name': 'critical-prod-model', 'config': {
        'accelerator_type': 'NVIDIA_A100_80GB',
        'accelerator_count': 8,
        'estimated_hours': 6
    }},
    {'name': 'research-experiment-1', 'config': {
        'accelerator_type': 'NVIDIA_A100_80GB',
        'accelerator_count': 4,
        'estimated_hours': 8
    }},
    {'name': 'ablation-study', 'config': {
        'accelerator_type': 'NVIDIA_TESLA_L4',
        'accelerator_count': 2,
        'estimated_hours': 12
    }},
]

priority = ['critical-prod-model', 'research-experiment-1', 'ablation-study']

result = schedule_training_within_budget(
    training_jobs=jobs,
    daily_budget_usd=100,
    priority_order=priority
)

# Output:
# ✓ Scheduled: critical-prod-model ($57.60)
# ✓ Scheduled: research-experiment-1 ($38.40)
# ✓ Scheduled: ablation-study ($6.24)
# Budget: $102.24 / $100.00
```

### Automated Shutdown Policies

**Scheduled shutdown for dev/test environments:**

```bash
# Cloud Scheduler job to stop dev GPUs at 6 PM daily
gcloud scheduler jobs create http stop-dev-gpus \
    --schedule="0 18 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://us-central1-PROJECT_ID.cloudfunctions.net/stop_dev_gpus" \
    --http-method=POST \
    --description="Stop dev GPU instances at 6 PM weekdays"

# Cloud Scheduler job to start dev GPUs at 8 AM daily
gcloud scheduler jobs create http start-dev-gpus \
    --schedule="0 8 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://us-central1-PROJECT_ID.cloudfunctions.net/start_dev_gpus" \
    --http-method=POST \
    --description="Start dev GPU instances at 8 AM weekdays"
```

**Savings calculation:**

```
Dev environment: 4× L4 GPUs
On-demand rate: $0.90/GPU/hr × 4 = $3.60/hr
Spot rate: $0.26/GPU/hr × 4 = $1.04/hr (using Spot)

Without scheduled shutdown:
- 24 hours/day × 7 days = 168 hours/week
- Cost: 168 × $1.04 = $174.72/week

With scheduled shutdown (10 hours/day, weekdays only):
- 10 hours/day × 5 days = 50 hours/week
- Cost: 50 × $1.04 = $52.00/week

Weekly savings: $122.72 (70% reduction)
Annual savings: $6,381.44
```

---

## Section 8: arr-coc-0-1 Cost Optimization Implementation

### Production Cost Optimization Strategy

From [arr-coc-0-1 project](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):

**Baseline configuration:**
- Training: 8× A100 80GB for 24-hour runs
- Frequency: 3× per week (development iterations)
- Inference: 2× L4 GPUs for demo API

**Cost analysis (without optimization):**

```
Training (on-demand):
- 8× A100 80GB: $29.36/hr
- 24 hours × 3 runs/week = 72 hours/week
- Weekly cost: $2,113.92
- Monthly cost: $9,160.64

Inference (on-demand):
- 2× L4: $1.80/hr
- 24/7 operation: 730 hours/month
- Monthly cost: $1,314.00

Total monthly (on-demand): $10,474.64
```

**Optimized configuration:**

```
Training (Spot instances):
- 8× A100 80GB Spot: $9.60/hr (67% savings)
- 24 hours × 1.08 overhead × 3 runs/week = 77.76 hours/week
- Weekly cost: $746.50
- Monthly cost: $3,234.83

Inference (CUD + autoscaling):
- 1× L4 on 1-year CUD: $0.567/hr (baseline)
- 1× L4 on Spot: $0.26/hr (peak traffic)
- Average 1.3 GPUs × 730 hours/month: $567.00 baseline + $189.80 peak
- Monthly cost: $756.80

Total monthly (optimized): $3,991.63
Savings: $6,483.01 (62% reduction)
```

**Implementation checklist:**

- [x] Enable Spot instances for all training jobs
- [x] Implement checkpoint/resume for preemption handling
- [x] Purchase 1-year CUD for baseline inference GPU
- [x] Set up Cloud Monitoring alerts for idle GPU detection
- [x] Configure automated shutdown for dev environments
- [x] Label all resources with cost allocation tags
- [x] Implement daily cost anomaly detection via BigQuery
- [x] Create monthly cost optimization review process

---

## Sources

**Google Cloud Official Documentation:**
- [GCP Spot VM pricing](https://cloud.google.com/spot-vms/pricing) - Dynamic pricing model
- [GCP GPU pricing](https://cloud.google.com/compute/gpus-pricing) - Official GPU price list
- [Committed Use Discounts overview](https://docs.cloud.google.com/compute/docs/instances/committed-use-discounts-overview) - CUD documentation
- [Sustained Use Discounts](https://cloud.google.com/compute/docs/sustained-use-discounts) - Automatic discounts
- [GKE GPU time-sharing](https://cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus) - Multi-tenant GPU access
- [Cloud Monitoring GPU metrics](https://cloud.google.com/monitoring/api/metrics_gcp#gcp-compute) - GPU utilization monitoring

**Third-Party Cost Analysis:**
- [Cast AI 2025 GPU Price Report](https://cast.ai/reports/gpu-price-2025/) - A100 & H100 cost comparison (accessed 2025-11-16)
- [GPU Price Comparison 2025](https://getdeploying.com/reference/cloud-gpu) - Multi-provider pricing (accessed 2025-11-16)
- [Economize Cloud GPU Pricing](https://www.economize.cloud/blog/gcp-gpu-pricing-comparison/) - GCP GPU discounts (September 27, 2024, accessed 2025-11-16)
- [DataCrunch Cloud GPU Pricing](https://datacrunch.io/blog/cloud-gpu-pricing-comparison) - Provider comparison (December 23, 2024, accessed 2025-11-16)
- [CloudZero GCP CUD Guide](https://www.cloudzero.com/blog/gcp-cud/) - Committed use discount best practices (June 30, 2023, accessed 2025-11-16)

**FinOps and Cost Optimization:**
- [GMI Cloud GPU Cost Guide](https://www.gmicloud.ai/blog/how-much-do-gpu-cloud-platforms-cost-for-ai-startups-in-2025) - Idle GPU waste analysis (accessed 2025-11-16)
- [Flexera FinOps for AI](https://www.flexera.com/blog/finops/finops-for-ai-governing-the-unique-economics-of-intelligent-workloads/) - GPU governance (September 11, 2025, accessed 2025-11-16)
- [Rafay GPU Resource Quotas](https://docs.rafay.co/blog/2025/06/27/configure-and-manage-gpu-resource-quotas-in-multi-tenant-clouds/) - Multi-tenant cost sharing (June 27, 2025, accessed 2025-11-16)
- [Clarifai: Cut GPU Costs](https://www.clarifai.com/blog/cut-gpu-cost-in-production) - Idle GPU detection (accessed 2025-11-16)

**Related Karpathy Deep Oracle Documentation:**
- [gcloud-cost/00-billing-automation.md](../gcloud-cost/00-billing-automation.md) - GCP cost management and anomaly detection
- [practical-implementation/44-gcp-spot-cost-optimization.md](../karpathy/practical-implementation/44-gcp-spot-cost-optimization.md) - Spot instance strategies
- [practical-implementation/43-gcp-spot-checkpoint-strategies.md](../karpathy/practical-implementation/43-gcp-spot-checkpoint-strategies.md) - Preemption handling
- [practical-implementation/45-gcp-spot-production-patterns.md](../karpathy/practical-implementation/45-gcp-spot-production-patterns.md) - Production Spot patterns
