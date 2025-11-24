# GCP Spot VM Advanced Strategies

## Overview

Spot VMs (formerly Preemptible VMs) offer up to 91% cost savings for fault-tolerant workloads but require sophisticated strategies to handle preemptions gracefully. This guide covers advanced patterns for checkpoint-resume workflows, cost tracking automation, preemption handling, and hybrid architectures.

**Key Difference from Basic Spot Usage**: This guide assumes you understand spot VM fundamentals (covered in [karpathy/practical-implementation/38-gcp-spot-fundamentals.md](../karpathy/practical-implementation/38-gcp-spot-fundamentals.md) and [cloud-build-advanced/00-beta-features.md](../cloud-build-advanced/00-beta-features.md) spot sections). Here we focus on **production-grade strategies** for long-running training jobs, cost optimization, and reliability patterns.

From [Vertex AI Spot VMs Documentation](https://docs.cloud.google.com/vertex-ai/docs/training/use-spot-vms) (accessed 2025-02-03):
- Spot VMs can be reclaimed at any time with 30-second notice
- Best suited for fault-tolerant workloads with checkpointing
- Billing: only pay for actual usage time (no minimum)

## Section 1: Advanced Spot Strategies (~150 lines)

### Multi-Cloud Spot Arbitrage

**Pattern**: Leverage spot availability across multiple clouds to maximize uptime and minimize costs.

From [SkyServe Research Paper](https://arxiv.org/html/2411.01438v2) (arXiv:2411.01438v2, accessed 2025-02-03):
- SkyServe implements "SpotHedge" - mixture of spot and on-demand replicas across regions/clouds
- Monitors spot availability and prices in real-time
- Automatically migrates workloads when spot instances become unavailable
- Achieves 3-4x cost savings vs pure on-demand while maintaining 99.9% uptime

**Implementation Strategy**:
```yaml
# Multi-region spot pool configuration
regions:
  - us-west2  # Primary region
  - us-central1  # Failover region 1
  - us-east1  # Failover region 2

spot_config:
  max_price: 0.30  # Maximum spot price per hour
  fallback_to_ondemand: true  # Use on-demand if spot unavailable
  min_spot_percentage: 70  # Target 70% spot usage
```

**Cost-Benefit Analysis**:
- Pure spot: 91% savings, but ~5-10% interruption rate
- Hybrid (70% spot + 30% on-demand): 64% savings, <1% interruption rate
- Multi-region spot: 80% savings, <2% interruption rate (better availability)

### Spot Instance Pools

**Pattern**: Maintain a pool of warm spot instances ready to accept work immediately.

From [Google Cloud Spot VM Best Practices](https://cloud.google.com/blog/products/compute/google-cloud-spot-vm-use-cases-and-best-practices) (accessed 2025-02-03):
- Financial modeling firms use spot pools for Monte Carlo simulations
- CI/CD pipelines benefit from warm instance pools (faster startup)
- Web services use spot pools for burst capacity

**Pool Architecture**:
```
┌─────────────────────────────────────┐
│        Spot Instance Pool           │
│                                     │
│  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │ Warm │  │ Warm │  │ Warm │     │ ← Pre-warmed instances
│  │ GPU  │  │ GPU  │  │ GPU  │     │   (Docker cached, deps installed)
│  └──────┘  └──────┘  └──────┘     │
│      ↓          ↓          ↓       │
│  ┌──────────────────────────────┐ │
│  │   Job Queue (W&B Launch)     │ │
│  └──────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Implementation** (Python + GCP APIs):
```python
# spot_pool_manager.py
from google.cloud import compute_v1
import time

class SpotPoolManager:
    def __init__(self, project_id, zone, target_pool_size=5):
        self.compute = compute_v1.InstancesClient()
        self.project = project_id
        self.zone = zone
        self.target_size = target_pool_size

    def maintain_pool(self):
        """Keep target number of warm spot instances ready"""
        while True:
            current_instances = self.get_pool_instances()
            active_count = len([i for i in current_instances if i.status == "RUNNING"])

            if active_count < self.target_size:
                needed = self.target_size - active_count
                print(f"Pool needs {needed} instances, creating...")
                self.create_spot_instances(needed)

            time.sleep(60)  # Check every minute

    def create_spot_instances(self, count):
        """Create spot instances with warm Docker cache"""
        for i in range(count):
            instance_config = {
                "name": f"spot-pool-{int(time.time())}-{i}",
                "machine_type": f"zones/{self.zone}/machineTypes/n1-standard-8",
                "scheduling": {
                    "preemptible": True,
                    "automatic_restart": False,
                },
                # Startup script: pull Docker images, cache deps
                "metadata": {
                    "items": [{
                        "key": "startup-script",
                        "value": """#!/bin/bash
                            docker pull gcr.io/project/training-image:latest
                            docker run --rm gcr.io/project/training-image:latest python -c 'import torch'
                            echo 'Instance warmed and ready'
                        """
                    }]
                }
            }

            operation = self.compute.insert(
                project=self.project,
                zone=self.zone,
                instance_resource=instance_config
            )
            print(f"Created spot instance: {instance_config['name']}")
```

**Pool Benefits**:
- **Fast job startup**: Instances pre-warmed with Docker cache (~30 seconds vs 5 minutes)
- **Better spot availability**: Spreads requests across time (not all at once)
- **Cost efficient**: Only pay for instances when they're running jobs

### Bid Strategy Optimization

**Pattern**: Dynamically adjust spot instance "bids" based on historical pricing data.

From [CloudBolt Spot VMs Guide](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) (accessed 2025-02-03):
- Spot prices vary by region, time of day, and demand
- No formal bidding on GCP (preemption based on capacity), but can set max price
- Historical pricing data helps predict low-cost windows

**Historical Pricing Analysis** (example data):
```
Time Window          | Average Spot Price (n1-standard-8)
---------------------|----------------------------------
Mon-Fri 9am-5pm EST  | $0.08/hour (higher demand)
Mon-Fri 6pm-8am EST  | $0.04/hour (lower demand)
Weekends             | $0.03/hour (lowest demand)
```

**Time-Aware Scheduling**:
```python
import datetime

def should_use_spot(current_price, max_price=0.06):
    """Decide spot vs on-demand based on current price and time"""
    hour = datetime.datetime.now().hour
    day = datetime.datetime.now().weekday()

    # Weekend: always use spot (low prices)
    if day >= 5:
        return True

    # Weekday off-hours: use spot if price reasonable
    if hour < 9 or hour > 17:
        return current_price <= max_price

    # Peak hours: only use spot if very cheap
    return current_price <= max_price * 0.8
```

### Spot + Committed Use Discounts (CUD)

**Pattern**: Combine spot instances with committed use discounts for optimal cost structure.

**Hybrid Architecture**:
```
Baseline workload (always running):
  → Use Committed Use Discount (1-year or 3-year)
  → Example: 4x n1-standard-8 instances = $400/month (57% discount)

Burst workload (variable):
  → Use Spot VMs for additional capacity
  → Example: 0-20 spot instances based on queue depth
```

**Cost Comparison** (100 hours of n1-standard-8 compute):
```
Strategy                      | Cost    | Savings vs On-Demand
------------------------------|---------|---------------------
Pure On-Demand                | $270    | 0% (baseline)
Pure Spot (no preemptions)    | $30     | 89%
Spot + CUD Baseline (hybrid)  | $58     | 78% (more reliable)
```

From [Pump.co GCP Optimization](https://www.pump.co/blog/spot-instances-gcp) (accessed 2025-02-03):
- Combining CUDs with spot provides "base + burst" cost optimization
- CUDs cover predictable workload, spot handles spikes
- Achieves 70-85% cost reduction with higher reliability than pure spot

## Section 2: Checkpoint-Resume Patterns (~150 lines)

### Robust Checkpointing Architecture

**Critical Requirements** for spot instance training:
1. **Periodic checkpointing**: Save progress every N steps/minutes
2. **Automatic resume**: Always attempt to load latest checkpoint on startup
3. **Checkpoint validation**: Verify checkpoint integrity before loading
4. **Fallback mechanism**: Use previous checkpoint if latest is corrupted

From [Vertex AI Training Best Practices](https://docs.cloud.google.com/vertex-ai/docs/training/understanding-training-service) (accessed 2025-02-03):
- For training >4 hours: checkpoint at least every 4 hours
- Vertex AI provides 30-second preemption notice
- Use Cloud Storage for checkpoint persistence

### Multi-Tier Checkpointing

**Pattern**: Use multiple storage tiers for checkpoint speed and durability.

From [Google Cloud Multi-Tier Checkpointing](https://cloud.google.com/blog/products/ai-machine-learning/using-multi-tier-checkpointing-for-large-ai-training-jobs) (accessed 2025-06-16, note: accessed 2025-02-03):
- **Tier 1**: Local NVMe SSD (fastest writes, volatile)
- **Tier 2**: Persistent disk (medium speed, survives preemption)
- **Tier 3**: Cloud Storage (slowest writes, globally accessible)

**Architecture**:
```
Training Loop
    ↓
[Every 100 steps] → Tier 1: Local SSD
    ↓               (/tmp/checkpoints/latest.pt)
    ↓               Write time: ~2 seconds
    ↓
[Every 1000 steps] → Tier 2: Persistent Disk
    ↓                (/mnt/pd/checkpoints/step_5000.pt)
    ↓                Write time: ~15 seconds
    ↓
[Every 10000 steps] → Tier 3: Cloud Storage
                       (gs://bucket/checkpoints/step_50000.pt)
                       Write time: ~2 minutes
```

**Benefits**:
- **Fast recovery**: Tier 1 checkpoint available immediately after preemption
- **Durability**: Tier 3 checkpoint survives complete instance loss
- **Reduced MTTR**: Mean Time To Recovery drops from 15 minutes → 2 minutes

**Implementation Example** (PyTorch):
```python
# multi_tier_checkpoint.py
from pathlib import Path
import torch
import time

class MultiTierCheckpointer:
    def __init__(self,
                 tier1_dir="/tmp/checkpoints",
                 tier2_dir="/mnt/pd/checkpoints",
                 tier3_bucket="gs://training-checkpoints"):
        self.tier1 = Path(tier1_dir)
        self.tier2 = Path(tier2_dir)
        self.tier3 = tier3_bucket

        self.tier1.mkdir(parents=True, exist_ok=True)
        self.tier2.mkdir(parents=True, exist_ok=True)

    def save(self, step, model, optimizer):
        """Save checkpoint to appropriate tiers based on step"""
        checkpoint = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'timestamp': time.time()
        }

        # Tier 1: Every step (fast local)
        tier1_path = self.tier1 / "latest.pt"
        torch.save(checkpoint, tier1_path)

        # Tier 2: Every 1000 steps (persistent disk)
        if step % 1000 == 0:
            tier2_path = self.tier2 / f"step_{step}.pt"
            torch.save(checkpoint, tier2_path)
            print(f"Saved Tier 2 checkpoint: {tier2_path}")

        # Tier 3: Every 10000 steps (cloud storage)
        if step % 10000 == 0:
            tier3_path = f"{self.tier3}/step_{step}.pt"
            # Use gsutil for cloud storage
            import subprocess
            temp_path = f"/tmp/checkpoint_upload_{step}.pt"
            torch.save(checkpoint, temp_path)
            subprocess.run(["gsutil", "cp", temp_path, tier3_path], check=True)
            print(f"Saved Tier 3 checkpoint: {tier3_path}")

    def load(self, model, optimizer):
        """Load most recent valid checkpoint (try Tier 1 → 2 → 3)"""
        # Try Tier 1 first (fastest)
        tier1_path = self.tier1 / "latest.pt"
        if tier1_path.exists():
            try:
                checkpoint = torch.load(tier1_path)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Loaded Tier 1 checkpoint from step {checkpoint['step']}")
                return checkpoint['step']
            except Exception as e:
                print(f"Tier 1 checkpoint corrupted: {e}")

        # Try Tier 2 (persistent disk)
        tier2_checkpoints = sorted(self.tier2.glob("step_*.pt"), reverse=True)
        for ckpt_path in tier2_checkpoints:
            try:
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Loaded Tier 2 checkpoint from step {checkpoint['step']}")
                return checkpoint['step']
            except Exception as e:
                print(f"Tier 2 checkpoint {ckpt_path} corrupted: {e}")
                continue

        # Try Tier 3 (cloud storage) - slowest but most durable
        import subprocess
        result = subprocess.run(
            ["gsutil", "ls", f"{self.tier3}/step_*.pt"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            cloud_checkpoints = sorted(result.stdout.strip().split('\n'), reverse=True)
            for ckpt_url in cloud_checkpoints:
                try:
                    temp_path = f"/tmp/checkpoint_download.pt"
                    subprocess.run(["gsutil", "cp", ckpt_url, temp_path], check=True)
                    checkpoint = torch.load(temp_path)
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"Loaded Tier 3 checkpoint from step {checkpoint['step']}")
                    return checkpoint['step']
                except Exception as e:
                    print(f"Tier 3 checkpoint {ckpt_url} corrupted: {e}")
                    continue

        print("No valid checkpoint found, starting from scratch")
        return 0
```

### Incremental Checkpointing

**Pattern**: Only save model parameter deltas to reduce checkpoint size and write time.

**Standard Checkpoint**:
- Full model state: 7GB (for LLaMA-7B)
- Write time to GCS: ~3 minutes
- Checkpoint every 1000 steps → 15% time spent checkpointing

**Incremental Checkpoint**:
- Base checkpoint: 7GB (saved once)
- Delta checkpoints: 200-500MB per step
- Write time to GCS: ~20 seconds
- Checkpoint every 100 steps → 2% time spent checkpointing

**Implementation**:
```python
# incremental_checkpoint.py
import torch
import copy

class IncrementalCheckpointer:
    def __init__(self, base_checkpoint_path):
        self.base_path = base_checkpoint_path
        self.base_state = None
        self.last_step = 0

    def save_base(self, model):
        """Save full base checkpoint (only once)"""
        self.base_state = copy.deepcopy(model.state_dict())
        torch.save(self.base_state, self.base_path)
        print(f"Saved base checkpoint: {self.base_path}")

    def save_delta(self, step, model, checkpoint_dir):
        """Save only changed parameters"""
        current_state = model.state_dict()
        delta = {}

        for key in current_state.keys():
            if key not in self.base_state:
                delta[key] = current_state[key]
            else:
                diff = current_state[key] - self.base_state[key]
                # Only save if difference is significant
                if torch.sum(torch.abs(diff)) > 1e-5:
                    delta[key] = diff

        delta_path = f"{checkpoint_dir}/delta_step_{step}.pt"
        torch.save({'step': step, 'delta': delta}, delta_path)
        print(f"Saved delta checkpoint: {delta_path} ({len(delta)} changed params)")

    def load_with_deltas(self, model, delta_dir):
        """Load base + all deltas up to latest step"""
        # Load base
        base_state = torch.load(self.base_path)
        model.load_state_dict(base_state)

        # Apply deltas in order
        import glob
        delta_files = sorted(glob.glob(f"{delta_dir}/delta_step_*.pt"))
        for delta_file in delta_files:
            delta_ckpt = torch.load(delta_file)
            delta = delta_ckpt['delta']

            current_state = model.state_dict()
            for key, diff in delta.items():
                if key in current_state:
                    current_state[key] += diff
                else:
                    current_state[key] = diff

            model.load_state_dict(current_state)
            print(f"Applied delta from step {delta_ckpt['step']}")

        return delta_ckpt['step']
```

### Checkpoint Compression

**Pattern**: Compress checkpoints to reduce storage costs and transfer time.

**Compression Comparison**:
```
Method          | Size (7GB model) | Compression Time | Decompression Time
----------------|------------------|------------------|--------------------
None            | 7.0 GB           | 0s               | 0s
gzip            | 3.2 GB (54%)     | 45s              | 15s
zstd (level 3)  | 3.5 GB (50%)     | 8s               | 3s
zstd (level 10) | 2.8 GB (60%)     | 60s              | 3s
```

From [Massed Compute Checkpointing Guide](https://massedcompute.com/faq-answers/?question=What%20are%20the%20best%20practices%20for%20checkpointing%20and%20resuming%20AI%20model%20training%20jobs%20on%20spot%20instances?) (accessed 2025-02-03):
- Regularly save checkpoints (frequency depends on training duration and risk tolerance)
- Compress checkpoints to reduce storage costs and transfer time
- Use fast compression (zstd level 3) for frequent checkpoints

**Implementation**:
```python
import torch
import zstandard as zstd

def save_compressed_checkpoint(checkpoint, path, compression_level=3):
    """Save checkpoint with zstd compression"""
    # Save to temporary buffer
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    # Compress with zstd
    compressor = zstd.ZstdCompressor(level=compression_level)
    compressed_data = compressor.compress(buffer.read())

    # Write compressed data
    with open(f"{path}.zst", 'wb') as f:
        f.write(compressed_data)

    print(f"Saved compressed checkpoint: {path}.zst "
          f"({len(compressed_data) / (1024**3):.2f} GB)")

def load_compressed_checkpoint(path):
    """Load zstd-compressed checkpoint"""
    # Read compressed data
    with open(f"{path}.zst", 'rb') as f:
        compressed_data = f.read()

    # Decompress with zstd
    decompressor = zstd.ZstdDecompressor()
    decompressed_data = decompressor.decompress(compressed_data)

    # Load checkpoint from buffer
    buffer = io.BytesIO(decompressed_data)
    checkpoint = torch.load(buffer)

    return checkpoint
```

## Section 3: Cost Tracking Automation (~150 lines)

### Real-Time Cost Monitoring

**Pattern**: Track spot instance costs in real-time and alert when budgets are exceeded.

**Monitoring Architecture**:
```
┌─────────────────────────────────────────────────────┐
│              Cloud Monitoring                       │
│                                                     │
│  ┌──────────────┐      ┌──────────────┐          │
│  │ Spot VM      │      │ Cost         │          │
│  │ Metrics      │ ───▶ │ Aggregator   │          │
│  │ (Stackdriver)│      │ (Cloud Fn)   │          │
│  └──────────────┘      └──────────────┘          │
│         ↓                      ↓                   │
│  ┌──────────────┐      ┌──────────────┐          │
│  │ BigQuery     │      │ Budget       │          │
│  │ Cost Table   │      │ Alerts       │          │
│  └──────────────┘      └──────────────┘          │
└─────────────────────────────────────────────────────┘
         ↓                       ↓
    Dashboard              Email/Slack Alert
```

**Implementation** (Cloud Function + BigQuery):
```python
# cost_tracker.py (Cloud Function)
from google.cloud import monitoring_v3
from google.cloud import bigquery
import datetime

def track_spot_costs(request):
    """Cloud Function: Track spot VM costs every hour"""
    project_id = "my-project"

    # Query Cloud Monitoring for spot instance usage
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    interval = monitoring_v3.TimeInterval({
        "end_time": {"seconds": int(datetime.datetime.now().timestamp())},
        "start_time": {"seconds": int((datetime.datetime.now() - datetime.timedelta(hours=1)).timestamp())}
    })

    # Get compute instance metrics
    results = client.list_time_series(
        request={
            "name": project_name,
            "filter": 'metric.type="compute.googleapis.com/instance/uptime" AND resource.labels.preemptible="true"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
        }
    )

    # Calculate costs
    bq_client = bigquery.Client()
    total_cost = 0

    for result in results:
        instance_type = result.resource.labels.get('instance_type', 'unknown')
        zone = result.resource.labels.get('zone', 'unknown')

        # Get spot pricing for instance type
        spot_price = get_spot_price(instance_type, zone)

        # Calculate uptime hours
        uptime_hours = sum([point.value.double_value for point in result.points]) / 3600

        # Calculate cost
        cost = uptime_hours * spot_price
        total_cost += cost

        # Insert into BigQuery
        table_id = f"{project_id}.cost_tracking.spot_vm_costs"
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "instance_type": instance_type,
            "zone": zone,
            "uptime_hours": uptime_hours,
            "spot_price": spot_price,
            "cost": cost
        }
        bq_client.insert_rows_json(table_id, [row])

    # Check budget and alert if exceeded
    check_budget_and_alert(total_cost)

    return {"status": "success", "total_cost": total_cost}

def get_spot_price(instance_type, zone):
    """Get current spot price for instance type in zone"""
    # Simplified pricing (in practice, query GCP pricing API)
    pricing = {
        "n1-standard-8": 0.08,
        "n1-highmem-8": 0.10,
        "n1-standard-16": 0.16,
    }
    return pricing.get(instance_type, 0.10)

def check_budget_and_alert(current_cost):
    """Alert if budget threshold exceeded"""
    DAILY_BUDGET = 100.0  # $100/day

    if current_cost > DAILY_BUDGET / 24:  # Hourly threshold
        send_alert(f"Spot VM costs exceed hourly budget: ${current_cost:.2f}/hour")
```

**BigQuery Cost Analysis Dashboard**:
```sql
-- Daily spot VM cost breakdown by instance type
SELECT
  DATE(timestamp) as date,
  instance_type,
  zone,
  SUM(uptime_hours) as total_hours,
  AVG(spot_price) as avg_price,
  SUM(cost) as total_cost
FROM `project.cost_tracking.spot_vm_costs`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY date, instance_type, zone
ORDER BY date DESC, total_cost DESC

-- Cost efficiency: Spot vs On-Demand savings
WITH spot_costs AS (
  SELECT DATE(timestamp) as date, SUM(cost) as spot_cost
  FROM `project.cost_tracking.spot_vm_costs`
  GROUP BY date
),
ondemand_equivalent AS (
  SELECT DATE(timestamp) as date, SUM(uptime_hours * ondemand_price) as ondemand_cost
  FROM `project.cost_tracking.spot_vm_costs`
  GROUP BY date
)
SELECT
  s.date,
  s.spot_cost,
  o.ondemand_cost,
  (o.ondemand_cost - s.spot_cost) as savings,
  ((o.ondemand_cost - s.spot_cost) / o.ondemand_cost * 100) as savings_percentage
FROM spot_costs s
JOIN ondemand_equivalent o ON s.date = o.date
ORDER BY s.date DESC
```

### Cost Attribution and Tagging

**Pattern**: Tag spot instances with project/team/experiment metadata for cost allocation.

**Tagging Strategy**:
```python
# spot_instance_tagger.py
from google.cloud import compute_v1

def create_tagged_spot_instance(project_id, zone, instance_name,
                                 team, project, experiment):
    """Create spot instance with cost allocation tags"""
    compute = compute_v1.InstancesClient()

    instance_config = {
        "name": instance_name,
        "machine_type": f"zones/{zone}/machineTypes/n1-standard-8",
        "scheduling": {
            "preemptible": True,
            "automatic_restart": False,
        },
        "labels": {
            "team": team,
            "project": project,
            "experiment": experiment,
            "cost_center": f"{team}_{project}",
            "instance_type": "spot",
        },
        # Metadata for tracking
        "metadata": {
            "items": [
                {"key": "created_by", "value": "training_automation"},
                {"key": "cost_tracking_enabled", "value": "true"},
            ]
        }
    }

    operation = compute.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance_config
    )

    print(f"Created tagged spot instance: {instance_name}")
    print(f"Tags: team={team}, project={project}, experiment={experiment}")

    return instance_name
```

**Cost Allocation Query** (BigQuery):
```sql
-- Monthly cost breakdown by team/project
SELECT
  EXTRACT(MONTH FROM timestamp) as month,
  labels.team,
  labels.project,
  COUNT(DISTINCT instance_name) as num_instances,
  SUM(uptime_hours) as total_hours,
  SUM(cost) as total_cost
FROM `project.cost_tracking.spot_vm_costs`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 MONTH)
GROUP BY month, labels.team, labels.project
ORDER BY month DESC, total_cost DESC
```

### Automated Budget Enforcement

**Pattern**: Automatically stop spot instances when budget limits are reached.

**Implementation** (Cloud Function + Pub/Sub):
```python
# budget_enforcer.py (Cloud Function triggered by budget alert)
from google.cloud import compute_v1
import json

def enforce_budget_limit(event, context):
    """Stop spot instances when budget exceeded"""
    # Parse budget alert from Pub/Sub
    pubsub_message = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    budget_exceeded = pubsub_message.get('costAmount', 0) > pubsub_message.get('budgetAmount', 0)

    if not budget_exceeded:
        print("Budget OK, no action needed")
        return

    print(f"Budget exceeded! Stopping lowest-priority spot instances...")

    project_id = pubsub_message['project']
    compute = compute_v1.InstancesClient()

    # List all spot instances
    aggregated_list = compute.aggregated_list(project=project_id)
    spot_instances = []

    for zone, response in aggregated_list:
        if response.instances:
            for instance in response.instances:
                if instance.scheduling.preemptible:
                    priority = instance.labels.get('priority', 'low')
                    spot_instances.append({
                        'name': instance.name,
                        'zone': zone.split('/')[-1],
                        'priority': priority
                    })

    # Sort by priority (stop low priority first)
    priority_order = {'low': 0, 'medium': 1, 'high': 2}
    spot_instances.sort(key=lambda x: priority_order.get(x['priority'], 0))

    # Stop lowest priority instances until budget back under limit
    stopped_count = 0
    for instance in spot_instances:
        if stopped_count >= 3:  # Stop max 3 instances per alert
            break

        compute.stop(
            project=project_id,
            zone=instance['zone'],
            instance=instance['name']
        )
        print(f"Stopped spot instance: {instance['name']} (priority={instance['priority']})")
        stopped_count += 1

    return {"status": "enforced", "stopped_instances": stopped_count}
```

**Budget Alert Configuration** (gcloud):
```bash
# Create budget with alert at 80% and 100%
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Spot VM Budget" \
  --budget-amount=1000 \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100 \
  --all-updates-rule-pubsub-topic=projects/PROJECT_ID/topics/budget-alerts

# Create Pub/Sub topic for budget alerts
gcloud pubsub topics create budget-alerts

# Deploy budget enforcement Cloud Function
gcloud functions deploy enforce_budget_limit \
  --runtime=python39 \
  --trigger-topic=budget-alerts \
  --entry-point=enforce_budget_limit \
  --source=.
```

## Section 4: Preemption Handling (~100 lines)

### Preemption Signal Detection

**Pattern**: Detect preemption notice and save checkpoint before termination.

From [GCP Preemptible VM Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) (accessed 2025-02-03):
- Spot VMs receive 30-second preemption notice via metadata server
- ACPI G2 soft-off signal sent to instance
- Shutdown scripts have ~30 seconds to execute

**Preemption Detection Script**:
```python
# preemption_handler.py
import requests
import signal
import sys
import time
import torch

class PreemptionHandler:
    def __init__(self, checkpoint_fn):
        self.checkpoint_fn = checkpoint_fn
        self.preempted = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_preemption)
        signal.signal(signal.SIGINT, self.handle_preemption)

        # Start metadata polling thread
        import threading
        self.monitor_thread = threading.Thread(target=self.poll_preemption_metadata)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def poll_preemption_metadata(self):
        """Poll GCP metadata server for preemption notice"""
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
        headers = {"Metadata-Flavor": "Google"}

        while not self.preempted:
            try:
                response = requests.get(metadata_url, headers=headers, timeout=1)
                if response.text == "TRUE":
                    print("PREEMPTION NOTICE RECEIVED - Saving checkpoint...")
                    self.handle_preemption(None, None)
            except Exception as e:
                pass  # Metadata not available yet or network error

            time.sleep(5)  # Poll every 5 seconds

    def handle_preemption(self, signum, frame):
        """Handle preemption by saving emergency checkpoint"""
        if self.preempted:
            return  # Already handling preemption

        self.preempted = True
        print(f"Preemption signal received (signal={signum})")
        print("Saving emergency checkpoint...")

        try:
            # Call checkpoint function (should be fast!)
            self.checkpoint_fn()
            print("Emergency checkpoint saved successfully")
        except Exception as e:
            print(f"Failed to save emergency checkpoint: {e}")

        sys.exit(0)

# Usage in training script
def main():
    model = create_model()
    optimizer = create_optimizer()

    # Setup preemption handler
    def emergency_checkpoint():
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': current_step
        }, '/tmp/emergency_checkpoint.pt')

    handler = PreemptionHandler(checkpoint_fn=emergency_checkpoint)

    # Training loop
    for step in range(max_steps):
        # ... training code ...

        # Regular checkpoint every 1000 steps
        if step % 1000 == 0:
            save_checkpoint(step, model, optimizer)
```

### Graceful Degradation

**Pattern**: Reduce resource usage when preemption is imminent to save costs.

**Adaptive Batch Size**:
```python
# adaptive_training.py
import time

class AdaptiveTrainer:
    def __init__(self, initial_batch_size=32):
        self.batch_size = initial_batch_size
        self.preemption_likely = False

    def check_preemption_risk(self):
        """Check if preemption is likely (heuristics)"""
        # Check instance uptime (spot VMs more likely to be preempted after 24h)
        uptime_hours = self.get_instance_uptime()
        if uptime_hours > 20:
            self.preemption_likely = True
            return True

        # Check regional spot availability (custom logic)
        spot_availability = self.query_spot_availability()
        if spot_availability < 0.3:
            self.preemption_likely = True
            return True

        return False

    def train_step(self, model, data_loader):
        """Training step with adaptive batch size"""
        if self.preemption_likely:
            # Reduce batch size to checkpoint more frequently
            self.batch_size = max(8, self.batch_size // 2)
            print(f"Preemption risk detected, reducing batch size to {self.batch_size}")

        batch = next(iter(data_loader))
        loss = model(batch)
        loss.backward()

        return loss.item()
```

## Section 5: Hybrid Architectures (~50 lines)

### Spot + On-Demand Hybrid

**Pattern**: Use spot for workers, on-demand for critical coordinator.

**Architecture**:
```
┌─────────────────────────────────────┐
│  On-Demand Instance (Coordinator)   │ ← Always available
│  - Training orchestration           │   (3% of total cost)
│  - Checkpoint management            │
│  - Job queue coordination           │
└─────────────────────────────────────┘
           ↓        ↓        ↓
    ┌──────────┬──────────┬──────────┐
    │ Spot GPU │ Spot GPU │ Spot GPU │ ← 97% cost savings
    │ Worker 1 │ Worker 2 │ Worker 3 │   (can be preempted)
    └──────────┴──────────┴──────────┘
```

**Benefits**:
- Coordinator ensures training continuity even if workers preempted
- Workers can be replaced quickly (coordinator maintains state)
- Cost savings: 85-90% vs pure on-demand (vs 91% pure spot)

From [Google Cloud Spot VM Use Cases](https://cloud.google.com/blog/products/compute/google-cloud-spot-vm-use-cases-and-best-practices) (accessed 2025-02-03):
- Web services use hybrid: spot for batch/background tasks, on-demand for critical services
- CI/CD pipelines: spot for test runners, on-demand for build coordination
- Achieves high availability while maintaining cost savings

## Sources

**Google Cloud Documentation:**
- [Vertex AI Spot VMs](https://docs.cloud.google.com/vertex-ai/docs/training/use-spot-vms) - Checkpointing and billing guidance
- [Preemptible VM Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) - Preemption process and best practices
- [Vertex AI Training Best Practices](https://docs.cloud.google.com/vertex-ai/docs/training/understanding-training-service) - Checkpoint frequency recommendations
- [Multi-Tier Checkpointing Blog](https://cloud.google.com/blog/products/ai-machine-learning/using-multi-tier-checkpointing-for-large-ai-training-jobs) - Multi-tier architecture for large jobs (accessed 2025-06-16, note: accessed 2025-02-03)
- [Google Cloud Spot VM Use Cases](https://cloud.google.com/blog/products/compute/google-cloud-spot-vm-use-cases-and-best-practices) - Best practices and design patterns (accessed 2025-02-03)

**Research Papers:**
- [SkyServe: Serving AI Models across Regions and Clouds](https://arxiv.org/html/2411.01438v2) - arXiv:2411.01438v2, SpotHedge strategy (accessed 2025-02-03)

**Technical Guides:**
- [SkyPilot Training Guide](https://docs.skypilot.co/en/latest/reference/training-guide.html) - Checkpointing patterns and code examples (accessed 2025-02-03)
- [Massed Compute Checkpointing FAQ](https://massedcompute.com/faq-answers/?question=What%20are%20the%20best%20practices%20for%20checkpointing%20and%20resuming%20AI%20model%20training%20jobs%20on%20spot%20instances?) - Best practices for spot training (accessed 2025-02-03)

**Cost Optimization:**
- [Pump.co GCP Spot VMs](https://www.pump.co/blog/spot-instances-gcp) - Cost optimization strategies (accessed 2025-02-03)
- [CloudBolt Spot VMs Guide](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) - Pricing and availability analysis (accessed 2025-02-03)
- [Damco GCP Cost Optimization](https://www.damcogroup.com/blogs/google-cloud-cost-optimization) - Right-sizing and monitoring (accessed 2025-02-03)
- [Lucidity GCP Cost Optimization](https://blog.lucidity.cloud/blog-post/gcp-cost-optimization) - FinOps practices (accessed 2025-02-03)
- [Edge Delta GCP Cost Optimization](https://edgedelta.com/company/blog/gcp-cost-optimization) - Budget alerts and auto-scaling (accessed 2025-02-03)
- [CitiusTech FinOps Dashboard](https://www.citiustech.com/citius-vision/article/mastering-cloud-cost-management-with-gcps-finops-dashboard) - Cloud cost management patterns (accessed 2025-02-03)
- [DZone GCP Cost Management](https://dzone.com/articles/the-ultimate-guide-to-cost-management-in-google) - Right-sizing and automation (accessed 2025-02-03)
- [CloudZero GCP Cost Tools](https://www.cloudzero.com/blog/gcp-cost-optimization-tools/) - Cost optimization tooling overview (accessed 2025-02-03)
- [Medium: GCP Cost Monitoring Secrets](https://ip-specialist.medium.com/the-10-must-know-secrets-for-gcp-cost-monitoring-5e1ab24e8549) - Monitoring best practices (accessed 2025-02-03)

All sources accessed 2025-02-03 unless otherwise noted.
