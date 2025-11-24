# GCP Spot Instance Cost Optimization

## Overview

Cost optimization strategies for Google Cloud Spot VMs combine pricing models, hybrid architectures, and comprehensive cost tracking to achieve 60-91% savings while maintaining production reliability. This guide provides actionable frameworks for maximizing spot instance value in ML training workloads.

**Key cost optimization pillars:**
- **No bidding required** - GCP uses dynamic pricing (unlike AWS/Azure)
- **Hybrid architectures** - Balance cost vs reliability with spot + on-demand
- **Cost tracking** - Billing export to BigQuery for granular analysis
- **Budget controls** - Automated alerts prevent runaway spending
- **Label-based attribution** - Track costs by team/project/experiment

---

## Section 1: GCP Spot Pricing Model (No Bidding Required)

### How GCP Spot Pricing Works

**Critical distinction:** GCP Spot VMs use dynamic pricing, NOT bidding.

From [Google Cloud Spot VM pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31):

| Cloud Provider | Pricing Model | User Control |
|----------------|---------------|--------------|
| **AWS** | Spot bidding | Set max bid price |
| **Azure** | Spot bidding | Set max eviction price |
| **GCP** | Dynamic pricing | No bidding - accept current price |

**GCP Spot pricing characteristics:**
- Prices vary by region/zone/machine type
- Update up to once per day (not real-time)
- Transparent - view current prices in console
- No price spikes from bidding wars
- Simpler cost forecasting

### Spot Discount Rates by Machine Type

**General-Purpose (N1, N2, N2D, E2):**
- N1: 60-91% discount vs on-demand
- N2: 60-91% discount vs on-demand
- N2D: 60-91% discount vs on-demand
- E2: Up to 91% discount (shared-core eligible)

**Compute-Optimized (C2, C3):**
- C2: 60-91% discount vs on-demand
- C3: 60-91% discount (latest generation)

**Memory-Optimized (M1, M2, M3):**
- M1: 60-91% discount vs on-demand
- M2: 60-91% discount (ultramem)
- M3: 60-91% discount (latest)

**Accelerator-Optimized (A2, A3, G2):**
- A2 (A100): 60-70% discount typical
- A3 (H100): 60-70% discount (limited availability)
- G2 (L4): 60-91% discount

### Regional Pricing Variations

**Price examples (spot vs on-demand, approximate):**

```
N2-standard-8 (8 vCPU, 32GB RAM):
- us-central1: $0.07/hr spot vs $0.35/hr on-demand (80% savings)
- us-west1: $0.07/hr spot vs $0.35/hr on-demand (80% savings)
- europe-west4: $0.08/hr spot vs $0.39/hr on-demand (79% savings)

A2-highgpu-1g (12 vCPU, 85GB RAM, 1x A100 40GB):
- us-central1: $1.20/hr spot vs $3.67/hr on-demand (67% savings)
- europe-west4: $1.35/hr spot vs $4.05/hr on-demand (67% savings)

G2-standard-4 (4 vCPU, 16GB RAM, 1x L4):
- us-central1: $0.35/hr spot vs $1.20/hr on-demand (71% savings)
```

**Pricing optimization strategies:**

1. **Regional arbitrage** - Compare prices across regions
2. **Zone selection** - Some zones cheaper within region
3. **Machine type flexibility** - Similar performance, different prices
4. **Time-of-day patterns** - Prices may vary (less predictable than AWS)

### Historical Pricing Analysis

**Unlike AWS, GCP spot prices are more stable:**

- Update frequency: Up to once/day (not every 5 minutes)
- Price volatility: Lower than AWS spot market
- Termination correlation: Not directly tied to price changes
- Forecasting: Easier due to daily price windows

**Cost forecasting methodology:**

```python
# Estimate training cost with spot instances
def estimate_spot_training_cost(
    base_on_demand_hourly: float,
    training_hours: float,
    spot_discount: float = 0.70,  # 70% typical
    preemption_overhead: float = 1.10  # 10% restart overhead
):
    """
    Calculate expected spot training cost with preemption overhead.

    Args:
        base_on_demand_hourly: On-demand hourly rate
        training_hours: Expected training duration
        spot_discount: Spot discount percentage (0.60-0.91)
        preemption_overhead: Additional time from restarts (1.05-1.15)

    Returns:
        Estimated total cost
    """
    spot_hourly = base_on_demand_hourly * (1 - spot_discount)
    adjusted_hours = training_hours * preemption_overhead
    total_cost = spot_hourly * adjusted_hours

    savings_vs_on_demand = (
        (base_on_demand_hourly * training_hours) - total_cost
    )

    return {
        "total_cost": total_cost,
        "savings": savings_vs_on_demand,
        "effective_discount": savings_vs_on_demand / (base_on_demand_hourly * training_hours)
    }

# Example: LLM fine-tuning on 8x A100
result = estimate_spot_training_cost(
    base_on_demand_hourly=29.39,  # a2-highgpu-8g us-central1
    training_hours=24,
    spot_discount=0.67,
    preemption_overhead=1.08  # Good checkpoint strategy
)
# Expected cost: ~$233 (vs $706 on-demand, 67% savings)
```

### Multi-Region Cost Optimization

**Strategy: Use lowest-cost region for batch jobs:**

```bash
# Compare spot prices across regions
gcloud compute machine-types list \
  --filter="name:a2-highgpu-1g" \
  --format="table(name,zone,guestCpus,memoryMb)"

# Check current spot pricing (via pricing API)
gcloud compute machine-types describe a2-highgpu-1g \
  --zone=us-central1-a \
  --format="get(pricing.spot)"
```

**Regional selection criteria:**

1. **Spot availability** - Check capacity patterns (see 42-gcp-spot-availability.md)
2. **Data locality** - Egress costs if data in different region
3. **Latency requirements** - Not critical for batch training
4. **Compliance** - Data residency requirements
5. **Price differences** - Often 5-15% variation between regions

---

## Section 2: Hybrid Spot + On-Demand Architectures

### Architecture Pattern 1: Spot Primary, On-Demand Fallback

**Pattern:** Run training on spot, automatically failover to on-demand if preempted.

From [hybrid cloud architecture best practices](https://cast.ai/blog/spot-instances-vs-on-demand-automation/) (accessed 2025-01-31):

**Implementation:**

```python
# W&B Launch hybrid queue configuration
# File: .wandb/launch-config.yaml

queues:
  - name: training-spot-primary
    resource: gcp-spot
    max_jobs: 10

  - name: training-on-demand-fallback
    resource: gcp-on-demand
    max_jobs: 2  # Limit expensive resources

resources:
  gcp-spot:
    provider: gcp
    region: us-central1
    machine_type: a2-highgpu-8g
    accelerators:
      type: nvidia-tesla-a100
      count: 8
    preemptible: true  # Spot instance
    max_retries: 3
    fallback_resource: gcp-on-demand

  gcp-on-demand:
    provider: gcp
    region: us-central1
    machine_type: a2-highgpu-8g
    accelerators:
      type: nvidia-tesla-a100
      count: 8
    preemptible: false  # On-demand
```

**Failover logic:**

```python
# Automatic failover in training script
import os
import wandb

def train_with_spot_failover():
    """Training with automatic spot -> on-demand failover."""

    # Detect if running on spot instance
    is_spot = os.getenv("PREEMPTIBLE", "false") == "true"

    # Initialize W&B with resource tracking
    run = wandb.init(
        project="llm-training",
        config={
            "resource_type": "spot" if is_spot else "on-demand",
            "machine_type": os.getenv("MACHINE_TYPE"),
            "region": os.getenv("REGION"),
        }
    )

    try:
        # Main training loop with checkpointing
        train_model(checkpoint_every_n_steps=100)

    except PreemptionError:
        if is_spot:
            # Save checkpoint immediately
            save_checkpoint("preemption_checkpoint.pt")

            # W&B Launch will automatically retry on on-demand
            wandb.log({"event": "spot_preemption"})
            raise  # Re-raise to trigger W&B Launch failover
        else:
            # On on-demand - real failure
            raise

    finally:
        run.finish()
```

**Cost analysis:**

```python
# Expected cost with 90% spot success rate
spot_cost_per_hour = 9.70  # $9.70/hr for a2-highgpu-8g spot
on_demand_cost_per_hour = 29.39  # $29.39/hr on-demand

training_hours = 24
spot_success_rate = 0.90  # 90% complete on spot

spot_hours = training_hours * spot_success_rate
on_demand_hours = training_hours * (1 - spot_success_rate)

total_cost = (spot_hours * spot_cost_per_hour) +
             (on_demand_hours * on_demand_cost_per_hour)

# Result: ~$279 (vs $706 pure on-demand, 60% savings)
# vs ~$233 pure spot with restarts, 67% savings
```

**When to use:** Critical deadlines where 5-10% on-demand cost acceptable for guaranteed completion.

### Architecture Pattern 2: Spot for Experimentation, On-Demand for Production

**Pattern:** Development/ablations on spot, final runs on on-demand.

**Workflow:**

```yaml
# Development phase (spot)
experiments:
  - name: hyperparameter-search
    resource: spot
    runs: 50
    budget: $500

  - name: ablation-studies
    resource: spot
    runs: 20
    budget: $200

# Production phase (on-demand)
production:
  - name: final-training-run
    resource: on-demand
    runs: 1
    budget: $700
    priority: high
```

**Cost optimization:**

- **Development (95% of runs):** Spot instances, $700 → $231 (67% savings)
- **Production (5% of runs):** On-demand, $700 → $700 (0% savings, but guaranteed)
- **Total savings:** ~$469 savings across 71 runs (52% overall)

### Architecture Pattern 3: Critical Path On-Demand, Parallel Tasks Spot

**Pattern:** Main training on-demand, data preprocessing/evaluation on spot.

From [hybrid cloud computing patterns](https://www.confluent.io/learn/hybrid-cloud/) (accessed 2025-01-31):

**Pipeline architecture:**

```python
# Training pipeline with hybrid resources
class HybridTrainingPipeline:
    def __init__(self):
        self.on_demand_queue = Queue(resource="gcp-on-demand")
        self.spot_queue = Queue(resource="gcp-spot")

    def run_pipeline(self, dataset, model_config):
        # Data preprocessing on SPOT (fault-tolerant)
        preprocessing_job = self.spot_queue.submit(
            job=preprocess_data,
            args=(dataset,),
            checkpoint_every=1000,  # Frequent checkpoints
            max_retries=5
        )

        # Wait for preprocessing
        preprocessed_data = preprocessing_job.result()

        # Main training on ON-DEMAND (critical path)
        training_job = self.on_demand_queue.submit(
            job=train_model,
            args=(preprocessed_data, model_config),
            priority="high"
        )

        # Evaluation on SPOT (parallel, non-critical)
        eval_jobs = []
        for checkpoint in training_job.checkpoints():
            eval_job = self.spot_queue.submit(
                job=evaluate_checkpoint,
                args=(checkpoint,),
                max_retries=3
            )
            eval_jobs.append(eval_job)

        # Wait for training completion
        final_model = training_job.result()

        # Collect eval results (best-effort)
        eval_results = [job.result() for job in eval_jobs if job.completed()]

        return final_model, eval_results
```

**Cost breakdown example:**

```
Total pipeline:
- Preprocessing: 8 hours @ $0.35/hr spot = $2.80
- Training: 24 hours @ $29.39/hr on-demand = $705.36
- Evaluation: 12 hours @ $0.35/hr spot = $4.20
Total: $712.36

Pure on-demand equivalent:
- Preprocessing: 8 hours @ $2.50/hr = $20
- Training: 24 hours @ $29.39/hr = $705.36
- Evaluation: 12 hours @ $2.50/hr = $30
Total: $755.36

Savings: $43 (6% overall, 86% on non-critical tasks)
```

### Architecture Pattern 4: Queue-Based Hybrid Scheduling

**Pattern:** Intelligent queue that prioritizes spot, uses on-demand when needed.

```python
# Smart queue scheduler
class HybridQueueScheduler:
    def __init__(self, spot_budget_limit, on_demand_budget_limit):
        self.spot_budget = spot_budget_limit
        self.on_demand_budget = on_demand_budget_limit
        self.spot_spent = 0
        self.on_demand_spent = 0

    def schedule_job(self, job, urgency="normal"):
        """
        Schedule job on spot or on-demand based on:
        - Budget remaining
        - Job urgency
        - Spot availability
        """

        # Check spot budget
        if self.spot_spent < self.spot_budget:
            # Try spot first
            try:
                return self.run_on_spot(job)
            except NoSpotCapacity:
                # Fall through to on-demand
                pass

        # Use on-demand if:
        # - Spot budget exhausted
        # - High urgency
        # - No spot capacity
        if urgency == "high" or self.on_demand_spent < self.on_demand_budget:
            return self.run_on_demand(job)
        else:
            raise BudgetExceeded("Both spot and on-demand budgets exhausted")

    def run_on_spot(self, job):
        cost = job.estimate_cost(resource_type="spot")
        if self.spot_spent + cost > self.spot_budget:
            raise BudgetExceeded("Spot budget would be exceeded")

        result = submit_to_spot_queue(job)
        self.spot_spent += cost
        return result

    def run_on_demand(self, job):
        cost = job.estimate_cost(resource_type="on-demand")
        if self.on_demand_spent + cost > self.on_demand_budget:
            raise BudgetExceeded("On-demand budget would be exceeded")

        result = submit_to_on_demand_queue(job)
        self.on_demand_spent += cost
        return result

# Usage
scheduler = HybridQueueScheduler(
    spot_budget_limit=1000,
    on_demand_budget_limit=500
)

# Most jobs run on spot
for job in experiment_queue:
    scheduler.schedule_job(job, urgency="normal")

# Critical final run on on-demand
scheduler.schedule_job(final_training, urgency="high")
```

### Cost vs Reliability Tradeoff Analysis

**Decision matrix:**

| Workload Type | Recommendation | Expected Savings | Reliability |
|---------------|----------------|------------------|-------------|
| Hyperparameter search | 100% spot | 67% | 90-95% |
| Ablation studies | 100% spot | 67% | 90-95% |
| Development training | 100% spot | 67% | 90-95% |
| Production training | Spot primary, on-demand fallback | 60% | 99% |
| Critical deadline | 100% on-demand | 0% | 99.9% |
| Data preprocessing | 100% spot | 67% | 95% |
| Evaluation | 100% spot | 67% | 95% |

**Hybrid architecture ROI:**

```
Scenario: 100 training runs over 3 months
- 90 runs: Experimentation (spot) = $231 each = $20,790
- 10 runs: Production (on-demand) = $706 each = $7,060
Total: $27,850

Pure on-demand equivalent:
- 100 runs @ $706 = $70,600

Total savings: $42,750 (61% savings)
```

---

## Section 3: Cost Tracking and Attribution

### Cloud Billing Export to BigQuery

From [GCP Cloud Billing export documentation](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) (accessed 2025-01-31):

**Setup billing export:**

```bash
# 1. Create BigQuery dataset for billing data
bq mk --dataset \
  --location=US \
  --default_table_expiration 0 \
  my-project:billing_export

# 2. Enable billing export in Cloud Console
# Navigation: Billing > Billing Export > BigQuery Export
# - Select billing account
# - Choose dataset: billing_export
# - Enable detailed usage cost export
# - Enable pricing export (optional)
```

**BigQuery billing schema:**

```sql
-- Detailed billing data structure
SELECT
  billing_account_id,
  service.description AS service,
  sku.description AS sku,
  usage_start_time,
  usage_end_time,
  project.id AS project_id,
  project.name AS project_name,
  location.region,
  resource.name,
  labels,  -- KEY for cost attribution
  cost,
  currency,
  usage.amount,
  usage.unit,
  credits  -- Committed use discounts, sustained use discounts
FROM
  `my-project.billing_export.gcp_billing_export_v1_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20250101' AND '20250131'
LIMIT 10
```

### Label-Based Cost Attribution

**Label strategy for ML workloads:**

```python
# Apply labels when creating instances
from google.cloud import compute_v1

def create_spot_instance_with_labels(
    project_id, zone, instance_name, machine_type
):
    """Create spot instance with cost tracking labels."""

    instance_client = compute_v1.InstancesClient()

    # Define comprehensive labels
    labels = {
        "team": "ml-research",
        "project": "llm-training",
        "experiment": "gpt-ablation-001",
        "researcher": "alice",
        "cost-center": "r-and-d",
        "environment": "development",
        "resource-type": "spot",  # Track spot vs on-demand
        "framework": "pytorch",
        "wandb-run-id": "abc123xyz",  # Link to W&B run
    }

    # Instance configuration
    instance = compute_v1.Instance()
    instance.name = instance_name
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"
    instance.labels = labels
    instance.scheduling = compute_v1.Scheduling(
        preemptible=True,  # Spot instance
        on_host_maintenance="TERMINATE",
        automatic_restart=False,
    )

    # Create instance
    operation = instance_client.insert(
        project=project_id, zone=zone, instance_resource=instance
    )

    return operation.result()
```

**Query costs by labels:**

```sql
-- Cost by team
SELECT
  labels.value AS team,
  SUM(cost) AS total_cost,
  COUNT(DISTINCT resource.name) AS resource_count
FROM
  `my-project.billing_export.gcp_billing_export_v1_*`,
  UNNEST(labels) AS labels
WHERE
  _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', CURRENT_DATE())
  AND labels.key = 'team'
GROUP BY team
ORDER BY total_cost DESC;

-- Cost by experiment
SELECT
  labels.value AS experiment,
  service.description AS service,
  SUM(cost) AS total_cost,
  SUM(usage.amount) AS total_usage,
  usage.unit
FROM
  `my-project.billing_export.gcp_billing_export_v1_*`,
  UNNEST(labels) AS labels
WHERE
  _TABLE_SUFFIX BETWEEN '20250101' AND '20250131'
  AND labels.key = 'experiment'
GROUP BY experiment, service, usage.unit
ORDER BY total_cost DESC;

-- Spot vs on-demand cost comparison
SELECT
  labels.value AS resource_type,
  SUM(cost) AS total_cost,
  AVG(cost) AS avg_cost_per_resource
FROM
  `my-project.billing_export.gcp_billing_export_v1_*`,
  UNNEST(labels) AS labels
WHERE
  _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', CURRENT_DATE())
  AND labels.key = 'resource-type'
GROUP BY resource_type;
```

### Budget Alerts Configuration

From [GCP budget alerts documentation](https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications) (accessed 2025-01-31):

**Create budget with alerts:**

```bash
# Create budget via gcloud
gcloud billing budgets create \
  --billing-account=012345-6789AB-CDEF01 \
  --display-name="ML Training Spot Budget" \
  --budget-amount=5000 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100 \
  --notification-channels=projects/my-project/notificationChannels/12345

# Create budget filtered by labels
gcloud billing budgets create \
  --billing-account=012345-6789AB-CDEF01 \
  --display-name="LLM Training Experiment Budget" \
  --budget-amount=1000 \
  --filter-projects=my-project \
  --filter-labels=project=llm-training \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100
```

**Programmatic budget alerts:**

```python
# Monitor budget via Pub/Sub
from google.cloud import pubsub_v1

def handle_budget_alert(message):
    """Handle budget alert from Pub/Sub."""
    import json

    data = json.loads(message.data.decode('utf-8'))

    budget_name = data['budgetDisplayName']
    cost_amount = data['costAmount']
    budget_amount = data['budgetAmount']
    threshold_percent = data['alertThresholdExceeded']

    if threshold_percent >= 1.0:  # 100% threshold
        # Critical: Stop all non-essential spot instances
        print(f"CRITICAL: Budget {budget_name} exceeded!")
        stop_non_essential_instances()
        send_alert_email(
            subject=f"Budget {budget_name} EXCEEDED",
            body=f"Cost: ${cost_amount:.2f} / ${budget_amount:.2f}"
        )

    elif threshold_percent >= 0.9:  # 90% threshold
        # Warning: Prepare to scale down
        print(f"WARNING: Budget {budget_name} at 90%")
        send_alert_slack(
            channel="#ml-ops",
            message=f"Budget warning: {budget_name} at 90%"
        )

    message.ack()

# Subscribe to budget alerts
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(
    'my-project', 'budget-alerts-sub'
)
subscriber.subscribe(subscription_path, callback=handle_budget_alert)
```

### Custom Cost Dashboards

**Create Data Studio dashboard from BigQuery:**

```sql
-- View for cost dashboard
CREATE OR REPLACE VIEW `my-project.billing_export.daily_ml_costs` AS
SELECT
  DATE(usage_start_time) AS date,
  service.description AS service,
  CASE
    WHEN sku.description LIKE '%Spot%' THEN 'Spot'
    WHEN sku.description LIKE '%Preemptible%' THEN 'Spot'
    ELSE 'On-Demand'
  END AS resource_type,
  location.region,
  (SELECT value FROM UNNEST(labels) WHERE key = 'team') AS team,
  (SELECT value FROM UNNEST(labels) WHERE key = 'project') AS project,
  (SELECT value FROM UNNEST(labels) WHERE key = 'experiment') AS experiment,
  SUM(cost) AS daily_cost,
  SUM(cost) - SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) AS net_cost
FROM
  `my-project.billing_export.gcp_billing_export_v1_*`
WHERE
  _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
                   AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
GROUP BY date, service, resource_type, region, team, project, experiment;
```

### W&B Cost Tracking Integration

**Link GCP costs to W&B runs:**

```python
import wandb
from google.cloud import bigquery

def log_run_cost_to_wandb(run_id, gcp_labels):
    """Query BigQuery for run cost and log to W&B."""

    client = bigquery.Client()

    # Query cost for specific run
    query = f"""
    SELECT
      SUM(cost) AS total_cost,
      SUM(usage.amount) AS total_gpu_hours
    FROM
      `my-project.billing_export.gcp_billing_export_v1_*`,
      UNNEST(labels) AS labels
    WHERE
      labels.key = 'wandb-run-id'
      AND labels.value = '{run_id}'
    """

    result = client.query(query).result()
    row = next(result)

    # Log to W&B
    api = wandb.Api()
    run = api.run(f"my-team/my-project/{run_id}")
    run.summary["gcp_cost_total"] = row.total_cost
    run.summary["gcp_gpu_hours"] = row.total_gpu_hours
    run.summary["gcp_cost_per_gpu_hour"] = (
        row.total_cost / row.total_gpu_hours if row.total_gpu_hours > 0 else 0
    )
    run.update()

# Run after training completes (wait for billing data)
# Billing data has 24-48 hour delay
log_run_cost_to_wandb("abc123xyz", {"wandb-run-id": "abc123xyz"})
```

### Anomaly Detection for Cost Spikes

**Automated cost anomaly detection:**

```python
# Cloud Function triggered by budget alert
def detect_cost_anomalies(request):
    """Detect unusual cost patterns."""
    from google.cloud import bigquery

    client = bigquery.Client()

    # Get daily costs for last 30 days
    query = """
    SELECT
      DATE(usage_start_time) AS date,
      SUM(cost) AS daily_cost
    FROM
      `my-project.billing_export.gcp_billing_export_v1_*`
    WHERE
      _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
                       AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
    GROUP BY date
    ORDER BY date
    """

    results = client.query(query).result()
    daily_costs = [row.daily_cost for row in results]

    # Simple anomaly detection: 2x standard deviation
    import statistics
    mean_cost = statistics.mean(daily_costs)
    stdev_cost = statistics.stdev(daily_costs)

    latest_cost = daily_costs[-1]
    if latest_cost > mean_cost + (2 * stdev_cost):
        # Anomaly detected
        send_alert(
            message=f"Cost anomaly: ${latest_cost:.2f} (mean: ${mean_cost:.2f}, stdev: ${stdev_cost:.2f})",
            severity="HIGH"
        )

        # Analyze what caused spike
        analyze_cost_spike()
```

### Complete Cost Monitoring Setup

**End-to-end monitoring implementation:**

```python
# comprehensive_cost_monitoring.py
from google.cloud import bigquery, monitoring_v3
import wandb

class CostMonitoringSystem:
    def __init__(self, project_id, billing_dataset):
        self.project_id = project_id
        self.billing_dataset = billing_dataset
        self.bq_client = bigquery.Client()
        self.monitoring_client = monitoring_v3.MetricServiceClient()

    def get_current_month_cost(self, labels=None):
        """Get month-to-date cost with optional label filters."""

        label_filter = ""
        if labels:
            conditions = [
                f"labels.key = '{k}' AND labels.value = '{v}'"
                for k, v in labels.items()
            ]
            label_filter = f"AND ({' AND '.join(conditions)})"

        query = f"""
        SELECT
          SUM(cost) AS total_cost
        FROM
          `{self.project_id}.{self.billing_dataset}.gcp_billing_export_v1_*`,
          UNNEST(labels) AS labels
        WHERE
          _TABLE_SUFFIX >= FORMAT_DATE('%Y%m01', CURRENT_DATE())
          {label_filter}
        """

        result = self.bq_client.query(query).result()
        return next(result).total_cost

    def get_cost_by_experiment(self, start_date, end_date):
        """Get cost breakdown by experiment."""

        query = f"""
        SELECT
          experiment_label.value AS experiment,
          resource_type_label.value AS resource_type,
          SUM(cost) AS total_cost
        FROM
          `{self.project_id}.{self.billing_dataset}.gcp_billing_export_v1_*`,
          UNNEST(labels) AS experiment_label,
          UNNEST(labels) AS resource_type_label
        WHERE
          _TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'
          AND experiment_label.key = 'experiment'
          AND resource_type_label.key = 'resource-type'
        GROUP BY experiment, resource_type
        ORDER BY total_cost DESC
        """

        return self.bq_client.query(query).result()

    def calculate_spot_savings(self):
        """Calculate actual savings from using spot instances."""

        query = f"""
        WITH spot_costs AS (
          SELECT SUM(cost) AS spot_total
          FROM `{self.project_id}.{self.billing_dataset}.gcp_billing_export_v1_*`,
               UNNEST(labels) AS labels
          WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m01', CURRENT_DATE())
            AND labels.key = 'resource-type'
            AND labels.value = 'spot'
        ),
        on_demand_costs AS (
          SELECT SUM(cost) AS on_demand_total
          FROM `{self.project_id}.{self.billing_dataset}.gcp_billing_export_v1_*`,
               UNNEST(labels) AS labels
          WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m01', CURRENT_DATE())
            AND labels.key = 'resource-type'
            AND labels.value = 'on-demand'
        )
        SELECT
          spot_total,
          on_demand_total,
          -- Estimate equivalent on-demand cost (spot / 0.33 for 67% discount)
          spot_total / 0.33 AS equivalent_on_demand,
          (spot_total / 0.33) - spot_total AS estimated_savings
        FROM spot_costs, on_demand_costs
        """

        result = next(self.bq_client.query(query).result())
        return {
            "spot_total": result.spot_total,
            "on_demand_total": result.on_demand_total,
            "equivalent_on_demand": result.equivalent_on_demand,
            "estimated_savings": result.estimated_savings
        }

# Usage
monitor = CostMonitoringSystem("my-project", "billing_export")

# Check current month spend
current_cost = monitor.get_current_month_cost(
    labels={"team": "ml-research", "project": "llm-training"}
)
print(f"Current month cost: ${current_cost:.2f}")

# Calculate spot savings
savings = monitor.calculate_spot_savings()
print(f"Estimated savings from spot: ${savings['estimated_savings']:.2f}")

# Log to W&B
wandb.log({
    "gcp_cost_mtd": current_cost,
    "gcp_spot_savings": savings['estimated_savings']
})
```

---

## Sources

**GCP Documentation:**
- [Spot VMs pricing](https://cloud.google.com/spot-vms/pricing) - GCP spot pricing model (accessed 2025-01-31)
- [Export Cloud Billing data to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) - Billing export setup (accessed 2025-01-31)
- [Set up programmatic notifications](https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications) - Budget alerts (accessed 2025-01-31)
- [Analyze billing data with Reports](https://cloud.google.com/billing/docs/how-to/reports) - Cost analysis tools (accessed 2025-01-31)

**Web Research:**
- [Spot Instances vs On-Demand: Save with Smart Automation](https://cast.ai/blog/spot-instances-vs-on-demand-automation/) - Cast AI (accessed 2025-01-31)
- [Hybrid Cloud Computing: Benefits, Architecture](https://www.confluent.io/learn/hybrid-cloud/) - Confluent (accessed 2025-01-31)
- [Cost Optimization Best Practices 2024 2025](https://aws.plainenglish.io/top-10-aws-cost-optimization-strategies-a59808c9f6e0) - AWS in Plain English (accessed 2025-01-31)
- [GCP Cost Management Tools](https://www.cloudbolt.io/gcp-cost-optimization/gcp-cost-management/) - CloudBolt (accessed 2025-01-31)

**Additional References:**
- [An Expert Guide On GCP Cost Monitoring](https://www.cloudzero.com/blog/gcp-cost-monitoring/) - CloudZero (accessed 2025-01-31)
- [GCP Budget Alerts: Setup and Best Practices](https://hokstadconsulting.com/blog/gcp-budget-alerts-setup-and-best-practices) - Hokstad Consulting (accessed 2025-01-31)
