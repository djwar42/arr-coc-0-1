# GCP Billing Export to BigQuery and Cost Analysis

**Purpose**: Export detailed billing data to BigQuery for analysis, cost optimization, and budget management.

**Last Updated**: 2025-02-03

---

## Section 1: Billing Export to BigQuery Overview (~100 lines)

### What is Cloud Billing Export?

Cloud Billing export to BigQuery automatically exports detailed Google Cloud billing data to BigQuery datasets for analysis and reporting. This includes:

- **Usage costs**: Per-SKU resource consumption
- **Cost estimates**: Projected spending
- **Pricing data**: Unit prices and discounts
- **Credits and adjustments**: Promotional credits, committed use discounts
- **Taxes**: Regional tax calculations

From [Export Cloud Billing data to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) (accessed 2025-02-03):
- Exports update throughout the day as usage occurs
- Data typically available within 24 hours
- Historical data backfilled when export is first enabled

### Two Export Types

**1. Standard Usage Cost Export**
- High-level summary of total spending
- Aggregated by project, SKU, and time period
- Suitable for basic cost tracking and budgeting
- Smaller dataset size

From [Structure of Standard data export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/standard-usage) (accessed 2025-02-03):
- Contains essential billing fields: cost, usage amount, credits
- Updated daily with previous day's charges
- Good for executive dashboards and monthly reporting

**2. Detailed Usage Cost Export**
- Granular, SKU-level usage data
- Includes resource labels, locations, and system tags
- Detailed breakdown of every charge
- Larger dataset but more powerful for analysis

From [Structure of Detailed data export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) (accessed 2025-02-03):
- Contains 50+ fields including resource tags, labels, locations
- Near real-time updates (multiple times per day)
- Essential for chargeback, cost allocation, and optimization

### Key Use Cases

**Cost Analysis**:
- Identify top spending projects and services
- Track cost trends over time
- Compare actual vs. budgeted spending

From [Visualize GCP Billing using BigQuery and Data Studio](https://medium.com/google-cloud/visualize-gcp-billing-using-bigquery-and-data-studio-d3e695f90c08) (accessed 2025-02-03):
- Create custom dashboards in Looker Studio (formerly Data Studio)
- Build cost forecasting models
- Detect anomalous spending patterns

**Cost Attribution**:
- Chargeback to teams using labels
- Department-level cost tracking
- Environment-based allocation (dev/staging/prod)

**Optimization**:
- Find idle or underutilized resources
- Identify commitment discount opportunities
- Analyze cost per service/feature

### Prerequisites

Before enabling billing export:

1. **Billing Account Access**: Billing Account Administrator or Billing Account User role
2. **Project with BigQuery**: Project where dataset will be created
3. **Enable APIs**: BigQuery API must be enabled
4. **Dataset Creation**: Create a BigQuery dataset to hold billing data

From [Set up Cloud Billing data export to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-setup) (accessed 2025-02-03):
- Recommended: Create dedicated "Billing Administration" project
- Use separate dataset for billing data (e.g., `billing_export`)
- Set appropriate IAM permissions on dataset

---

## Section 2: Setting Up Billing Export (~150 lines)

### Step 1: Create BigQuery Dataset

**Via Console**:
```
1. Navigate to BigQuery in Cloud Console
2. Select project for billing data
3. Click "Create Dataset"
4. Dataset ID: billing_export
5. Data location: US (or your preferred multi-region)
6. Default table expiration: None (preserve billing history)
7. Click "Create Dataset"
```

**Via gcloud**:
```bash
# Create dataset for billing export
bq mk --dataset \
  --location=US \
  --description="Cloud Billing Export" \
  PROJECT_ID:billing_export

# Verify dataset created
bq ls --project_id=PROJECT_ID
```

From [Cloud Billing interactive tutorials](https://docs.cloud.google.com/billing/docs/interactive-tutorials) (accessed 2025-02-03):
- Dataset location affects BigQuery query costs
- Choose location closest to your primary analysis region
- US multi-region recommended for global organizations

### Step 2: Enable Billing Export

**Via Console**:
```
1. Navigate to Billing → Billing Export
2. Click "BigQuery Export" tab
3. Select export type:
   - Standard usage cost (recommended for most users)
   - Detailed usage cost (for advanced analysis)
4. Choose project: Select project with dataset
5. Choose dataset: Select billing_export dataset
6. Click "Enable"
```

From [Export Cloud Billing data to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) (accessed 2025-02-03):
- Can enable both Standard AND Detailed exports simultaneously
- Data export begins immediately
- Historical backfill typically completes within 48 hours

**Via gcloud** (currently not supported for initial setup):
```bash
# Note: Must use Console or API for first-time setup
# After setup, can query and manage via gcloud/bq CLI
```

### Step 3: Grant Permissions

**Required IAM Roles**:

For Cloud Billing account to write to BigQuery:
```bash
# Grant BigQuery Data Editor to Cloud Billing service account
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:cloud-billing-export@system.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

For users to query billing data:
```bash
# Grant BigQuery User role (query only)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:analyst@example.com" \
  --role="roles/bigquery.user"

# Grant BigQuery Data Viewer (read tables)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:analyst@example.com" \
  --role="roles/bigquery.dataViewer"
```

From [Set up Cloud Billing data export to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-setup) (accessed 2025-02-03):
- Service account permissions handled automatically via Console setup
- User permissions must be granted separately
- Consider using groups for team access

### Step 4: Verify Export

**Check table creation**:
```bash
# List tables in billing dataset
bq ls PROJECT_ID:billing_export

# Expected output:
# gcp_billing_export_YYYY_MM  (Standard export, monthly tables)
# gcp_billing_export_v1_XXXXXX (Detailed export, single table)
```

**Query sample data**:
```sql
-- Standard export
SELECT
  billing_account_id,
  project.id as project_id,
  service.description as service,
  SUM(cost) as total_cost
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN '2025_01' AND '2025_02'
GROUP BY 1, 2, 3
ORDER BY total_cost DESC
LIMIT 10;
```

From [Example queries for Cloud Billing data export](https://docs.cloud.google.com/billing/docs/how-to/bq-examples) (accessed 2025-02-03):
- Standard export creates new table each month
- Detailed export uses partitioned table for all data
- Use `_TABLE_SUFFIX` wildcard for multi-month queries

### Export Schema

**Standard Export Key Fields**:
```
billing_account_id       STRING      Billing account identifier
service.id               STRING      Service ID (e.g., Compute Engine)
service.description      STRING      Human-readable service name
sku.id                   STRING      Stock keeping unit ID
sku.description          STRING      SKU description
usage_start_time         TIMESTAMP   Start of usage period
usage_end_time           TIMESTAMP   End of usage period
project.id               STRING      Project ID
project.name             STRING      Project name
location.location        STRING      Resource location (e.g., us-central1)
location.region          STRING      Region (e.g., us-central1)
cost                     FLOAT       Total cost in billing currency
currency                 STRING      Billing currency (e.g., USD)
usage.amount             FLOAT       Usage quantity
usage.unit               STRING      Usage unit (e.g., byte-seconds)
credits                  ARRAY       Array of credits applied
```

**Detailed Export Additional Fields**:
```
labels                   ARRAY       Custom resource labels
system_labels            ARRAY       GCP system labels
tags                     ARRAY       Network tags
cost_type                STRING      REGULAR, TAX, ADJUSTMENT, ROUNDING_ERROR
resource.name            STRING      Specific resource identifier
invoice.month            STRING      Invoice month (YYYYMM)
```

From [Structure of Detailed data export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) (accessed 2025-02-03):
- Detailed schema contains 50+ fields
- Labels and tags enable cost allocation by team/environment
- Cost type distinguishes between usage, taxes, and adjustments

---

## Section 3: Cost Analysis Queries (~150 lines)

### Basic Cost Queries

**Total cost by project**:
```sql
SELECT
  project.id,
  project.name,
  SUM(cost) + SUM(IFNULL((SELECT SUM(amount) FROM UNNEST(credits)), 0)) as total_cost,
  COUNT(DISTINCT DATE(usage_start_time)) as days_of_usage
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m', DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
  AND FORMAT_DATE('%Y%m', CURRENT_DATE())
GROUP BY 1, 2
ORDER BY total_cost DESC;
```

From [Example queries for Cloud Billing data export](https://docs.cloud.google.com/billing/docs/how-to/bq-examples) (accessed 2025-02-03):
- Always include credits in cost calculations
- Use `_TABLE_SUFFIX` for date range filtering on Standard export
- `UNNEST(credits)` extracts credit amounts from array

**Top 10 most expensive services**:
```sql
SELECT
  service.description as service,
  SUM(cost) as total_cost,
  SUM(usage.amount) as total_usage,
  usage.unit
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y_%m', CURRENT_DATE())
  AND cost > 0
GROUP BY 1, 4
ORDER BY total_cost DESC
LIMIT 10;
```

**Daily spending trend**:
```sql
SELECT
  DATE(usage_start_time) as usage_date,
  SUM(cost) as daily_cost,
  ROUND(AVG(SUM(cost)) OVER (ORDER BY DATE(usage_start_time) ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) as seven_day_avg
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m', DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH))
  AND FORMAT_DATE('%Y%m', CURRENT_DATE())
GROUP BY 1
ORDER BY 1 DESC;
```

From [Visualize GCP Billing using BigQuery and Data Studio](https://medium.com/google-cloud/visualize-gcp-billing-using-bigquery-and-data-studio-d3e695f90c08) (accessed 2025-02-03):
- Rolling averages smooth out daily variations
- Identify spending spikes and trends
- Use for anomaly detection

### Advanced Cost Optimization Queries

**Identify idle Compute Engine instances**:
```sql
-- Instances with low CPU usage in billing data
SELECT
  project.id,
  sku.description,
  resource.name,
  location.region,
  SUM(cost) as total_cost,
  COUNT(DISTINCT DATE(usage_start_time)) as days_running
FROM `PROJECT_ID.billing_export.gcp_billing_export_v1_*`
WHERE service.description = 'Compute Engine'
  AND sku.description LIKE '%Instance Core%'
  AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY 1, 2, 3, 4
HAVING total_cost > 100
ORDER BY total_cost DESC;
```

**Committed use discount potential**:
```sql
-- Find steady resource usage that could benefit from commitments
SELECT
  service.description,
  sku.description,
  location.region,
  AVG(usage.amount) as avg_daily_usage,
  MIN(usage.amount) as min_daily_usage,
  MAX(usage.amount) as max_daily_usage,
  SUM(cost) as total_cost
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m', DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH))
  AND FORMAT_DATE('%Y%m', CURRENT_DATE())
  AND service.description IN ('Compute Engine', 'Cloud SQL', 'Cloud Spanner')
GROUP BY 1, 2, 3
HAVING min_daily_usage > 0  -- Consistent usage
  AND (max_daily_usage / min_daily_usage) < 2  -- Low variability
  AND total_cost > 1000
ORDER BY total_cost DESC;
```

From [GCP BigQuery pricing guide and cost optimization tips](https://holori.com/gcp-bigquery-pricing-guide-and-cost-optimization-tips/) (accessed 2025-02-03):
- Committed use discounts offer 37-55% savings
- Requires consistent baseline usage
- Analyze usage patterns before committing

**Cost by label (chargeback)**:
```sql
-- Detailed export required for labels
SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'team') as team,
  (SELECT value FROM UNNEST(labels) WHERE key = 'environment') as environment,
  service.description,
  SUM(cost) as total_cost
FROM `PROJECT_ID.billing_export.gcp_billing_export_v1_*`
WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  AND EXISTS (SELECT 1 FROM UNNEST(labels) WHERE key = 'team')
GROUP BY 1, 2, 3
ORDER BY total_cost DESC;
```

**Month-over-month cost comparison**:
```sql
WITH monthly_costs AS (
  SELECT
    FORMAT_DATE('%Y-%m', DATE(usage_start_time)) as month,
    project.id,
    service.description,
    SUM(cost) as monthly_cost
  FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
  WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m', DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH))
    AND FORMAT_DATE('%Y%m', CURRENT_DATE())
  GROUP BY 1, 2, 3
)
SELECT
  curr.project_id,
  curr.service_description,
  curr.monthly_cost as current_month_cost,
  prev.monthly_cost as previous_month_cost,
  curr.monthly_cost - prev.monthly_cost as cost_change,
  ROUND(((curr.monthly_cost - prev.monthly_cost) / prev.monthly_cost) * 100, 2) as percent_change
FROM monthly_costs curr
LEFT JOIN monthly_costs prev
  ON curr.project_id = prev.project_id
  AND curr.service_description = prev.service_description
  AND curr.month = FORMAT_DATE('%Y-%m', DATE_ADD(PARSE_DATE('%Y-%m', prev.month), INTERVAL 1 MONTH))
WHERE curr.month = FORMAT_DATE('%Y-%m', CURRENT_DATE())
  AND prev.monthly_cost > 0
ORDER BY ABS(cost_change) DESC
LIMIT 20;
```

From [Example queries for Cloud Billing data export](https://docs.cloud.google.com/billing/docs/how-to/bq-examples) (accessed 2025-02-03):
- Self-joins enable period-over-period comparisons
- Identify cost spikes requiring investigation
- Track cost reduction initiatives

### Query Cost Optimization

**BigQuery analysis costs**:
- On-demand: $6.25 per TB of data scanned
- Flat-rate: $2,000/month for 100 slots (predictable cost)

From [BigQuery Pricing Overview](https://cloud.google.com/bigquery/pricing) (accessed 2025-02-03):
- Partition and cluster billing export tables
- Use `_TABLE_SUFFIX` or `_PARTITIONTIME` filters
- Select only needed columns (avoid `SELECT *`)

**Optimize billing queries**:
```sql
-- BAD: Scans all data (expensive)
SELECT *
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE cost > 100;

-- GOOD: Partition filter + column selection
SELECT
  project.id,
  service.description,
  cost,
  usage_start_time
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y_%m', CURRENT_DATE())
  AND cost > 100;
```

---

## Section 4: Budget Alerts and Automation (~150 lines)

### Cloud Billing Budgets

Budgets trigger alerts when spending exceeds thresholds. Create via Console, gcloud, or API.

**Budget alert types**:
- **Email alerts**: Notify users at spending thresholds
- **Pub/Sub notifications**: Programmatic responses to budget events
- **Budget API**: Query budget status programmatically

From [Create, edit, or delete budgets and budget alerts](https://cloud.google.com/billing/docs/how-to/budgets) (accessed 2025-02-03):
- Set percentage-based thresholds (50%, 90%, 100%)
- Choose actual cost or forecasted cost alerts
- Configure monthly, quarterly, or custom time periods

### Creating Budgets via Console

**Steps**:
```
1. Navigate to Billing → Budgets & Alerts
2. Click "Create Budget"
3. Select scope:
   - Entire billing account
   - Specific projects
   - Specific services
4. Set amount:
   - Specified amount: $10,000/month
   - Last month's spend
   - Custom based on metric
5. Configure alert thresholds:
   - 50% (Alert 1)
   - 90% (Alert 2)
   - 100% (Alert 3)
   - 110% (Optional overage alert)
6. Add notification recipients
7. Optional: Connect Pub/Sub topic
8. Click "Finish"
```

From [Set up programmatic notifications](https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications) (accessed 2025-02-03):
- Pub/Sub notifications enable automated responses
- Receive JSON payload with budget status
- Build custom cost control automation

### Creating Budgets via API

**Using gcloud (Cloud Billing Budget API)**:
```bash
# Install beta commands
gcloud components install beta

# Create budget
gcloud beta billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Monthly Production Budget" \
  --budget-amount=10000 \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --all-projects-scope
```

From [Get started with the Cloud Billing Budget API](https://docs.cloud.google.com/billing/docs/how-to/budget-api-overview) (accessed 2025-02-03):
- API enables budget management at scale
- Supports filters by project, service, label
- Automate budget creation for new projects

**Budget with Pub/Sub notification**:
```bash
# Create Pub/Sub topic
gcloud pubsub topics create budget-alerts

# Create budget with Pub/Sub
gcloud beta billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Production Alert Budget" \
  --budget-amount=10000 \
  --threshold-rule=percent=1.0 \
  --pubsub-topic=projects/PROJECT_ID/topics/budget-alerts
```

**Budget JSON schema**:
```json
{
  "displayName": "ML Training Budget",
  "budgetFilter": {
    "projects": ["projects/my-ml-project"],
    "services": ["services/95FF-2EF5-5EA1"],  // Compute Engine
    "labels": {
      "env": "production"
    }
  },
  "amount": {
    "specifiedAmount": {
      "currencyCode": "USD",
      "units": "5000"
    }
  },
  "thresholdRules": [
    {"thresholdPercent": 0.5},
    {"thresholdPercent": 0.9},
    {"thresholdPercent": 1.0}
  ],
  "allUpdatesRule": {
    "pubsubTopic": "projects/PROJECT_ID/topics/budget-alerts"
  }
}
```

### Automated Budget Responses

**Cloud Function to disable billing** (use with caution):

From [Automating Budget Alerts with Cloud Functions](https://wqwq3215.medium.com/automating-budget-alerts-with-cloud-functions-ac8cb207ef4b) (accessed 2025-02-03):
- Pub/Sub triggers Cloud Function on budget alert
- Function can disable billing, shut down resources, send Slack alerts
- Requires careful IAM permissions

```python
# Cloud Function triggered by Pub/Sub budget alert
import base64
import json
from google.cloud import billing_v1

def budget_alert(event, context):
    """Disable billing when budget exceeded"""
    pubsub_data = base64.b64decode(event['data']).decode('utf-8')
    budget_notification = json.loads(pubsub_data)

    cost_amount = budget_notification['costAmount']
    budget_amount = budget_notification['budgetAmount']

    if cost_amount >= budget_amount:
        project_id = budget_notification['projectId']
        disable_billing_for_project(project_id)
        send_alert_to_slack(f"CRITICAL: Billing disabled for {project_id}")

def disable_billing_for_project(project_id):
    """Remove billing account from project"""
    client = billing_v1.CloudBillingClient()
    name = f"projects/{project_id}"

    project_billing_info = billing_v1.ProjectBillingInfo()
    project_billing_info.name = name
    project_billing_info.billing_account_name = ""  # Remove billing account

    client.update_project_billing_info(name=name, project_billing_info=project_billing_info)
```

**Warning**: Disabling billing stops ALL services immediately. Consider alternatives:
- Send escalating alerts to teams
- Shut down specific resources (non-production VMs)
- Auto-scale down to minimum capacity
- Require manual approval before disabling billing

From [How to Avoid an Unexpected Cloud Bill — Fully Automated](https://medium.com/google-cloud/how-to-avoid-a-massive-cloud-bill-41a76251caba) (accessed 2025-02-03):
- Use budget alerts as early warning system
- Implement multi-tier response (alert → warn → act)
- Test automation in non-production environments
- Document emergency procedures

### Slack/Email Budget Alerts

**Cloud Function for Slack notifications**:
```python
import requests
import json
import base64

def budget_to_slack(event, context):
    """Send budget alert to Slack channel"""
    pubsub_data = base64.b64decode(event['data']).decode('utf-8')
    budget_notification = json.loads(pubsub_data)

    budget_name = budget_notification['budgetDisplayName']
    cost_amount = budget_notification['costAmount']
    budget_amount = budget_notification['budgetAmount']
    alert_threshold = budget_notification['alertThresholdExceeded']

    percentage = (cost_amount / budget_amount) * 100

    slack_message = {
        "text": f"⚠️ Budget Alert: {budget_name}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Budget Alert*: {budget_name}\n"
                           f"*Current Spend*: ${cost_amount:,.2f}\n"
                           f"*Budget*: ${budget_amount:,.2f}\n"
                           f"*Percentage*: {percentage:.1f}%\n"
                           f"*Threshold*: {alert_threshold * 100}%"
                }
            }
        ]
    }

    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    requests.post(webhook_url, json=slack_message)
```

**Deploy Cloud Function**:
```bash
# Deploy function
gcloud functions deploy budget-slack-alert \
  --runtime=python39 \
  --trigger-topic=budget-alerts \
  --entry-point=budget_to_slack \
  --region=us-central1

# Grant permissions
gcloud functions add-iam-policy-binding budget-slack-alert \
  --member="serviceAccount:budget-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker" \
  --region=us-central1
```

---

## Section 5: Cost Optimization Strategies (~50 lines)

### Committed Use Discounts (CUDs)

**Resource-based CUDs** (Compute Engine, Cloud SQL):
- 1-year: 37% discount
- 3-year: 55% discount
- Regional or global scope

From [Understanding and analyzing your costs with Google Cloud Billing reports](https://cloud.google.com/blog/products/gcp/monitor-and-manage-your-costs-with) (accessed 2025-02-03):
- Analyze steady-state workloads in billing export
- Identify minimum baseline usage
- Purchase commitments at 70-80% of baseline

**Spend-based CUDs**:
- Available for BigQuery, Cloud Run, other services
- Commit to minimum spend level
- Discounts increase with spend level

**Query to identify CUD candidates**:
```sql
-- Find consistent Compute Engine usage
SELECT
  DATE_TRUNC(DATE(usage_start_time), MONTH) as month,
  location.region,
  sku.description,
  MIN(SUM(usage.amount)) OVER (PARTITION BY location.region, sku.description) as min_monthly_usage,
  AVG(SUM(usage.amount)) OVER (PARTITION BY location.region, sku.description) as avg_monthly_usage,
  SUM(cost) as total_cost
FROM `PROJECT_ID.billing_export.gcp_billing_export_*`
WHERE service.description = 'Compute Engine'
  AND sku.description LIKE '%Core%'
  AND _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m', DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH))
    AND FORMAT_DATE('%Y%m', CURRENT_DATE())
GROUP BY 1, 2, 3
HAVING min_monthly_usage > 0
ORDER BY total_cost DESC;
```

### Right-sizing Recommendations

Identify oversized resources from billing patterns:

```sql
-- Find large instance costs with low utilization signals
SELECT
  project.id,
  resource.name,
  sku.description,
  SUM(cost) as total_cost,
  SUM(usage.amount) as total_usage
FROM `PROJECT_ID.billing_export.gcp_billing_export_v1_*`
WHERE service.description = 'Compute Engine'
  AND sku.description LIKE '%n2-standard-16%'  -- Large instances
  AND DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY 1, 2, 3
ORDER BY total_cost DESC;
```

**Action**: Cross-reference with Cloud Monitoring CPU/memory metrics to confirm underutilization.

### Idle Resource Detection

From [GCP BigQuery pricing guide and cost optimization tips](https://holori.com/gcp-bigquery-pricing-guide-and-cost-optimization-tips/) (accessed 2025-02-03):
- Persistent disks with no attached VM
- Cloud SQL instances with zero connections
- Load balancers with no backend instances

```sql
-- Find persistent disk costs without VM attachments
SELECT
  project.id,
  sku.description,
  SUM(cost) as total_cost,
  COUNT(DISTINCT resource.name) as disk_count
FROM `PROJECT_ID.billing_export.gcp_billing_export_v1_*`
WHERE service.description = 'Compute Engine'
  AND sku.description LIKE '%Storage PD%'
  AND DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY 1, 2
HAVING total_cost > 50
ORDER BY total_cost DESC;
```

**Action**: Identify disks, check attachment status, delete if truly idle.

---

## Sources

**Official Google Cloud Documentation**:
- [Export Cloud Billing data to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery) (accessed 2025-02-03)
- [Set up Cloud Billing data export to BigQuery](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-setup) (accessed 2025-02-03)
- [Example queries for Cloud Billing data export](https://docs.cloud.google.com/billing/docs/how-to/bq-examples) (accessed 2025-02-03)
- [Structure of Standard data export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/standard-usage) (accessed 2025-02-03)
- [Structure of Detailed data export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) (accessed 2025-02-03)
- [Create, edit, or delete budgets and budget alerts](https://cloud.google.com/billing/docs/how-to/budgets) (accessed 2025-02-03)
- [Set up programmatic notifications](https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications) (accessed 2025-02-03)
- [Get started with the Cloud Billing Budget API](https://docs.cloud.google.com/billing/docs/how-to/budget-api-overview) (accessed 2025-02-03)
- [BigQuery Pricing Overview](https://cloud.google.com/bigquery/pricing) (accessed 2025-02-03)
- [Cloud Billing interactive tutorials](https://docs.cloud.google.com/billing/docs/interactive-tutorials) (accessed 2025-02-03)

**Web Research**:
- [Visualize GCP Billing using BigQuery and Data Studio](https://medium.com/google-cloud/visualize-gcp-billing-using-bigquery-and-data-studio-d3e695f90c08) - Medium article by Mike Zinni (accessed 2025-02-03)
- [Automating Budget Alerts with Cloud Functions](https://wqwq3215.medium.com/automating-budget-alerts-with-cloud-functions-ac8cb207ef4b) - Medium article by wqwq (accessed 2025-02-03)
- [How to Avoid an Unexpected Cloud Bill — Fully Automated](https://medium.com/google-cloud/how-to-avoid-a-massive-cloud-bill-41a76251caba) - Medium article by Darren Lester (Dazbo) (accessed 2025-02-03)
- [GCP BigQuery pricing guide and cost optimization tips](https://holori.com/gcp-bigquery-pricing-guide-and-cost-optimization-tips/) - Holori blog (accessed 2025-02-03)
- [Understanding and analyzing your costs with Google Cloud Billing reports](https://cloud.google.com/blog/products/gcp/monitor-and-manage-your-costs-with) - Google Cloud blog (accessed 2025-02-03)

**Additional References**:
- [Exporting Your Google Cloud Billing Data to BigQuery](https://medium.com/google-cloud/exporting-your-google-cloud-billing-data-to-bigquery-296cae9a07f2) - Medium article by Aryan Irani (accessed 2025-02-03)
- [How to set up and use Google Cloud budget alerts](https://support.terra.bio/hc/en-us/articles/360057589931-How-to-set-up-and-use-Google-Cloud-budget-alerts) - Terra.bio support docs (accessed 2025-02-03)
