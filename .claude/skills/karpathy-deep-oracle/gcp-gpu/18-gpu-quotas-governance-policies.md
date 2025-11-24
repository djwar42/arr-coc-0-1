# GPU Resource Quotas & Governance Policies

Comprehensive guide to managing GPU resource quotas, implementing governance policies, cost allocation, and approval workflows for Google Cloud Platform. Covers organization policies, chargeback mechanisms, and resource management best practices for multi-team GPU infrastructure.

---

## Overview

GPU resources on GCP are expensive and scarce. Without proper governance and quota management, teams can experience resource conflicts, cost overruns, and compliance violations. This guide covers the complete lifecycle of GPU quota management—from organization policy constraints to approval workflows, cost allocation, and monitoring.

**Key governance areas:**
- Organization Policy constraints for GPU resources
- Project-level quota allocation across teams
- IAM policies for GPU access control
- Chargeback and cost allocation (labels, tagging)
- Quota monitoring and alerting
- Approval workflows for high-cost GPU requests
- Resource hierarchy governance (org → folder → project)

From [Google Cloud Organization Policy Constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-11-16):
> "Organization policies let administrators set restrictions on how resources can be configured. Policies are inherited down the resource hierarchy and can enforce restrictions like allowed regions, required labels, or restricted GPU types."

**Critical principle**: GPU quotas are zero by default. All cloud providers (GCP, AWS, Azure) start with GPU quota = 0 to prevent accidental cost overruns. You must explicitly request GPU quota increases.

---

## 1. Organization Policies for GPU Resources

### 1.1 GPU Type Restrictions

**Use case**: Restrict which GPU types teams can provision (e.g., only T4/L4 for dev, A100/H100 for prod).

**Organization Policy constraints:**
```yaml
# Restrict GPU types to cost-effective options
constraint: constraints/compute.restrictGpuTypes
listPolicy:
  allowedValues:
    - "nvidia-tesla-t4"        # Cheapest option
    - "nvidia-l4"              # Cost-effective inference
    - "nvidia-tesla-v100"      # Training (legacy)
  deniedValues:
    - "nvidia-tesla-a100"      # Expensive, prod-only
    - "nvidia-h100-80gb"       # Most expensive
```

**Applying GPU type restrictions:**
```bash
# Create organization policy YAML
cat > gpu-type-restriction.yaml <<EOF
name: projects/PROJECT_ID/policies/compute.restrictGpuTypes
spec:
  rules:
    - values:
        allowedValues:
          - "nvidia-tesla-t4"
          - "nvidia-l4"
EOF

# Apply policy to organization
gcloud org-policies set-policy gpu-type-restriction.yaml \
  --organization=ORG_ID

# Or apply to specific project
gcloud org-policies set-policy gpu-type-restriction.yaml \
  --project=PROJECT_ID

# Verify policy
gcloud org-policies describe compute.restrictGpuTypes \
  --organization=ORG_ID
```

**Testing the policy:**
```bash
# This will succeed (T4 allowed)
gcloud compute instances create gpu-dev-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1

# This will fail (A100 denied by policy)
gcloud compute instances create gpu-prod-instance \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1
# Error: Constraint compute.restrictGpuTypes violated
```

From [Rafay Multi-Tenant GPU Quota Management](https://docs.rafay.co/blog/2025/06/27/configure-and-manage-gpu-resource-quotas-in-multi-tenant-clouds/) (June 27, 2025):
> "The Org Admin is responsible for partitioning the organization's total GPU quota across internal teams and projects. This is essential to prevent resource contention and ensure fair allocation."

---

### 1.2 Regional Constraints for GPUs

**Use case**: Restrict GPU provisioning to specific regions (cost optimization, data residency, quota availability).

**Regional restrictions:**
```yaml
# Restrict GPU instances to specific regions
constraint: constraints/gcp.resourceLocations
listPolicy:
  allowedValues:
    - "in:us-locations"           # All US regions
    - "in:us-central1-locations"  # Specific region
    - "in:us-west2-locations"     # arr-coc-0-1 primary region
  deniedValues:
    - "in:asia-locations"         # Expensive, higher latency
    - "in:europe-locations"       # Data residency concerns
```

**Applying regional constraints:**
```bash
# Create policy file
cat > gpu-region-restriction.yaml <<EOF
name: organizations/ORG_ID/policies/gcp.resourceLocations
spec:
  rules:
    - values:
        allowedValues:
          - "in:us-west2-locations"
          - "in:us-central1-locations"
EOF

# Apply to organization
gcloud org-policies set-policy gpu-region-restriction.yaml \
  --organization=ORG_ID

# Attempting to create GPU instance outside allowed regions will fail
gcloud compute instances create gpu-instance \
  --zone=europe-west4-a \  # Denied by policy
  --accelerator=type=nvidia-tesla-t4,count=1
# Error: Location europe-west4-a violates organization policy
```

---

### 1.3 Custom Constraints for GPU Governance

**Use case**: Require approval labels, enforce tagging standards, restrict GPU count per instance.

**Custom constraint examples:**
```yaml
# Example 1: Require approval label on GPU instances
custom_constraint:
  name: "customConstraints/requireGpuApproval"
  resource_types:
    - "compute.googleapis.com/Instance"
  condition:
    expression: |
      resource.labels.has('gpu-approved') &&
      resource.labels['gpu-approved'] == 'true' &&
      resource.accelerators.exists(a, a.acceleratorType.contains('nvidia'))
    title: "Require approval label for GPU instances"
    description: "All GPU instances must have gpu-approved=true label"
  action_type: DENY

# Example 2: Limit GPU count per instance
custom_constraint:
  name: "customConstraints/maxGpuCount"
  resource_types:
    - "compute.googleapis.com/Instance"
  condition:
    expression: |
      !resource.accelerators.exists(a,
        a.acceleratorCount > 4
      )
    title: "Maximum 4 GPUs per instance"
    description: "Prevent single instances from consuming too many GPUs"
  action_type: DENY

# Example 3: Require cost center label
custom_constraint:
  name: "customConstraints/requireCostCenter"
  resource_types:
    - "compute.googleapis.com/Instance"
  condition:
    expression: |
      resource.labels.has('cost-center') &&
      resource.labels.has('project-code') &&
      resource.accelerators.size() > 0
    title: "Require cost tracking labels on GPU instances"
    description: "GPU instances must have cost-center and project-code labels"
  action_type: DENY
```

**Creating custom constraints:**
```bash
# Create custom constraint
gcloud org-policies set-custom-constraint custom-gpu-constraint.yaml \
  --organization=ORG_ID

# custom-gpu-constraint.yaml:
cat > custom-gpu-constraint.yaml <<EOF
name: organizations/ORG_ID/customConstraints/requireGpuApproval
resourceTypes:
  - compute.googleapis.com/Instance
methodTypes:
  - CREATE
  - UPDATE
condition: |
  resource.labels.has('gpu-approved') &&
  resource.labels['gpu-approved'] == 'true' &&
  resource.accelerators.size() > 0
actionType: DENY
displayName: "Require GPU Approval Label"
description: "All GPU instances must have gpu-approved=true label before creation"
EOF

# Enforce the custom constraint
cat > enforce-gpu-approval.yaml <<EOF
name: organizations/ORG_ID/policies/customConstraints/requireGpuApproval
spec:
  rules:
    - enforce: true
EOF

gcloud org-policies set-policy enforce-gpu-approval.yaml \
  --organization=ORG_ID

# Now GPU instances require approval label
gcloud compute instances create gpu-dev \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --labels=gpu-approved=true,cost-center=ml-team,project-code=arr-coc-0-1
  # Success - has required labels

gcloud compute instances create gpu-dev-2 \
  --accelerator=type=nvidia-tesla-t4,count=1
  # Error: Missing gpu-approved=true label
```

From [Google Cloud Organization Policy](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-11-16):
> "Custom constraints provide fine-grained control over specific resource fields. They use Common Expression Language (CEL) to define validation rules."

---

## 2. Project-Level Quota Allocation

### 2.1 Quota Structure and Hierarchy

**Quota types:**
```yaml
# GPU quotas are per-region, per-GPU-type
quota_hierarchy:
  organization_level:
    total_quota: "Managed by Google (request increases via support)"

  project_level:
    quota_types:
      - "GPUs (all regions)"                # Global GPU quota
      - "NVIDIA T4 GPUs"                    # Specific GPU type
      - "NVIDIA A100 GPUs"                  # High-end GPU quota
      - "Preemptible NVIDIA T4 GPUs"        # Separate preemptible quota

    regional_quotas:
      us_west2:
        - "NVIDIA_T4_GPUS: 8"
        - "NVIDIA_A100_80GB_GPUS: 4"
        - "PREEMPTIBLE_NVIDIA_T4_GPUS: 16"
      us_central1:
        - "NVIDIA_T4_GPUS: 4"
        - "NVIDIA_L4_GPUS: 8"
```

**Viewing current quotas:**
```bash
# List all GPU quotas for project
gcloud compute project-info describe --project=PROJECT_ID \
  --format="table(quotas.filter('GPU').metric, quotas.filter('GPU').limit, quotas.filter('GPU').usage)"

# List quotas by region
gcloud compute regions describe us-west2 \
  --project=PROJECT_ID \
  --format="table(quotas.filter('GPU').metric, quotas.filter('GPU').limit)"

# View quota usage
gcloud compute project-info describe --project=PROJECT_ID \
  --format="yaml(quotas)" | grep -A 3 "GPU"

# Output:
# - metric: NVIDIA_T4_GPUS
#   limit: 8.0
#   usage: 6.0
# - metric: NVIDIA_A100_80GB_GPUS
#   limit: 4.0
#   usage: 0.0
```

---

### 2.2 Allocating Quotas Across Teams

**Multi-team quota allocation pattern:**
```yaml
# Organization quota partitioning
organization_quota: 32 A100 GPUs (us-west2)

team_allocations:
  ml_research:
    project: "ml-research-prod"
    quota: 16 A100 GPUs
    justification: "Large-scale model training"

  ml_engineering:
    project: "ml-engineering-prod"
    quota: 8 A100 GPUs
    justification: "Production model fine-tuning"

  ml_experimentation:
    project: "ml-experimentation-dev"
    quota: 4 A100 GPUs
    justification: "Prototype development"

  shared_pool:
    project: "ml-shared-resources"
    quota: 4 A100 GPUs
    justification: "Overflow capacity for urgent requests"
```

**Implementing team quotas:**
```bash
# Request quota for ML Research team
gcloud compute project-info describe --project=ml-research-prod

# Submit quota increase request (via Console or Support)
# 1. Navigate to: IAM & Admin > Quotas
# 2. Filter: Metric = "NVIDIA A100 80GB GPUs", Region = "us-west2"
# 3. Select quota, click "EDIT QUOTAS"
# 4. New limit: 16
# 5. Justification: "ML research team - training 70B parameter models"

# Alternative: gcloud CLI (requires quota adjustment API access)
gcloud alpha compute project-info set-usage-export-bucket \
  gs://ml-research-quota-tracking \
  --project=ml-research-prod
```

**Quota request justification template:**
```
Project: ml-research-prod
Quota: NVIDIA_A100_80GB_GPUS (us-west2)
Current limit: 0
Requested limit: 16

Justification:
- Team size: 15 ML researchers
- Workload: Training 70B parameter language models
- Training time: 3-5 days per experiment, 2-3 experiments/week
- Estimated GPU hours: ~500 GPU-hours/week
- Business impact: Research supporting $10M/year product line
- Timeline: Immediate (Q1 2025 roadmap dependency)
```

---

### 2.3 Quota Monitoring and Alerting

**Setting up quota alerts:**
```bash
# Create Cloud Monitoring alert for quota usage
cat > gpu-quota-alert.yaml <<EOF
displayName: "GPU Quota Usage Alert (80% Threshold)"
conditions:
  - displayName: "GPU quota usage > 80%"
    conditionThreshold:
      filter: |
        resource.type = "consumer_quota"
        AND metric.type = "serviceruntime.googleapis.com/quota/allocation/usage"
        AND metric.labels.quota_metric = "compute.googleapis.com/nvidia_a100_80gb_gpus"
      comparison: COMPARISON_GT
      thresholdValue: 0.8  # 80% of quota
      duration: 300s       # 5 minutes
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
notificationChannels:
  - "projects/PROJECT_ID/notificationChannels/CHANNEL_ID"
alertStrategy:
  notificationRateLimit:
    period: 3600s  # Max 1 alert per hour
EOF

gcloud alpha monitoring policies create --policy-from-file=gpu-quota-alert.yaml

# Create notification channel for alerts
gcloud alpha monitoring channels create \
  --display-name="GPU Quota Alerts - Slack" \
  --type=slack \
  --channel-labels=url=SLACK_WEBHOOK_URL
```

**Monitoring quota usage programmatically:**
```python
# Example: Monitor GPU quota usage
from google.cloud import monitoring_v3
from google.cloud import compute_v1

def check_gpu_quota_usage(project_id, region="us-west2"):
    """Check current GPU quota usage."""
    compute_client = compute_v1.RegionsClient()

    # Get region quotas
    region_info = compute_client.get(project=project_id, region=region)

    gpu_quotas = {}
    for quota in region_info.quotas:
        if "GPU" in quota.metric:
            usage_pct = (quota.usage / quota.limit) * 100 if quota.limit > 0 else 0
            gpu_quotas[quota.metric] = {
                "limit": quota.limit,
                "usage": quota.usage,
                "available": quota.limit - quota.usage,
                "usage_pct": usage_pct
            }

    return gpu_quotas

# Run quota check
quotas = check_gpu_quota_usage("ml-research-prod", "us-west2")
for metric, data in quotas.items():
    print(f"{metric}: {data['usage']}/{data['limit']} ({data['usage_pct']:.1f}%)")

    # Alert if > 80%
    if data['usage_pct'] > 80:
        print(f"⚠️  WARNING: {metric} quota usage > 80%")
        # Send alert (Slack, email, PagerDuty, etc.)
```

From [Medium: Things Cloud Providers Don't Tell You About GPUs](https://medium.com/@jonathan.hoffman91/things-cloud-providers-dont-explicitly-tell-you-about-deploying-gpu-compute-instances-a40aaf3c7027) (2023):
> "It may come as a surprise, but all the cloud providers I tested, including GCP, AWS, and Azure, start with a default GPU instance quota of zero. You must explicitly request quota increases before provisioning GPU resources."

---

## 3. IAM Policies for GPU Resources

### 3.1 Role-Based Access Control for GPUs

**IAM role hierarchy for GPU resources:**
```yaml
# GPU access control matrix
roles:
  gpu_admin:
    role: "roles/compute.instanceAdmin"
    permissions:
      - Create GPU instances
      - Delete GPU instances
      - Modify GPU configurations
      - Request quota increases
    assigned_to:
      - "group:ml-ops@company.com"

  gpu_user:
    role: "roles/compute.instanceAdmin.v1"
    permissions:
      - Create GPU instances (within quota)
      - Stop/start GPU instances
      - View GPU metrics
    assigned_to:
      - "group:ml-engineers@company.com"
      - "group:ml-researchers@company.com"

  gpu_viewer:
    role: "roles/compute.viewer"
    permissions:
      - View GPU instances
      - View GPU metrics
      - NO create/modify permissions
    assigned_to:
      - "group:finance-team@company.com"  # Cost tracking
      - "group:security-auditors@company.com"
```

**Setting up IAM roles for GPU access:**
```bash
# Grant GPU admin role to ML Ops team
gcloud projects add-iam-policy-binding ml-research-prod \
  --member="group:ml-ops@company.com" \
  --role="roles/compute.instanceAdmin"

# Grant GPU user role to ML engineers (can create instances)
gcloud projects add-iam-policy-binding ml-research-prod \
  --member="group:ml-engineers@company.com" \
  --role="roles/compute.instanceAdmin.v1"

# Grant viewer role to finance (cost tracking only)
gcloud projects add-iam-policy-binding ml-research-prod \
  --member="group:finance@company.com" \
  --role="roles/compute.viewer"

# Custom role: GPU instance creator (limited permissions)
gcloud iam roles create gpuInstanceCreator \
  --project=ml-research-prod \
  --title="GPU Instance Creator" \
  --description="Can create GPU instances but not delete" \
  --permissions=compute.instances.create,compute.instances.get,compute.instances.list,compute.instances.start,compute.instances.stop \
  --stage=GA

gcloud projects add-iam-policy-binding ml-research-prod \
  --member="group:ml-interns@company.com" \
  --role="projects/ml-research-prod/roles/gpuInstanceCreator"
```

---

### 3.2 Conditional IAM for GPU Resources

**Use case**: Allow GPU instance creation only during business hours, only for specific GPU types, or only with approval labels.

**Conditional IAM examples:**
```bash
# Example 1: Restrict GPU creation to business hours (9am-5pm UTC)
gcloud projects add-iam-policy-binding ml-research-prod \
  --member="user:researcher@company.com" \
  --role="roles/compute.instanceAdmin" \
  --condition='
    expression=request.time.getHours() >= 9 && request.time.getHours() < 17,
    title=business-hours-only,
    description=GPU instances can only be created during business hours
  '

# Example 2: Allow only T4 GPU creation (cost control)
gcloud projects add-iam-policy-binding ml-experimentation-dev \
  --member="group:ml-interns@company.com" \
  --role="roles/compute.instanceAdmin" \
  --condition='
    expression=resource.type == "compute.googleapis.com/Instance" &&
               resource.acceleratorType.contains("nvidia-tesla-t4"),
    title=t4-only-access,
    description=Interns can only create T4 GPU instances
  '

# Example 3: Require manager approval via label
gcloud projects add-iam-policy-binding ml-research-prod \
  --member="user:researcher@company.com" \
  --role="roles/compute.instanceAdmin" \
  --condition='
    expression=resource.labels.has("manager-approved") &&
               resource.labels["manager-approved"] == "true",
    title=manager-approval-required,
    description=GPU instances require manager-approved label
  '

# Example 4: Time-limited GPU access (temporary project)
gcloud projects add-iam-policy-binding ml-hackathon-2025 \
  --member="group:hackathon-participants@company.com" \
  --role="roles/compute.instanceAdmin" \
  --condition='
    expression=request.time < timestamp("2025-12-31T00:00:00Z"),
    title=hackathon-temporary-access,
    description=GPU access expires on December 31 2025
  '
```

From [GCP IAM Service Accounts for ML Security](../gcloud-iam/00-service-accounts-ml-security.md):
> "Conditional IAM policies provide fine-grained access control based on attributes like time, resource labels, or request context. Use them to enforce approval workflows and cost controls."

---

## 4. Chargeback and Cost Allocation

### 4.1 Label-Based Cost Allocation

**Labeling strategy for GPU chargeback:**
```yaml
# Required labels for GPU instances
required_labels:
  cost-center:
    values: ["ml-research", "ml-engineering", "ml-ops"]
    purpose: "Departmental chargeback"

  project-code:
    values: ["arr-coc-0-1", "sentiment-analysis", "recommendation-engine"]
    purpose: "Project-level cost tracking"

  environment:
    values: ["dev", "staging", "prod"]
    purpose: "Environment-based budgeting"

  owner:
    values: ["user-email"]
    purpose: "Individual accountability"

  gpu-approved-by:
    values: ["manager-email"]
    purpose: "Approval audit trail"
```

**Enforcing labels with organization policy:**
```bash
# Create policy requiring cost labels
cat > require-cost-labels.yaml <<EOF
name: organizations/ORG_ID/policies/compute.requireLabels
spec:
  rules:
    - values:
        requiredValues:
          - "cost-center"
          - "project-code"
          - "environment"
          - "owner"
      condition:
        expression: "resource.accelerators.size() > 0"
        title: "Require labels on GPU instances"
EOF

gcloud org-policies set-policy require-cost-labels.yaml \
  --organization=ORG_ID

# Now GPU instances require these labels
gcloud compute instances create gpu-training \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --labels=cost-center=ml-research,project-code=arr-coc-0-1,environment=prod,owner=researcher@company.com,gpu-approved-by=manager@company.com
```

**Querying costs by labels:**
```bash
# Export billing data to BigQuery (required for cost analysis)
# 1. Enable BigQuery billing export in Console
# 2. Query costs by labels

bq query --use_legacy_sql=false '
SELECT
  labels.value AS cost_center,
  SUM(cost) AS total_cost,
  SUM(usage.amount) AS gpu_hours
FROM `project.dataset.gcp_billing_export_v1_BILLING_ID`
WHERE
  DATE(usage_start_time) >= "2025-11-01"
  AND sku.description LIKE "%GPU%"
  AND labels.key = "cost-center"
GROUP BY cost_center
ORDER BY total_cost DESC
'

# Output:
# +--------------+-------------+-----------+
# | cost_center  | total_cost  | gpu_hours |
# +--------------+-------------+-----------+
# | ml-research  | 24531.50    | 1250      |
# | ml-engineering| 12800.25   | 650       |
# | ml-ops       | 3200.10     | 160       |
# +--------------+-------------+-----------+
```

---

### 4.2 Cost Center Allocation and Chargeback

**Monthly chargeback report generation:**
```python
# Example: Generate GPU cost chargeback report
from google.cloud import bigquery
import pandas as pd

def generate_gpu_chargeback_report(billing_table, start_date, end_date):
    """Generate monthly GPU cost chargeback by cost center."""
    client = bigquery.Client()

    query = f"""
    SELECT
      labels.value AS cost_center,
      labels.value AS project_code,
      sku.description AS gpu_type,
      COUNT(DISTINCT resource.name) AS instance_count,
      SUM(cost) AS total_cost,
      SUM(usage.amount) AS total_gpu_hours,
      AVG(cost / usage.amount) AS avg_cost_per_hour
    FROM `{billing_table}`
    CROSS JOIN UNNEST(labels) AS labels
    WHERE
      DATE(usage_start_time) BETWEEN '{start_date}' AND '{end_date}'
      AND sku.description LIKE '%GPU%'
      AND (labels.key = 'cost-center' OR labels.key = 'project-code')
    GROUP BY cost_center, project_code, gpu_type
    ORDER BY total_cost DESC
    """

    df = client.query(query).to_dataframe()

    # Generate summary
    print("=== GPU Cost Chargeback Report ===")
    print(f"Period: {start_date} to {end_date}\n")

    for cost_center in df['cost_center'].unique():
        center_df = df[df['cost_center'] == cost_center]
        total = center_df['total_cost'].sum()
        hours = center_df['total_gpu_hours'].sum()

        print(f"\n{cost_center}:")
        print(f"  Total cost: ${total:,.2f}")
        print(f"  GPU hours: {hours:,.0f}")
        print(f"  Projects:")

        for _, row in center_df.iterrows():
            print(f"    - {row['project_code']}: ${row['total_cost']:,.2f} ({row['gpu_type']})")

    return df

# Run report
df = generate_gpu_chargeback_report(
    billing_table="project.dataset.gcp_billing_export_v1_BILLING_ID",
    start_date="2025-11-01",
    end_date="2025-11-30"
)

# Export to CSV for finance team
df.to_csv("gpu_chargeback_november_2025.csv", index=False)
```

**Automated chargeback workflow:**
```bash
# Schedule monthly chargeback report with Cloud Scheduler
gcloud scheduler jobs create pubsub gpu-chargeback-monthly \
  --schedule="0 9 1 * *" \  # 9am on 1st of month
  --topic=gpu-chargeback-trigger \
  --message-body='{"report_type": "monthly_chargeback"}' \
  --location=us-west2

# Cloud Function to generate and email report
# (Triggered by Pub/Sub message from Cloud Scheduler)
```

From [Rafay Multi-Tenant GPU Quotas](https://docs.rafay.co/blog/2025/06/27/configure-and-manage-gpu-resource-quotas-in-multi-tenant-clouds/) (June 27, 2025):
> "GPU resource quotas allow organizations to allocate computing capacity at multiple levels—across the entire organization, at individual project levels, and even per-user allocations."

---

## 5. Approval Workflows for High-Cost GPU Requests

### 5.1 Multi-Stage Approval Process

**Approval workflow architecture:**
```yaml
# GPU request approval stages
approval_stages:
  stage_1_self_service:
    gpu_types: ["nvidia-tesla-t4", "nvidia-l4"]
    max_gpus: 2
    max_hours: 48
    approval: "Automatic (within quota)"

  stage_2_team_lead:
    gpu_types: ["nvidia-tesla-v100", "nvidia-a100"]
    max_gpus: 4
    max_cost: "$500/week"
    approval: "Team lead approval required"
    sla: "4 hours"

  stage_3_director:
    gpu_types: ["nvidia-a100", "nvidia-h100"]
    max_gpus: 8
    max_cost: "$2000/week"
    approval: "Director + Finance approval"
    sla: "24 hours"

  stage_4_executive:
    gpu_types: ["Any"]
    max_gpus: "16+"
    max_cost: ">$5000/week"
    approval: "Executive + CFO approval"
    sla: "72 hours"
```

**Implementing approval workflow with Cloud Functions:**
```python
# Example: GPU approval workflow
from google.cloud import firestore
from google.cloud import compute_v1
import functions_framework
import smtplib

@functions_framework.http
def request_gpu_approval(request):
    """HTTP endpoint to request GPU approval."""
    data = request.get_json()

    # Extract request details
    requester = data.get("requester_email")
    gpu_type = data.get("gpu_type")
    gpu_count = data.get("gpu_count")
    duration_hours = data.get("duration_hours")
    justification = data.get("justification")
    project_code = data.get("project_code")

    # Calculate estimated cost
    gpu_pricing = {
        "nvidia-tesla-t4": 0.35,
        "nvidia-l4": 0.78,
        "nvidia-tesla-a100": 3.67,
        "nvidia-h100-80gb": 5.12
    }
    hourly_cost = gpu_pricing.get(gpu_type, 0) * gpu_count
    total_cost = hourly_cost * duration_hours

    # Determine approval level
    if total_cost < 100:
        approval_level = "auto-approved"
        approvers = []
    elif total_cost < 500:
        approval_level = "team-lead"
        approvers = ["team-lead@company.com"]
    elif total_cost < 2000:
        approval_level = "director"
        approvers = ["director@company.com", "finance-manager@company.com"]
    else:
        approval_level = "executive"
        approvers = ["cto@company.com", "cfo@company.com"]

    # Create approval request in Firestore
    db = firestore.Client()
    approval_ref = db.collection("gpu_approvals").document()
    approval_ref.set({
        "requester": requester,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "duration_hours": duration_hours,
        "estimated_cost": total_cost,
        "justification": justification,
        "project_code": project_code,
        "approval_level": approval_level,
        "approvers": approvers,
        "status": "pending" if approvers else "approved",
        "requested_at": firestore.SERVER_TIMESTAMP
    })

    # Send approval emails
    if approvers:
        send_approval_email(
            approvers=approvers,
            requester=requester,
            gpu_details=f"{gpu_count}x {gpu_type} for {duration_hours}h",
            cost=total_cost,
            approval_link=f"https://approval-ui.company.com/approve/{approval_ref.id}"
        )

    return {
        "approval_id": approval_ref.id,
        "status": "pending" if approvers else "approved",
        "approval_level": approval_level,
        "estimated_cost": total_cost,
        "approvers": approvers
    }

def send_approval_email(approvers, requester, gpu_details, cost, approval_link):
    """Send approval request email to approvers."""
    subject = f"GPU Approval Request: ${cost:.2f} - {gpu_details}"
    body = f"""
    GPU Resource Approval Required

    Requester: {requester}
    GPU Configuration: {gpu_details}
    Estimated Cost: ${cost:.2f}

    Approve or deny this request:
    {approval_link}

    This request requires your approval within 24 hours.
    """

    # Send email (using SendGrid, SMTP, or Cloud Email API)
    # Implementation omitted for brevity
```

**Approval UI (simple webhook):**
```bash
# Approve GPU request via webhook
curl -X POST https://us-west2-PROJECT.cloudfunctions.net/approve-gpu-request \
  -H "Content-Type: application/json" \
  -d '{
    "approval_id": "ABC123",
    "approver": "director@company.com",
    "decision": "approved",
    "comments": "Approved for Q1 research project"
  }'

# Deny GPU request
curl -X POST https://us-west2-PROJECT.cloudfunctions.net/approve-gpu-request \
  -H "Content-Type: application/json" \
  -d '{
    "approval_id": "ABC123",
    "approver": "director@company.com",
    "decision": "denied",
    "comments": "Cost too high. Use T4 GPUs instead."
  }'
```

---

### 5.2 Integration with Infrastructure as Code

**Terraform with approval gates:**
```hcl
# Example: GPU instance with approval check
resource "google_compute_instance" "gpu_training" {
  name         = "gpu-training-approved"
  machine_type = "a2-highgpu-1g"
  zone         = "us-west2-a"

  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = 1
  }

  # Required approval labels
  labels = {
    gpu-approved    = "true"
    approved-by     = "director@company.com"
    approval-id     = "ABC123"
    cost-center     = "ml-research"
    project-code    = "arr-coc-0-1"
    environment     = "prod"
    owner           = "researcher@company.com"
  }

  lifecycle {
    # Prevent creation without approval label
    precondition {
      condition     = self.labels.gpu-approved == "true"
      error_message = "GPU instance requires approval. Set gpu-approved=true label."
    }
  }
}

# Terraform plan checks organization policy
# If org policy requires approval label, plan will fail without it
```

---

## 6. Resource Hierarchy Governance

### 6.1 Organization → Folder → Project Structure

**Multi-tier governance hierarchy:**
```yaml
# GPU governance hierarchy
organization: "company.com"
  policies:
    - "Require cost labels on all resources"
    - "Restrict GPU types globally (no H100 in dev)"

  folders:
    - name: "Production"
      policies:
        - "Allow A100, H100 GPUs"
        - "Require director approval"
      projects:
        - "ml-research-prod" (quota: 16 A100)
        - "ml-engineering-prod" (quota: 8 A100)

    - name: "Development"
      policies:
        - "Only T4, L4 GPUs allowed"
        - "Self-service approval"
      projects:
        - "ml-experimentation-dev" (quota: 8 T4)
        - "ml-prototyping-dev" (quota: 4 L4)

    - name: "Shared Resources"
      policies:
        - "All GPU types allowed"
        - "Executive approval required"
      projects:
        - "ml-shared-pool" (quota: 4 A100, 8 T4)
```

**Creating folder structure:**
```bash
# Create folders
gcloud resource-manager folders create \
  --display-name="Production" \
  --organization=ORG_ID

gcloud resource-manager folders create \
  --display-name="Development" \
  --organization=ORG_ID

# Apply folder-level policies
gcloud org-policies set-policy prod-gpu-policy.yaml \
  --folder=PRODUCTION_FOLDER_ID

# prod-gpu-policy.yaml:
cat > prod-gpu-policy.yaml <<EOF
name: folders/FOLDER_ID/policies/compute.restrictGpuTypes
spec:
  rules:
    - values:
        allowedValues:
          - "nvidia-tesla-a100"
          - "nvidia-h100-80gb"
        deniedValues:
          - "nvidia-tesla-t4"  # Not performant enough for prod
EOF

# Move projects into folders
gcloud projects move ml-research-prod \
  --folder=PRODUCTION_FOLDER_ID

gcloud projects move ml-experimentation-dev \
  --folder=DEVELOPMENT_FOLDER_ID
```

---

### 6.2 Quota Inheritance and Override

**Quota inheritance rules:**
```yaml
# Quotas are NOT inherited - they are per-project
# But policies ARE inherited down the hierarchy

inheritance_example:
  organization_policy:
    "Require cost labels": true
    # ↓ Inherited by all folders and projects

  folder_policy:
    "Allow only T4 GPUs in dev": true
    # ↓ Inherited by projects in this folder

  project_quota:
    "NVIDIA_T4_GPUS: 8"
    # ✗ NOT inherited by other projects
```

---

## 7. arr-coc-0-1 Governance Model

### 7.1 arr-coc-0-1 GPU Quota and Approval Workflow

**Project governance configuration:**
```yaml
# arr-coc-0-1 GPU governance
project: "arr-coc-0-1"
region: "us-west2"

quotas:
  dev:
    nvidia_t4_gpus: 4
    preemptible_nvidia_t4_gpus: 8
    approval: "Auto-approved (within quota)"

  staging:
    nvidia_l4_gpus: 2
    nvidia_a100_gpus: 2
    approval: "Team lead approval"

  prod:
    nvidia_a100_gpus: 8
    approval: "Director + Finance approval"

labels_required:
  - "cost-center: ml-research"
  - "project-code: arr-coc-0-1"
  - "environment: [dev|staging|prod]"
  - "owner: [user-email]"

approval_workflow:
  dev_instances:
    max_cost_per_week: "$200"
    auto_shutdown: "48 hours"

  staging_instances:
    max_cost_per_week: "$500"
    approval_sla: "4 hours"

  prod_instances:
    max_cost_per_week: "$2000"
    approval_sla: "24 hours"
    requires:
      - "Team lead sign-off"
      - "Finance review"
      - "Cost justification document"
```

**Implementing arr-coc-0-1 governance:**
```bash
# Set project-level policies
gcloud org-policies set-policy arr-coc-gpu-policy.yaml \
  --project=arr-coc-0-1

# arr-coc-gpu-policy.yaml:
cat > arr-coc-gpu-policy.yaml <<EOF
name: projects/arr-coc-0-1/policies/compute.requireLabels
spec:
  rules:
    - values:
        requiredValues:
          - "cost-center"
          - "project-code"
          - "environment"
          - "owner"
      condition:
        expression: "resource.accelerators.size() > 0"
        title: "Require labels on GPU instances"
EOF

# Create approval workflow (Cloud Functions + Firestore)
# Deploy approval UI
gcloud functions deploy request-arr-coc-gpu-approval \
  --runtime=python310 \
  --trigger-http \
  --entry-point=request_gpu_approval \
  --region=us-west2

# Set up cost monitoring
gcloud alpha monitoring policies create \
  --notification-channels=arr-coc-alerts \
  --display-name="arr-coc-0-1 Weekly GPU Cost Alert" \
  --condition-threshold-value=2000 \  # $2000/week
  --condition-filter='
    resource.type="global"
    AND metric.type="billing.googleapis.com/project/costs"
    AND metric.labels.project_id="arr-coc-0-1"
    AND metric.labels.sku LIKE "%GPU%"
  '
```

**arr-coc-0-1 example GPU request:**
```bash
# Request 8x A100 GPUs for production training
curl -X POST https://us-west2-arr-coc-0-1.cloudfunctions.net/request-arr-coc-gpu-approval \
  -H "Content-Type: application/json" \
  -d '{
    "requester_email": "ml-engineer@company.com",
    "gpu_type": "nvidia-tesla-a100",
    "gpu_count": 8,
    "duration_hours": 72,
    "justification": "Training 70B parameter model for texture relevance scoring. Critical for Q1 product launch.",
    "project_code": "arr-coc-0-1",
    "environment": "prod",
    "cost_center": "ml-research"
  }'

# Response:
# {
#   "approval_id": "ARR-2025-001",
#   "status": "pending",
#   "approval_level": "director",
#   "estimated_cost": 2114.40,
#   "approvers": ["director@company.com", "finance@company.com"],
#   "message": "Approval request sent. Expected SLA: 24 hours."
# }
```

---

## 8. Best Practices and Common Patterns

### 8.1 GPU Governance Checklist

**Pre-deployment governance setup:**
```yaml
governance_checklist:
  organization_policies:
    - [ ] Set GPU type restrictions (prod vs dev)
    - [ ] Enforce regional constraints (cost optimization)
    - [ ] Require cost labels (cost-center, project-code, owner)
    - [ ] Set max GPU count per instance

  quota_management:
    - [ ] Allocate quotas by team/project
    - [ ] Set up quota monitoring alerts (80% threshold)
    - [ ] Document quota request process
    - [ ] Establish SLA for quota increases

  cost_allocation:
    - [ ] Enable BigQuery billing export
    - [ ] Define labeling taxonomy
    - [ ] Automate monthly chargeback reports
    - [ ] Set up budget alerts per project

  approval_workflows:
    - [ ] Define approval tiers (self-service, team lead, director, executive)
    - [ ] Implement approval automation (Cloud Functions)
    - [ ] Set up notification channels
    - [ ] Document approval SLAs

  iam_configuration:
    - [ ] Create custom GPU roles
    - [ ] Set up conditional IAM (business hours, GPU types)
    - [ ] Grant least-privilege access
    - [ ] Audit IAM bindings quarterly
```

---

### 8.2 Cost Control Patterns

**GPU cost optimization strategies:**
```yaml
# Cost control best practices
cost_control:
  preemptible_gpus:
    savings: "60-91%"
    use_cases:
      - "Fault-tolerant training (checkpointing)"
      - "Batch inference jobs"
      - "Development/testing"

  auto_shutdown:
    pattern: "Shut down idle GPU instances after 2 hours"
    implementation: "Cloud Scheduler + Cloud Functions"
    savings: "30-40% on dev instances"

  right_sizing:
    pattern: "Use T4 for inference, A100 for training"
    decision_tree:
      - "Inference latency < 50ms → L4"
      - "Inference throughput → T4"
      - "Training < 10B params → T4/V100"
      - "Training > 10B params → A100"
      - "Training > 70B params → H100"

  committed_use_discounts:
    savings: "57% for 3-year commitment"
    recommendation: "Use for steady-state prod workloads"
    caveat: "Only for predictable GPU usage"
```

**Automated cost control:**
```python
# Example: Auto-shutdown idle GPU instances
from google.cloud import compute_v1
from datetime import datetime, timedelta

def shutdown_idle_gpu_instances(project_id, zone, idle_threshold_hours=2):
    """Shut down GPU instances idle for > threshold hours."""
    compute_client = compute_v1.InstancesClient()

    instances = compute_client.list(project=project_id, zone=zone)

    for instance in instances:
        # Check if instance has GPUs
        if not instance.guest_accelerators:
            continue

        # Check instance status
        if instance.status != "RUNNING":
            continue

        # Check last activity (CPU utilization)
        # (Requires Cloud Monitoring API integration)
        last_activity = get_last_activity_time(instance.name)
        idle_duration = datetime.now() - last_activity

        if idle_duration > timedelta(hours=idle_threshold_hours):
            print(f"Shutting down idle GPU instance: {instance.name}")
            print(f"  Idle for: {idle_duration}")
            print(f"  GPU type: {instance.guest_accelerators[0].accelerator_type}")

            # Shut down instance
            compute_client.stop(
                project=project_id,
                zone=zone,
                instance=instance.name
            )

# Schedule with Cloud Scheduler (runs every hour)
# gcloud scheduler jobs create pubsub shutdown-idle-gpus \
#   --schedule="0 * * * *" \
#   --topic=shutdown-idle-gpus-trigger
```

From [Google Cloud Billing and Labels](https://cloud.google.com/compute/docs/labeling-resources) (accessed 2025-11-16):
> "Labels help you organize resources and manage costs at scale. Use labels to filter billing reports, create cost allocation reports, and implement chargeback mechanisms."

---

## Summary

### GPU Governance Quick Reference

**Key commands:**
```bash
# Organization policies
gcloud org-policies set-policy POLICY.yaml --organization=ORG_ID

# Quota management
gcloud compute regions describe REGION --format="yaml(quotas)"

# IAM configuration
gcloud projects add-iam-policy-binding PROJECT --member=MEMBER --role=ROLE

# Cost tracking
bq query "SELECT labels.value, SUM(cost) FROM billing_export WHERE sku LIKE '%GPU%' GROUP BY labels.value"

# Quota alerts
gcloud alpha monitoring policies create --policy-from-file=ALERT.yaml

# Approval workflow
# Deploy Cloud Function for approval automation
```

**Governance decision tree:**
```yaml
decision_tree:
  question_1: "Who needs GPU access?"
  answer:
    - "ML researchers → Full access (roles/compute.instanceAdmin)"
    - "ML engineers → Limited access (custom role)"
    - "Interns → T4-only access (conditional IAM)"

  question_2: "What cost controls are needed?"
  answer:
    - "< $500/month → Self-service"
    - "$500-2000/month → Manager approval"
    - "> $2000/month → Director approval"

  question_3: "How to allocate costs?"
  answer:
    - "By team → cost-center label"
    - "By project → project-code label"
    - "By individual → owner label"

  question_4: "What quotas per team?"
  answer:
    - "Research → 16 A100 GPUs"
    - "Engineering → 8 A100 GPUs"
    - "Experimentation → 8 T4 GPUs"
```

---

## Sources

**Source Documents:**
- [gcp-vertex/18-compliance-governance-audit.md](../gcp-vertex/18-compliance-governance-audit.md) - Vertex AI governance and compliance
- [gcloud-iam/00-service-accounts-ml-security.md](../gcloud-iam/00-service-accounts-ml-security.md) - IAM and service account security

**Web Research:**
- [Google Cloud Organization Policy Constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-11-16)
- [Compute Engine Allocation Quotas](https://cloud.google.com/compute/resource-usage) (accessed 2025-11-16)
- [Google Cloud Labeling Resources](https://cloud.google.com/compute/docs/labeling-resources) (accessed 2025-11-16)
- [Rafay: Configure and Manage GPU Resource Quotas in Multi-Tenant Clouds](https://docs.rafay.co/blog/2025/06/27/configure-and-manage-gpu-resource-quotas-in-multi-tenant-clouds/) (June 27, 2025)
- [Medium: Things Cloud Providers Don't Tell You About GPUs](https://medium.com/@jonathan.hoffman91/things-cloud-providers-dont-explicitly-tell-you-about-deploying-gpu-compute-instances-a40aaf3c7027) (2023)
- [CloudZero: An Expert Guide On GCP Cost Monitoring](https://www.cloudzero.com/blog/gcp-cost-monitoring/) (April 9, 2024)
- [Cloud Security Alliance: A Guide to GCP Organization Policy](https://cloudsecurityalliance.org/articles/a-guide-to-gcp-organization-policy-managing-access) (March 12, 2024)

**Related Documentation:**
- gcp-gpu/01-gpu-quotas-management.md - GPU quota basics and regional availability
- gcp-gpu/16-gpu-cost-optimization-strategies.md - GPU cost optimization (PART 17)
- gcp-gpu/17-gpu-monitoring-observability.md - GPU monitoring (PART 18)

---

*This knowledge file provides production-ready GPU governance, quota management, and cost allocation for multi-team GCP environments.*
