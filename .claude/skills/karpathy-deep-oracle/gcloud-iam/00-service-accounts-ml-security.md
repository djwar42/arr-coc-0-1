# GCP IAM & Service Accounts for Machine Learning Security

**Knowledge File: Identity and Access Management for production ML workloads on Google Cloud**

---

## Overview

Identity and Access Management (IAM) is the foundation of secure ML operations on Google Cloud. Unlike AWS's complex IAM structure or Azure's role-based approach, GCP IAM provides a unified model where service accounts act as both identities for machines and grantees for permissions. For ML workloads running on Vertex AI, GKE, or Compute Engine, proper IAM configuration is the difference between a secure production system and a compliance nightmare.

**Why IAM Matters for ML:**
- Training jobs access sensitive datasets (PII, financial, health data)
- Model artifacts require protection (IP theft, adversarial manipulation)
- Inference endpoints handle user data (GDPR, CCPA, HIPAA compliance)
- Pipeline orchestration crosses project boundaries
- GPU/TPU resources cost thousands per hour (resource theft prevention)

From [Best practices for using service accounts securely](https://docs.cloud.google.com/iam/docs/best-practices-service-accounts) (accessed 2025-02-03):
> "Service accounts are a special type of Google account intended to represent a non-human user that needs to authenticate and be authorized to access data in Google APIs. They are the primary way workloads running on Google Cloud access other resources."

**Key Principle**: Least privilege is not a suggestion—it's a requirement. Google now automatically disables leaked service account keys detected in public repos (launched May 2024).

---

## Section 1: IAM Fundamentals for ML Engineers

### 1.1 GCP IAM Model

**Three Core Components:**

1. **Principals** (Who)
   - User accounts (humans)
   - Service accounts (machines, applications, workloads)
   - Groups
   - Domains

2. **Roles** (What they can do)
   - **Basic roles**: Owner, Editor, Viewer (avoid for production)
   - **Predefined roles**: ~300 curated by Google (e.g., `roles/aiplatform.user`)
   - **Custom roles**: User-defined permissions

3. **Resources** (Where)
   - Organization → Folder → Project → Resource hierarchy
   - Permissions inherit down the tree

**IAM Policy Binding:**
```
Principal + Role + Resource = Permission
```

### 1.2 Service Accounts Explained

A service account is a Google account for non-human entities. Unlike AWS IAM roles (which are assumed), GCP service accounts ARE the identity.

**Service Account Email Format:**
```
SERVICE_ACCOUNT_NAME@PROJECT_ID.iam.gserviceaccount.com
```

**Two Types:**

1. **User-managed** (you create):
   - Full control over lifecycle
   - Custom permissions
   - Key management responsibility
   - Example: `ml-training-sa@my-project.iam.gserviceaccount.com`

2. **Google-managed** (automatically created):
   - Default Compute Engine service account
   - Default App Engine service account
   - **Security risk**: Often over-privileged (Editor role by default)

From [Datadog Security Labs: Exploring Google Cloud Default Service Accounts](https://securitylabs.datadoghq.com/articles/google-cloud-default-service-accounts/) (October 29, 2024):
> "The default Compute Engine service account is granted the Editor role at the project level, providing broad access to most Google Cloud services. This violates the principle of least privilege and creates significant security exposure."

**Critical Decision**: Always create custom service accounts for ML workloads. Never use default service accounts in production.

### 1.3 Roles vs Permissions

**Permissions** are atomic actions:
```
aiplatform.datasets.create
aiplatform.models.upload
storage.buckets.get
storage.objects.list
```

**Roles** bundle permissions:
```
roles/aiplatform.user includes:
  - aiplatform.batchPredictionJobs.*
  - aiplatform.customJobs.*
  - aiplatform.models.*
  - storage.objects.get (if needed)
```

**Hierarchy:**
```
Organization-level role grant
  ↓ (inherited)
Folder-level role grant
  ↓ (inherited)
Project-level role grant
  ↓ (inherited)
Resource-level role grant
```

**Union of permissions**: If granted `roles/viewer` at project level and `roles/storage.objectCreator` at bucket level, you get BOTH sets of permissions.

---

## Section 2: Service Accounts for ML Workloads

### 2.1 Default vs Custom Service Accounts

**Default Compute Engine Service Account:**
```
PROJECT_NUMBER-compute@developer.gserviceaccount.com
```

**Problems:**
- Granted Editor role by default (WAY too permissive)
- Shared across all VMs in project
- If compromised, entire project is exposed
- Enables lateral movement attacks

**Solution: Create Custom Service Accounts**

**Training Job Service Account:**
```bash
# Create service account
gcloud iam service-accounts create ml-training-sa \
    --display-name="ML Training Jobs" \
    --description="Service account for Vertex AI training jobs"

# Grant minimal required roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

**Inference Endpoint Service Account:**
```bash
gcloud iam service-accounts create ml-inference-sa \
    --display-name="ML Inference Endpoints" \
    --description="Service account for Vertex AI prediction endpoints"

# Inference needs read-only access to models, not training data
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-inference-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.predictor"
```

### 2.2 Essential IAM Roles for ML

**Vertex AI Roles:**

From [Vertex AI access control with IAM](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) (accessed 2025-02-03):

| Role | Use Case | Key Permissions |
|------|----------|----------------|
| `roles/aiplatform.admin` | Admin only | Full control (avoid for workloads) |
| `roles/aiplatform.user` | Training/pipelines | Create jobs, upload models, manage endpoints |
| `roles/aiplatform.predictor` | Inference only | Call prediction endpoints (read-only) |
| `roles/aiplatform.viewer` | Monitoring | View jobs, models, endpoints (no writes) |

**Storage Roles:**

| Role | Use Case | Permissions |
|------|----------|-------------|
| `roles/storage.objectViewer` | Read training data | List/get objects (no write) |
| `roles/storage.objectCreator` | Write model artifacts | Create objects (no delete/overwrite) |
| `roles/storage.objectAdmin` | Full bucket access | All operations (use sparingly) |

**BigQuery Roles (for feature stores):**

| Role | Use Case |
|------|----------|
| `roles/bigquery.dataViewer` | Read features for training |
| `roles/bigquery.jobUser` | Run queries for feature engineering |

**Example: Training Job Permissions:**
```bash
# Vertex AI training job needs:
# 1. Create/manage training jobs
# 2. Read training data from GCS
# 3. Write model artifacts to GCS
# 4. (Optional) Read features from BigQuery

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Grant GCS read on training data bucket
gsutil iam ch serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://my-training-data-bucket

# Grant GCS write on model artifacts bucket
gsutil iam ch serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
    gs://my-model-artifacts-bucket
```

### 2.3 Terraform Example for ML Service Accounts

```hcl
# Create service account for training
resource "google_service_account" "ml_training" {
  account_id   = "ml-training-sa"
  display_name = "ML Training Jobs"
  project      = var.project_id
}

# Grant Vertex AI user role
resource "google_project_iam_member" "training_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}

# Grant GCS read on training data
resource "google_storage_bucket_iam_member" "training_data_read" {
  bucket = google_storage_bucket.training_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ml_training.email}"
}

# Grant GCS write on model artifacts
resource "google_storage_bucket_iam_member" "model_artifacts_write" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.ml_training.email}"
}

# Create service account for inference
resource "google_service_account" "ml_inference" {
  account_id   = "ml-inference-sa"
  display_name = "ML Inference Endpoints"
  project      = var.project_id
}

# Grant Vertex AI predictor role (read-only for inference)
resource "google_project_iam_member" "inference_predictor" {
  project = var.project_id
  role    = "roles/aiplatform.predictor"
  member  = "serviceAccount:${google_service_account.ml_inference.email}"
}
```

---

## Section 3: Security Best Practices for ML

### 3.1 Principle of Least Privilege

From [Google Cloud Blog: Move from always-on privileges to on-demand access](https://cloud.google.com/blog/products/identity-security/move-from-always-on-privileges-to-on-demand-access-with-privileged-access-manager) (June 11, 2024):
> "The principle of least privilege, when applied practically, balances security and operational efficiency. Privileged Access Manager (PAM) allows administrators to grant just-in-time access with approval workflows, reducing standing privileges."

**Practical Implementation:**

**1. Separate Service Accounts by Function**
```
ml-training-dev-sa       → Development training jobs
ml-training-staging-sa   → Staging validation
ml-training-prod-sa      → Production training
ml-inference-prod-sa     → Production inference only
ml-pipeline-orchestrator → Pipeline coordination
```

**2. Use Resource-Level IAM (not project-level)**
```bash
# Bad: Project-wide access
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Good: Bucket-specific access
gsutil iam ch serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://specific-training-bucket
```

**3. Time-Bound Credentials**
```bash
# Generate short-lived access token (1 hour)
gcloud auth print-access-token \
    --impersonate-service-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --lifetime=3600s
```

### 3.2 Service Account Key Management

**Critical Security Update (May 15, 2024):**

From [Automatically disabling leaked service account keys](https://cloud.google.com/blog/products/identity-security/automatically-disabling-leaked-service-account-keys-what-you-need-to-know) (accessed 2025-02-03):
> "Starting June 16, 2024, Google Cloud automatically disables service account keys detected in public repositories, secrets scanning services, and other exposed locations. Keys are disabled within minutes of detection."

**Key Management Best Practices:**

**1. Avoid Keys Entirely (Preferred)**
```bash
# Instead of downloading keys, use Workload Identity or VM-attached service accounts
# Vertex AI training jobs automatically use attached service account
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=my-training-job \
    --service-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --config=job.yaml
```

**2. If Keys Required, Rotate Regularly**
```bash
# Create new key
gcloud iam service-accounts keys create key-new.json \
    --iam-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com

# Update application to use new key
# Test thoroughly
# Delete old key
gcloud iam service-accounts keys delete KEY_ID \
    --iam-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com
```

**3. Monitor Key Age**
```bash
# List all keys and creation dates
gcloud iam service-accounts keys list \
    --iam-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --format="table(name,validAfterTime,validBeforeTime)"

# Keys older than 90 days should be rotated
```

**4. Use Key Rotation Automation**

From recent industry analysis (Growth Market Reports, 2024):
> "The Service Account Key Rotation Automation market reached $1.26 billion in 2024, driven by regulatory compliance (SOC 2, PCI DSS) and reduction of operational overhead. Automated rotation systems cut manual effort by 80%."

### 3.3 Workload Identity (GKE Best Practice)

**Problem**: Pods on GKE traditionally used downloaded service account keys stored as Kubernetes secrets—a security anti-pattern.

**Solution**: Workload Identity Federation for GKE

From [About Workload Identity Federation for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) (August 30, 2024):
> "Workload Identity Federation for GKE allows Kubernetes workloads to impersonate IAM service accounts, eliminating the need to manage and distribute service account keys. It provides fine-grained identity and authorization for each workload."

**How It Works:**
```
Kubernetes Service Account (KSA)
        ↓ (bound via annotation)
Google Service Account (GSA)
        ↓ (has IAM roles)
Google Cloud APIs
```

**Setup Example:**
```bash
# 1. Enable Workload Identity on cluster
gcloud container clusters create ml-cluster \
    --workload-pool=PROJECT_ID.svc.id.goog \
    --region=us-central1

# 2. Create Google Service Account
gcloud iam service-accounts create ml-workload-sa

# 3. Grant IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-workload-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# 4. Create Kubernetes Service Account
kubectl create serviceaccount ml-ksa -n ml-namespace

# 5. Bind KSA to GSA
gcloud iam service-accounts add-iam-policy-binding \
    ml-workload-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[ml-namespace/ml-ksa]" \
    --role="roles/iam.workloadIdentityUser"

# 6. Annotate KSA
kubectl annotate serviceaccount ml-ksa \
    -n ml-namespace \
    iam.gke.io/gcp-service-account=ml-workload-sa@PROJECT_ID.iam.gserviceaccount.com

# 7. Use in Pod
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
  namespace: ml-namespace
spec:
  serviceAccountName: ml-ksa  # No keys needed!
  containers:
  - name: training
    image: gcr.io/PROJECT_ID/ml-training:latest
```

**Benefits:**
- No service account keys to manage
- Automatic credential rotation
- Fine-grained per-pod identity
- Audit trail in Cloud Logging

### 3.4 Service Account Impersonation

**Use Case**: Developers need temporary elevated access without permanent permissions.

```bash
# Developer's account doesn't have direct access
# Instead, impersonate service account temporarily
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=test-job \
    --impersonate-service-account=ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --config=job.yaml

# Requires developer to have roles/iam.serviceAccountUser on the service account
gcloud iam service-accounts add-iam-policy-binding \
    ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="user:developer@company.com" \
    --role="roles/iam.serviceAccountUser" \
    --condition='expression=request.time < timestamp("2025-12-31T00:00:00Z"),title=temporary'
```

---

## Section 4: Common ML Patterns

### 4.1 Training Job Service Account

**Scenario**: Vertex AI Custom Job for distributed training

**Required Permissions:**
- Read training data from GCS
- Write checkpoints/logs to GCS
- Upload final model to Vertex AI Model Registry
- (Optional) Log metrics to W&B or MLflow

**Service Account Setup:**
```bash
# Create service account
gcloud iam service-accounts create vertex-training-sa

# Vertex AI permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# GCS training data (read-only)
gsutil iam ch serviceAccount:vertex-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://training-data-bucket

# GCS checkpoints (read-write)
gsutil iam ch serviceAccount:vertex-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectAdmin \
    gs://checkpoints-bucket

# GCS model artifacts (write-only)
gsutil iam ch serviceAccount:vertex-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
    gs://model-artifacts-bucket
```

### 4.2 Pipeline Service Account (Orchestration)

**Scenario**: Vertex AI Pipelines coordinating multi-step ML workflow

**Needs:**
- Create training jobs
- Create batch prediction jobs
- Manage datasets
- Trigger other pipelines

**Service Account Setup:**
```bash
gcloud iam service-accounts create vertex-pipeline-sa

# Pipeline orchestration permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-pipeline-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Ability to act as training service account
gcloud iam service-accounts add-iam-policy-binding \
    vertex-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:vertex-pipeline-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

### 4.3 Inference Endpoint Service Account

**Scenario**: Vertex AI Prediction Endpoint serving real-time traffic

**Needs:**
- Load model from Model Registry
- Read model artifacts from GCS
- Log predictions (optional)
- NO training data access

**Service Account Setup:**
```bash
gcloud iam service-accounts create vertex-inference-sa

# Inference-only permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:vertex-inference-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.predictor"

# GCS model artifacts (read-only)
gsutil iam ch serviceAccount:vertex-inference-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://model-artifacts-bucket
```

### 4.4 Cross-Project Access

**Scenario**: Training job in Project A needs data from Project B

**Setup:**
```bash
# In Project B (data owner)
gsutil iam ch serviceAccount:training-sa@PROJECT_A.iam.gserviceaccount.com:objectViewer \
    gs://project-b-data-bucket

# In Project A (training job)
gcloud ai custom-jobs create \
    --region=us-central1 \
    --service-account=training-sa@PROJECT_A.iam.gserviceaccount.com \
    --config=job.yaml
```

**Security Considerations:**
- Use VPC Service Controls to restrict data exfiltration
- Enable Data Access audit logs
- Consider GCS bucket locks for immutable datasets

---

## Section 5: Troubleshooting IAM Issues

### 5.1 Common Permission Errors

**Error: `Permission denied on resource`**

```bash
# Check effective permissions
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --format="table(bindings.role)"

# Check resource-level permissions
gsutil iam get gs://my-bucket | grep ml-sa@PROJECT_ID.iam.gserviceaccount.com
```

**Error: `The caller does not have permission to impersonate the service account`**

Solution: Grant `roles/iam.serviceAccountUser`:
```bash
gcloud iam service-accounts add-iam-policy-binding \
    ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="user:developer@company.com" \
    --role="roles/iam.serviceAccountUser"
```

**Error: `Invalid service account credentials`**

Causes:
- Service account key expired or revoked
- Service account deleted
- Key leaked and auto-disabled by Google

Solution:
```bash
# Verify service account exists
gcloud iam service-accounts describe ml-sa@PROJECT_ID.iam.gserviceaccount.com

# Check key status
gcloud iam service-accounts keys list \
    --iam-account=ml-sa@PROJECT_ID.iam.gserviceaccount.com
```

### 5.2 Debugging with Cloud Logging

**Query IAM deny audit logs:**
```
resource.type="project"
protoPayload.authorizationInfo.permission=~"aiplatform.*"
protoPayload.authorizationInfo.granted=false
```

**Example log entry:**
```json
{
  "protoPayload": {
    "authorizationInfo": [
      {
        "permission": "aiplatform.customJobs.create",
        "granted": false,
        "resourceAttributes": {}
      }
    ],
    "authenticationInfo": {
      "principalEmail": "ml-training-sa@PROJECT_ID.iam.gserviceaccount.com"
    }
  }
}
```

### 5.3 IAM Policy Troubleshooter

```bash
# Analyze why a principal can/cannot access a resource
gcloud asset analyze-iam-policy \
    --organization=ORGANIZATION_ID \
    --full-resource-name="//storage.googleapis.com/projects/_/buckets/my-bucket" \
    --identity="serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --permissions="storage.objects.get"
```

---

## Sources

**Google Cloud Official Documentation:**
- [Best practices for using service accounts securely](https://docs.cloud.google.com/iam/docs/best-practices-service-accounts) (accessed 2025-02-03)
- [Best practices for managing service account keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03)
- [Vertex AI access control with IAM](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) (accessed 2025-02-03)
- [About Workload Identity Federation for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) (August 30, 2024)
- [Use IAM securely](https://docs.cloud.google.com/iam/docs/using-iam-securely) (accessed 2025-02-03)

**Google Cloud Blog Posts:**
- [Help keep your Google Cloud service account keys safe](https://cloud.google.com/blog/products/identity-security/help-keep-your-google-cloud-service-account-keys-safe) (July 19, 2017)
- [Automatically disabling leaked service account keys](https://cloud.google.com/blog/products/identity-security/automatically-disabling-leaked-service-account-keys-what-you-need-to-know) (May 15, 2024)
- [Move from always-on privileges to on-demand access with Privileged Access Manager](https://cloud.google.com/blog/products/identity-security/move-from-always-on-privileges-to-on-demand-access-with-privileged-access-manager) (June 11, 2024)
- [Scaling the IAM mountain: An in-depth guide to identity in Google Cloud](https://cloud.google.com/blog/products/identity-security/scaling-the-iam-mountain-an-in-depth-guide-to-identity-in-google-cloud) (July 10, 2024)

**Security Research:**
- [Datadog Security Labs: Exploring Google Cloud Default Service Accounts](https://securitylabs.datadoghq.com/articles/google-cloud-default-service-accounts/) (October 29, 2024)
- [Red Canary: The dark cloud around GCP service accounts](https://redcanary.com/blog/threat-detection/gcp-service-accounts/) (accessed 2025-02-03)

**Industry Analysis:**
- Growth Market Reports: Service Account Key Rotation Automation Market (2024) - $1.26B market size
- Dataintelo: Service Account Key Rotation Automation (2024)

**Additional References:**
- [IAM controls for generative AI use cases](https://cloud.google.com/docs/security/security-best-practices-genai/iam-controls) (3 days ago, accessed 2025-02-03)
- [Patterns and practices for identity and access governance on Google Cloud](https://docs.cloud.google.com/architecture/patterns-practices-identity-access-governance-google-cloud) (July 11, 2024)
- [AI and ML perspective: Security](https://docs.cloud.google.com/architecture/framework/perspectives/ai-ml/security) (October 11, 2024)

**Existing Oracle Knowledge:**
- [karpathy/practical-implementation/30-vertex-ai-fundamentals.md](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md) - Basic Vertex AI setup
- [karpathy/practical-implementation/33-vertex-ai-containers.md](../karpathy/practical-implementation/33-vertex-ai-containers.md) - Container IAM patterns
- [karpathy/practical-implementation/69-gke-autopilot-ml-workloads.md](../karpathy/practical-implementation/69-gke-autopilot-ml-workloads.md) - Workload Identity on GKE
- [gcloud-cost/00-billing-automation.md](../gcloud-cost/00-billing-automation.md) - IAM for billing automation

---

**File created**: 2025-02-03
**Lines**: ~710
**Oracle**: karpathy-deep-oracle
**Topic**: GCP IAM & Service Accounts for ML Security
