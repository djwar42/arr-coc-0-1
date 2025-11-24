# Vertex AI IAM, Service Accounts & Security

**Knowledge File: Advanced IAM security patterns for production ML workloads on Vertex AI**

---

## Overview

Identity and Access Management (IAM) is the security foundation for Vertex AI production deployments. While basic service account creation is straightforward, production ML systems require sophisticated IAM patterns: conditional access policies, cross-project permissions, Workload Identity Federation for GKE, comprehensive audit logging, and defense-in-depth security layers.

**Why Advanced IAM Matters for Vertex AI:**
- Training jobs process sensitive datasets (PII, financial, medical records)
- Model artifacts contain intellectual property requiring protection
- Multi-region deployments need cross-project service account delegation
- Compliance requirements (SOC 2, HIPAA, GDPR) mandate audit trails
- GPU/TPU resources cost thousands per hour (unauthorized usage prevention)
- Pipeline orchestration crosses organizational boundaries

From [GCP IAM & Service Accounts for ML Security](../gcloud-iam/00-service-accounts-ml-security.md):
> "Service accounts are the primary way workloads running on Google Cloud access other resources. The principle of least privilege is not a suggestion—it's a requirement."

**Critical Security Update (May 2024):**
Google automatically disables service account keys detected in public repositories within minutes of detection. This makes keyless authentication (Workload Identity, VM-attached service accounts) essential for production.

---

## Section 1: Vertex AI Predefined IAM Roles

### 1.1 Core Vertex AI Roles

From [Vertex AI access control with IAM](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) (accessed 2025-11-16):

**Role Hierarchy and Use Cases:**

| Role | Permissions | Production Use Case |
|------|-------------|---------------------|
| `roles/aiplatform.admin` | Full control over all Vertex AI resources | **Admin only** - Infrastructure setup, not workloads |
| `roles/aiplatform.user` | Create jobs, upload models, manage endpoints | **Training pipelines** - Automated ML workflows |
| `roles/aiplatform.predictor` | Call prediction endpoints (read-only) | **Inference services** - Production serving |
| `roles/aiplatform.viewer` | View jobs, models, endpoints | **Monitoring/observability** - Dashboards, alerts |

**Granular Permissions Breakdown:**

**`roles/aiplatform.user` includes:**
```
aiplatform.customJobs.create
aiplatform.customJobs.get
aiplatform.customJobs.list
aiplatform.customJobs.cancel
aiplatform.models.upload
aiplatform.models.get
aiplatform.models.list
aiplatform.endpoints.create
aiplatform.endpoints.deploy
aiplatform.batchPredictionJobs.create
```

**`roles/aiplatform.predictor` includes:**
```
aiplatform.endpoints.predict
aiplatform.endpoints.get
aiplatform.models.get
```

**Security Best Practice:**
Never grant `roles/aiplatform.admin` to service accounts running workloads. Reserve admin roles for human operators performing infrastructure changes.

### 1.2 Custom Roles for Least Privilege

**Training-Only Custom Role:**
```bash
# Create custom role with minimal training permissions
gcloud iam roles create vertexTrainingOnly \
    --project=PROJECT_ID \
    --title="Vertex AI Training Only" \
    --description="Create and manage training jobs without model deployment" \
    --permissions=aiplatform.customJobs.create,aiplatform.customJobs.get,aiplatform.customJobs.list,aiplatform.customJobs.cancel,aiplatform.models.upload,storage.objects.create,storage.objects.get

# Grant to training service account
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="projects/PROJECT_ID/roles/vertexTrainingOnly"
```

**Inference-Only Custom Role:**
```bash
# Create custom role for serving endpoints only
gcloud iam roles create vertexInferenceOnly \
    --project=PROJECT_ID \
    --title="Vertex AI Inference Only" \
    --description="Call prediction endpoints without training or deployment access" \
    --permissions=aiplatform.endpoints.predict,aiplatform.endpoints.get,aiplatform.models.get

# Grant to inference service account
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-inference-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="projects/PROJECT_ID/roles/vertexInferenceOnly"
```

**Pipeline Orchestrator Custom Role:**
```bash
# Pipeline coordination requires broader permissions
gcloud iam roles create vertexPipelineOrchestrator \
    --project=PROJECT_ID \
    --title="Vertex AI Pipeline Orchestrator" \
    --description="Coordinate multi-step ML pipelines" \
    --permissions=aiplatform.pipelineJobs.create,aiplatform.pipelineJobs.get,aiplatform.pipelineJobs.list,aiplatform.customJobs.create,aiplatform.batchPredictionJobs.create,aiplatform.datasets.create,iam.serviceAccounts.actAs

# Grant to pipeline service account
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-pipeline-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="projects/PROJECT_ID/roles/vertexPipelineOrchestrator"
```

---

## Section 2: Service Account Best Practices

### 2.1 One Service Account Per Workload

**Anti-Pattern: Shared Service Account**
```bash
# ❌ BAD: Single service account for all ML workloads
ml-everything-sa@PROJECT_ID.iam.gserviceaccount.com
  - Granted roles/aiplatform.admin (way too permissive)
  - Used by training, inference, pipelines, experiments
  - If compromised, entire ML infrastructure exposed
```

**Best Practice: Segregated Service Accounts**
```bash
# ✅ GOOD: Separate service accounts by function
ml-training-dev-sa@PROJECT_ID.iam.gserviceaccount.com      # Dev training
ml-training-staging-sa@PROJECT_ID.iam.gserviceaccount.com  # Staging validation
ml-training-prod-sa@PROJECT_ID.iam.gserviceaccount.com     # Production training
ml-inference-prod-sa@PROJECT_ID.iam.gserviceaccount.com    # Production serving
ml-pipeline-orchestrator-sa@PROJECT_ID.iam.gserviceaccount.com  # Pipeline coordination
ml-experiments-sa@PROJECT_ID.iam.gserviceaccount.com       # Experiment tracking
```

**Terraform Example:**
```hcl
# Training service account
resource "google_service_account" "ml_training_prod" {
  account_id   = "ml-training-prod-sa"
  display_name = "Production ML Training Jobs"
  project      = var.project_id
}

# Minimal Vertex AI permissions
resource "google_project_iam_member" "training_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ml_training_prod.email}"
}

# GCS training data bucket (read-only)
resource "google_storage_bucket_iam_member" "training_data_read" {
  bucket = google_storage_bucket.training_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ml_training_prod.email}"
}

# GCS model artifacts bucket (write-only)
resource "google_storage_bucket_iam_member" "model_artifacts_write" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.ml_training_prod.email}"
}

# Inference service account (completely separate)
resource "google_service_account" "ml_inference_prod" {
  account_id   = "ml-inference-prod-sa"
  display_name = "Production ML Inference Endpoints"
  project      = var.project_id
}

# Inference-only permissions (no training access)
resource "google_project_iam_member" "inference_predictor" {
  project = var.project_id
  role    = "roles/aiplatform.predictor"
  member  = "serviceAccount:${google_service_account.ml_inference_prod.email}"
}

# GCS model artifacts (read-only, NO training data access)
resource "google_storage_bucket_iam_member" "inference_model_read" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ml_inference_prod.email}"
}
```

### 2.2 Resource-Level vs Project-Level Permissions

**Anti-Pattern: Project-Wide Permissions**
```bash
# ❌ BAD: Grant storage admin at project level
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
# Problem: Service account can access ALL GCS buckets in project
```

**Best Practice: Bucket-Specific Permissions**
```bash
# ✅ GOOD: Grant access only to specific buckets
# Training data bucket (read-only)
gsutil iam ch serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://training-data-bucket

# Checkpoints bucket (read-write)
gsutil iam ch serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectAdmin \
    gs://training-checkpoints-bucket

# Model artifacts bucket (write-only)
gsutil iam ch serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
    gs://model-artifacts-bucket
```

**BigQuery Dataset-Level Permissions:**
```bash
# Grant read access to specific feature dataset
bq add-iam-policy-binding \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer" \
    PROJECT_ID:feature_store_dataset

# NO access to other datasets in project
```

---

## Section 3: IAM Conditions (Context-Aware Access Control)

### 3.1 Time-Based Access Restrictions

From [Attribute reference for IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-attribute-reference) (accessed 2025-11-16):

**Temporary Developer Access:**
```bash
# Grant temporary elevated access (expires December 31, 2025)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:developer@company.com" \
    --role="roles/aiplatform.admin" \
    --condition='expression=request.time < timestamp("2025-12-31T23:59:59Z"),title=temporary-admin-access,description=Expires end of 2025'
```

**Business Hours Only Access:**
```bash
# Allow service account impersonation only during business hours (UTC)
gcloud iam service-accounts add-iam-policy-binding \
    ml-production-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="user:operator@company.com" \
    --role="roles/iam.serviceAccountUser" \
    --condition='expression=request.time.getHours("UTC") >= 9 && request.time.getHours("UTC") <= 17,title=business-hours-only,description=9am-5pm UTC only'
```

**Scheduled Maintenance Window:**
```bash
# Grant elevated permissions only during maintenance window
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:maintenance-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.admin" \
    --condition='expression=request.time >= timestamp("2025-12-01T02:00:00Z") && request.time <= timestamp("2025-12-01T06:00:00Z"),title=maintenance-window,description=Dec 1 2am-6am UTC'
```

### 3.2 Resource-Based Conditions

**Region Restriction:**
```bash
# Allow model deployment only in us-central1 and us-west1
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-deployment-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user" \
    --condition='expression=resource.name.startsWith("projects/PROJECT_ID/locations/us-central1") || resource.name.startsWith("projects/PROJECT_ID/locations/us-west1"),title=us-regions-only,description=Restrict to US regions'
```

**Environment-Specific Access:**
```bash
# Allow access only to resources with "env:production" label
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:prod-monitoring-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.viewer" \
    --condition='expression=resource.matchTag("env", "production"),title=production-only,description=Only production resources'
```

**Bucket Name Pattern Matching:**
```bash
# Allow GCS access only to buckets matching pattern
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer" \
    --condition='expression=resource.name.startsWith("projects/_/buckets/ml-training-"),title=training-buckets-only,description=Only training data buckets'
```

### 3.3 IP Restriction Conditions

**Important Limitation:**
From GCP documentation and Stack Overflow research (accessed 2025-11-16):
> "Google Cloud Service Account credentials cannot be restricted by location of the user or IP address. IAM conditions do not support IP-based restrictions for service accounts."

**Workaround: VPC Service Controls + Private Google Access**

Instead of IP-based IAM conditions, use VPC Service Controls:

```bash
# Create VPC Service Controls perimeter
gcloud access-context-manager perimeters create ml-production-perimeter \
    --title="ML Production Environment" \
    --resources=projects/PROJECT_NUMBER \
    --restricted-services=aiplatform.googleapis.com,storage.googleapis.com \
    --enable-vpc-accessible-services \
    --vpc-allowed-services=aiplatform.googleapis.com,storage.googleapis.com

# Configure ingress policy (allow only from corporate VPN IP ranges)
gcloud access-context-manager perimeters update ml-production-perimeter \
    --add-ingress-policies=ingress-policy.yaml
```

**ingress-policy.yaml:**
```yaml
- ingressFrom:
    sources:
      - accessLevel: accessPolicies/POLICY_ID/accessLevels/corporate_network
    identities:
      - serviceAccount:ml-training-sa@PROJECT_ID.iam.gserviceaccount.com
  ingressTo:
    resources:
      - projects/PROJECT_NUMBER
    operations:
      - serviceName: aiplatform.googleapis.com
        methodSelectors:
          - method: '*'
```

---

## Section 4: Workload Identity Federation for GKE

### 4.1 Why Workload Identity Matters

From [About Workload Identity Federation for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) (August 30, 2024):

**Traditional Problem: Service Account Keys in Kubernetes**
```yaml
# ❌ OLD WAY: Download service account key, store as Kubernetes secret
apiVersion: v1
kind: Secret
metadata:
  name: gcp-key
type: Opaque
data:
  key.json: <base64-encoded-key>  # Security anti-pattern!

---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: ml-training
    env:
    - name: GOOGLE_APPLICATION_CREDENTIALS
      value: /var/secrets/google/key.json
    volumeMounts:
    - name: gcp-key
      mountPath: /var/secrets/google
  volumes:
  - name: gcp-key
    secret:
      secretName: gcp-key
```

**Problems:**
- Service account keys are long-lived credentials (no automatic rotation)
- Keys stored in etcd (Kubernetes secret storage)
- If pod compromised, attacker gets permanent credentials
- Key rotation requires manual intervention
- No fine-grained per-pod identity

**Workload Identity Solution: Keyless Authentication**

```
Kubernetes Service Account (KSA)
        ↓ (annotated binding)
Google Service Account (GSA)
        ↓ (IAM roles)
Vertex AI API
```

### 4.2 Workload Identity Setup

**Step 1: Enable Workload Identity on GKE Cluster**
```bash
# Create new cluster with Workload Identity
gcloud container clusters create ml-training-cluster \
    --region=us-central1 \
    --workload-pool=PROJECT_ID.svc.id.goog \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --machine-type=n1-standard-8

# Or enable on existing cluster
gcloud container clusters update existing-cluster \
    --region=us-central1 \
    --workload-pool=PROJECT_ID.svc.id.goog
```

**Step 2: Create Google Service Account (GSA)**
```bash
# Create GSA for Vertex AI training
gcloud iam service-accounts create gke-ml-training-sa \
    --display-name="GKE ML Training Workload Identity" \
    --description="Service account for ML training pods on GKE"

# Grant Vertex AI permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:gke-ml-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Grant GCS permissions
gsutil iam ch serviceAccount:gke-ml-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://training-data-bucket
```

**Step 3: Create Kubernetes Service Account (KSA)**
```bash
# Create namespace
kubectl create namespace ml-training

# Create Kubernetes service account
kubectl create serviceaccount ml-training-ksa -n ml-training
```

**Step 4: Bind KSA to GSA**
```bash
# Allow KSA to impersonate GSA
gcloud iam service-accounts add-iam-policy-binding \
    gke-ml-training-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[ml-training/ml-training-ksa]" \
    --role="roles/iam.workloadIdentityUser"

# Annotate KSA with GSA binding
kubectl annotate serviceaccount ml-training-ksa \
    -n ml-training \
    iam.gke.io/gcp-service-account=gke-ml-training-sa@PROJECT_ID.iam.gserviceaccount.com
```

**Step 5: Use in Pod**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
  namespace: ml-training
spec:
  serviceAccountName: ml-training-ksa  # ✅ No keys needed!
  containers:
  - name: training
    image: gcr.io/PROJECT_ID/ml-training:latest
    command:
    - python
    - train.py
    - --dataset=gs://training-data-bucket/data
    - --output=gs://model-artifacts-bucket/models
  nodeSelector:
    iam.gke.io/gke-metadata-server-enabled: "true"
```

**Verification:**
```bash
# Test from inside pod
kubectl exec -it ml-training-pod -n ml-training -- \
    gcloud auth list

# Output should show:
# gke-ml-training-sa@PROJECT_ID.iam.gserviceaccount.com
```

### 4.3 Multi-Environment Workload Identity Pattern

**Production Setup:**
```bash
# Production namespace with production GSA
kubectl create namespace ml-production

kubectl create serviceaccount ml-prod-ksa -n ml-production

gcloud iam service-accounts add-iam-policy-binding \
    ml-production-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[ml-production/ml-prod-ksa]" \
    --role="roles/iam.workloadIdentityUser"

kubectl annotate serviceaccount ml-prod-ksa \
    -n ml-production \
    iam.gke.io/gcp-service-account=ml-production-sa@PROJECT_ID.iam.gserviceaccount.com
```

**Development Setup (Less Privileged):**
```bash
# Dev namespace with dev GSA (restricted permissions)
kubectl create namespace ml-dev

kubectl create serviceaccount ml-dev-ksa -n ml-dev

gcloud iam service-accounts add-iam-policy-binding \
    ml-dev-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[ml-dev/ml-dev-ksa]" \
    --role="roles/iam.workloadIdentityUser"

kubectl annotate serviceaccount ml-dev-ksa \
    -n ml-dev \
    iam.gke.io/gcp-service-account=ml-dev-sa@PROJECT_ID.iam.gserviceaccount.com
```

---

## Section 5: Cross-Project Service Account Usage

### 5.1 Training in Project A, Data in Project B

**Scenario:**
- Training jobs run in `ml-training-project`
- Training data stored in `data-lake-project`
- Model artifacts stored in `ml-artifacts-project`

**Setup:**

**Step 1: Create Service Account in Training Project**
```bash
# In ml-training-project
gcloud iam service-accounts create cross-project-training-sa \
    --project=ml-training-project \
    --display-name="Cross-Project Training Service Account"

# Grant Vertex AI permissions in training project
gcloud projects add-iam-policy-binding ml-training-project \
    --member="serviceAccount:cross-project-training-sa@ml-training-project.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

**Step 2: Grant Data Access in Data Project**
```bash
# In data-lake-project (data owner grants access)
gsutil iam ch serviceAccount:cross-project-training-sa@ml-training-project.iam.gserviceaccount.com:objectViewer \
    gs://data-lake-project-training-data
```

**Step 3: Grant Model Write Access in Artifacts Project**
```bash
# In ml-artifacts-project
gsutil iam ch serviceAccount:cross-project-training-sa@ml-training-project.iam.gserviceaccount.com:objectCreator \
    gs://ml-artifacts-project-models
```

**Step 4: Launch Training Job**
```bash
# From ml-training-project
gcloud ai custom-jobs create \
    --region=us-central1 \
    --project=ml-training-project \
    --display-name=cross-project-training \
    --service-account=cross-project-training-sa@ml-training-project.iam.gserviceaccount.com \
    --config=job.yaml
```

**job.yaml:**
```yaml
workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-8
      acceleratorType: NVIDIA_TESLA_V100
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/ml-training-project/training:latest
      args:
      - --training-data=gs://data-lake-project-training-data/dataset.tfrecords
      - --output-model=gs://ml-artifacts-project-models/model-v1
```

### 5.2 Service Account Impersonation for Cross-Project Access

**Developer Workflow:**
```bash
# Developer in project A needs to test with data in project B
# Instead of granting direct access, allow impersonation

# In data-lake-project
gcloud iam service-accounts add-iam-policy-binding \
    data-access-sa@data-lake-project.iam.gserviceaccount.com \
    --member="user:developer@company.com" \
    --role="roles/iam.serviceAccountTokenCreator" \
    --condition='expression=request.time < timestamp("2025-12-31T23:59:59Z"),title=temporary-dev-access'

# Developer impersonates service account
gcloud auth application-default login \
    --impersonate-service-account=data-access-sa@data-lake-project.iam.gserviceaccount.com

# Now developer can access data via impersonated SA
gsutil ls gs://data-lake-project-training-data/
```

---

## Section 6: Cloud Audit Logs

### 6.1 Admin Activity vs Data Access Logs

From [Cloud Audit Logs overview](https://docs.cloud.google.com/logging/docs/audit) (accessed 2025-11-16):

**Four Types of Audit Logs:**

1. **Admin Activity Logs** (always enabled, free)
   - Who created/deleted/modified resources
   - Training job creation, model deployment, endpoint updates
   - **Cannot be disabled**

2. **Data Access Logs** (disabled by default, billable)
   - Who read/wrote user data
   - Training data reads, model predictions, checkpoint writes
   - **Must be explicitly enabled**

3. **System Event Logs** (always enabled, free)
   - GCP-initiated actions (VM migrations, automatic scaling)

4. **Policy Denied Logs** (always enabled, free)
   - IAM permission denials (helpful for debugging)

**Admin Activity Log Example:**
```json
{
  "protoPayload": {
    "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
    "serviceName": "aiplatform.googleapis.com",
    "methodName": "google.cloud.aiplatform.v1.JobService.CreateCustomJob",
    "authenticationInfo": {
      "principalEmail": "ml-training-sa@PROJECT_ID.iam.gserviceaccount.com"
    },
    "authorizationInfo": [
      {
        "permission": "aiplatform.customJobs.create",
        "granted": true,
        "resourceAttributes": {
          "name": "projects/PROJECT_ID/locations/us-central1"
        }
      }
    ],
    "request": {
      "parent": "projects/PROJECT_ID/locations/us-central1",
      "customJob": {
        "displayName": "training-job-001"
      }
    }
  },
  "insertId": "abc123",
  "resource": {
    "type": "aiplatform.googleapis.com/CustomJob",
    "labels": {
      "project_id": "PROJECT_ID",
      "location": "us-central1"
    }
  },
  "timestamp": "2025-11-16T14:30:00Z",
  "severity": "NOTICE"
}
```

**Data Access Log Example:**
```json
{
  "protoPayload": {
    "@type": "type.googleapis.com/google.cloud.audit.AuditLog",
    "serviceName": "aiplatform.googleapis.com",
    "methodName": "google.cloud.aiplatform.v1.PredictionService.Predict",
    "authenticationInfo": {
      "principalEmail": "ml-inference-sa@PROJECT_ID.iam.gserviceaccount.com"
    },
    "authorizationInfo": [
      {
        "permission": "aiplatform.endpoints.predict",
        "granted": true,
        "resourceAttributes": {
          "name": "projects/PROJECT_ID/locations/us-central1/endpoints/123456"
        }
      }
    ],
    "resourceName": "projects/PROJECT_ID/locations/us-central1/endpoints/123456",
    "numResponseItems": "1"
  },
  "insertId": "def456",
  "resource": {
    "type": "aiplatform.googleapis.com/Endpoint",
    "labels": {
      "project_id": "PROJECT_ID",
      "location": "us-central1",
      "endpoint_id": "123456"
    }
  },
  "timestamp": "2025-11-16T14:35:00Z",
  "severity": "INFO"
}
```

### 6.2 Enabling Data Access Logs

**Enable for Vertex AI:**
```bash
# Get current IAM policy
gcloud projects get-iam-policy PROJECT_ID > policy.yaml

# Edit policy.yaml to add audit config
cat >> policy.yaml << 'EOF'
auditConfigs:
- auditLogConfigs:
  - logType: ADMIN_READ
  - logType: DATA_READ
  - logType: DATA_WRITE
  service: aiplatform.googleapis.com
EOF

# Apply updated policy
gcloud projects set-iam-policy PROJECT_ID policy.yaml
```

**Enable for Cloud Storage (Training Data Access Tracking):**
```bash
# Add to policy.yaml
cat >> policy.yaml << 'EOF'
- auditLogConfigs:
  - logType: ADMIN_READ
  - logType: DATA_READ
  - logType: DATA_WRITE
  service: storage.googleapis.com
EOF

# Apply
gcloud projects set-iam-policy PROJECT_ID policy.yaml
```

**Terraform Configuration:**
```hcl
resource "google_project_iam_audit_config" "vertex_ai_audit" {
  project = var.project_id
  service = "aiplatform.googleapis.com"

  audit_log_config {
    log_type = "ADMIN_READ"
  }

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

resource "google_project_iam_audit_config" "storage_audit" {
  project = var.project_id
  service = "storage.googleapis.com"

  audit_log_config {
    log_type = "ADMIN_READ"
  }

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}
```

### 6.3 Querying Audit Logs

**Who Created This Training Job?**
```
resource.type="aiplatform.googleapis.com/CustomJob"
protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob"
```

**Who Called This Prediction Endpoint?**
```
resource.type="aiplatform.googleapis.com/Endpoint"
protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"
resource.labels.endpoint_id="123456"
```

**Who Read Training Data from GCS?**
```
resource.type="gcs_bucket"
protoPayload.methodName="storage.objects.get"
resource.labels.bucket_name="training-data-bucket"
protoPayload.authenticationInfo.principalEmail=~".*-sa@.*"
```

**Permission Denials (Troubleshooting):**
```
resource.type="aiplatform.googleapis.com/CustomJob"
protoPayload.authorizationInfo.granted=false
```

**Export to BigQuery for Analysis:**
```bash
# Create log sink
gcloud logging sinks create vertex-ai-audit-sink \
    bigquery.googleapis.com/projects/PROJECT_ID/datasets/audit_logs \
    --log-filter='resource.type=("aiplatform.googleapis.com/CustomJob" OR "aiplatform.googleapis.com/Endpoint")'

# Query in BigQuery
SELECT
  timestamp,
  protopayload_auditlog.authenticationInfo.principalEmail,
  protopayload_auditlog.methodName,
  protopayload_auditlog.resourceName
FROM `PROJECT_ID.audit_logs.cloudaudit_googleapis_com_activity_*`
WHERE DATE(_PARTITIONTIME) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 100
```

### 6.4 Audit Log Retention and Compliance

**Default Retention:**
- Admin Activity: 400 days
- Data Access: 30 days
- System Event: 400 days

**Long-Term Retention for Compliance:**
```bash
# Export to Cloud Storage for 7-year retention (HIPAA/SOX)
gcloud logging sinks create compliance-audit-archive \
    storage.googleapis.com/compliance-audit-logs-bucket \
    --log-filter='logName:"cloudaudit.googleapis.com"'

# Configure bucket lifecycle for compliance
gsutil lifecycle set lifecycle.json gs://compliance-audit-logs-bucket
```

**lifecycle.json:**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "ARCHIVE"
        },
        "condition": {
          "age": 90,
          "matchesPrefix": ["cloudaudit.googleapis.com"]
        }
      },
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 2555,
          "matchesPrefix": ["cloudaudit.googleapis.com"]
        }
      }
    ]
  }
}
```

---

## Section 7: arr-coc-0-1 Security Configuration

### 7.1 Production Security Setup

From [arr-coc-0-1 project structure](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):

**Service Account Architecture:**

```bash
# Training service account (minimal permissions)
gcloud iam service-accounts create arr-coc-training-sa \
    --project=PROJECT_ID \
    --display-name="ARR-COC Training Jobs" \
    --description="Service account for ARR-COC-VIS training workloads"

# Grant Vertex AI training permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# GCS training data (ImageNet, COCO) - read-only
gsutil iam ch serviceAccount:arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://arr-coc-training-data

# GCS checkpoints - read-write
gsutil iam ch serviceAccount:arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectAdmin \
    gs://arr-coc-checkpoints

# GCS model artifacts - write-only
gsutil iam ch serviceAccount:arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com:objectCreator \
    gs://arr-coc-models

# W&B API key (Secret Manager)
gcloud secrets add-iam-policy-binding wandb-api-key \
    --member="serviceAccount:arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**Inference Service Account (Separate):**
```bash
# Inference endpoint service account
gcloud iam service-accounts create arr-coc-inference-sa \
    --project=PROJECT_ID \
    --display-name="ARR-COC Inference Endpoints" \
    --description="Service account for ARR-COC-VIS prediction endpoints"

# Inference-only permissions (no training access)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.predictor"

# GCS model artifacts - read-only (NO training data or checkpoint access)
gsutil iam ch serviceAccount:arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://arr-coc-models
```

### 7.2 arr-coc-0-1 Audit Logging Configuration

**Enable Comprehensive Audit Logs:**
```bash
# Enable Data Access logs for arr-coc-0-1 project
cat > arr-coc-audit-config.yaml << 'EOF'
auditConfigs:
- auditLogConfigs:
  - logType: ADMIN_READ
  - logType: DATA_READ
  - logType: DATA_WRITE
  service: aiplatform.googleapis.com
- auditLogConfigs:
  - logType: ADMIN_READ
  - logType: DATA_READ
  - logType: DATA_WRITE
  service: storage.googleapis.com
- auditLogConfigs:
  - logType: ADMIN_READ
  service: secretmanager.googleapis.com
EOF

# Apply audit configuration
gcloud projects get-iam-policy PROJECT_ID > current-policy.yaml
cat arr-coc-audit-config.yaml >> current-policy.yaml
gcloud projects set-iam-policy PROJECT_ID current-policy.yaml
```

**Monitor Training Job Creation:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob"
protoPayload.authenticationInfo.principalEmail="arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com"
```

**Monitor Model Predictions:**
```
resource.type="aiplatform.googleapis.com/Endpoint"
protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"
protoPayload.authenticationInfo.principalEmail="arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com"
```

**Monitor Training Data Access:**
```
resource.type="gcs_bucket"
resource.labels.bucket_name="arr-coc-training-data"
protoPayload.methodName="storage.objects.get"
protoPayload.authenticationInfo.principalEmail="arr-coc-training-sa@PROJECT_ID.iam.gserviceaccount.com"
```

### 7.3 arr-coc-0-1 Workload Identity (GKE Deployment)

**If arr-coc-0-1 deployed on GKE:**

```bash
# Create GKE cluster with Workload Identity
gcloud container clusters create arr-coc-cluster \
    --region=us-central1 \
    --workload-pool=PROJECT_ID.svc.id.goog \
    --machine-type=n1-standard-8 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5

# Create Kubernetes namespace
kubectl create namespace arr-coc

# Create KSA
kubectl create serviceaccount arr-coc-inference-ksa -n arr-coc

# Bind KSA to GSA
gcloud iam service-accounts add-iam-policy-binding \
    arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[arr-coc/arr-coc-inference-ksa]" \
    --role="roles/iam.workloadIdentityUser"

# Annotate KSA
kubectl annotate serviceaccount arr-coc-inference-ksa \
    -n arr-coc \
    iam.gke.io/gcp-service-account=arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com

# Deploy arr-coc-0-1 inference
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arr-coc-inference
  namespace: arr-coc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arr-coc
  template:
    metadata:
      labels:
        app: arr-coc
    spec:
      serviceAccountName: arr-coc-inference-ksa  # Workload Identity binding
      containers:
      - name: inference
        image: gcr.io/PROJECT_ID/arr-coc-inference:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: gs://arr-coc-models/model-v1
      nodeSelector:
        iam.gke.io/gke-metadata-server-enabled: "true"
EOF
```

### 7.4 arr-coc-0-1 Security Monitoring Dashboard

**Cloud Monitoring Query:**
```sql
-- Unauthorized access attempts
resource.type="aiplatform.googleapis.com/Endpoint"
protoPayload.authorizationInfo.granted=false
protoPayload.authenticationInfo.principalEmail=~"arr-coc.*"

-- Anomalous prediction volume
resource.type="aiplatform.googleapis.com/Endpoint"
protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"
protoPayload.authenticationInfo.principalEmail="arr-coc-inference-sa@PROJECT_ID.iam.gserviceaccount.com"
```

**Alerting Policy:**
```bash
# Alert on permission denials
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="ARR-COC IAM Permission Denials" \
    --condition-display-name="Permission denied attempts" \
    --condition-expression='
      resource.type = "aiplatform.googleapis.com/Endpoint"
      AND protoPayload.authorizationInfo.granted = false
      AND protoPayload.authenticationInfo.principalEmail =~ "arr-coc.*"
    ' \
    --condition-threshold-value=5 \
    --condition-threshold-duration=300s
```

---

## Section 8: Security Best Practices Summary

### 8.1 Least Privilege Checklist

**✅ Service Account Hygiene:**
- [ ] One service account per workload (training, inference, pipelines)
- [ ] Never use default Compute Engine service account
- [ ] Grant resource-level permissions (bucket/dataset) not project-level
- [ ] Use custom roles for fine-grained permissions
- [ ] No `roles/owner` or `roles/editor` for service accounts

**✅ Key Management:**
- [ ] Avoid service account keys entirely (use Workload Identity, VM-attached SA)
- [ ] If keys required, rotate every 90 days
- [ ] Store keys in Secret Manager, never in code/config
- [ ] Monitor for leaked keys (Google auto-disables public keys)

**✅ Access Control:**
- [ ] Use IAM conditions for time-based, resource-based restrictions
- [ ] Implement VPC Service Controls for IP-based restrictions
- [ ] Enable service account impersonation for developers (not direct keys)
- [ ] Require MFA for human users with elevated permissions

**✅ Audit and Compliance:**
- [ ] Enable Admin Activity logs (always on, free)
- [ ] Enable Data Access logs for sensitive workloads
- [ ] Export audit logs to BigQuery for analysis
- [ ] Configure long-term retention for compliance (7 years for HIPAA)
- [ ] Monitor permission denials for security incidents

**✅ Workload Identity (GKE):**
- [ ] Enable Workload Identity on all GKE clusters
- [ ] Create separate KSA for each namespace/workload
- [ ] Bind KSA to GSA with minimal permissions
- [ ] Never download service account keys to pods

### 8.2 Common Security Mistakes

**❌ Mistake 1: Project-Wide Storage Permissions**
```bash
# BAD
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

**✅ Fix: Bucket-Specific Permissions**
```bash
# GOOD
gsutil iam ch serviceAccount:ml-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer \
    gs://specific-training-bucket
```

**❌ Mistake 2: Shared Service Account**
```bash
# BAD: Same SA for training and inference
ml-everything-sa@PROJECT_ID.iam.gserviceaccount.com
```

**✅ Fix: Separate Service Accounts**
```bash
# GOOD: Different SAs with different permissions
ml-training-sa@PROJECT_ID.iam.gserviceaccount.com    # aiplatform.user
ml-inference-sa@PROJECT_ID.iam.gserviceaccount.com   # aiplatform.predictor
```

**❌ Mistake 3: Service Account Keys in Git**
```bash
# BAD
git add service-account-key.json
git commit -m "Add credentials"
```

**✅ Fix: Use Workload Identity or Secret Manager**
```bash
# GOOD
gcloud secrets create sa-key --data-file=service-account-key.json
# Then delete local file
rm service-account-key.json
```

**❌ Mistake 4: No Audit Logging**
```bash
# BAD: Default configuration (Data Access logs disabled)
```

**✅ Fix: Enable Data Access Logs**
```bash
# GOOD
gcloud projects get-iam-policy PROJECT_ID > policy.yaml
# Add auditConfigs for aiplatform.googleapis.com
gcloud projects set-iam-policy PROJECT_ID policy.yaml
```

---

## Sources

**Google Cloud Official Documentation:**
- [Vertex AI access control with IAM](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) (accessed 2025-11-16)
- [Attribute reference for IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-attribute-reference) (accessed 2025-11-16)
- [Overview of IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-overview) (accessed 2025-11-16)
- [About Workload Identity Federation for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) (August 30, 2024)
- [Authenticate to Google Cloud APIs from GKE workloads](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) (accessed 2025-11-16)
- [Cloud Audit Logs overview](https://docs.cloud.google.com/logging/docs/audit) (accessed 2025-11-16)
- [Best practices for using service accounts securely](https://docs.cloud.google.com/iam/docs/best-practices-service-accounts) (accessed 2025-11-16)

**Web Research (Accessed 2025-11-16):**
- Google Cloud Blog: "Automatically disabling leaked service account keys" (May 15, 2024)
- Stack Overflow: "GCP: IP address restriction to use service account" - Confirms IAM conditions don't support IP restrictions
- Medium: "Why You Need to Enable Audit Logs in Google Cloud" (October 2024)
- CyberArk Developer: "GKE Workload Identity Federation for Kubernetes Principals" (August 10, 2024)
- GitHub: salrashid123/k8s_federation_with_gcp - Workload Identity implementation patterns
- DoiT: "GKE Workload Identity is now named Workload Identity Federation" (May 13, 2024)

**Existing Oracle Knowledge:**
- [gcloud-iam/00-service-accounts-ml-security.md](../gcloud-iam/00-service-accounts-ml-security.md) - Service account fundamentals
- [practical-implementation/30-vertex-ai-fundamentals.md](../practical-implementation/30-vertex-ai-fundamentals.md) - Basic Vertex AI setup
- [practical-implementation/69-gke-autopilot-ml-workloads.md](../practical-implementation/69-gke-autopilot-ml-workloads.md) - Workload Identity on GKE

**arr-coc-0-1 Project:**
- [arr-coc-0-1 CLAUDE.md](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/CLAUDE.md) - Project-specific security requirements
- [arr-coc-0-1 training/cli.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/cli.py) - Service account usage patterns

---

**File created**: 2025-11-16
**Lines**: ~700
**Oracle**: karpathy-deep-oracle
**Topic**: Vertex AI IAM, Service Accounts & Security
