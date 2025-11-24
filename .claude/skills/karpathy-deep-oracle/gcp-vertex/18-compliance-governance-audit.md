# Vertex AI Compliance, Governance, and Audit

Comprehensive guide to compliance certifications, governance frameworks, and audit capabilities for Vertex AI deployments. Covers regulatory requirements, data residency, approval workflows, and lineage tracking.

---

## Overview

Vertex AI provides enterprise-grade compliance and governance capabilities to meet regulatory requirements across healthcare (HIPAA), financial services (PCI-DSS), and general enterprise security (SOC 2, ISO 27001). This guide covers compliance certifications, data residency controls, model governance workflows, and audit reporting.

**Key compliance areas:**
- Compliance certifications (SOC 2, ISO 27001, HIPAA, PCI-DSS)
- Data residency and regional endpoints
- Model approval workflows and deployment gates
- Metadata lineage tracking and provenance
- Organization Policy constraints
- Compliance reporting and audit dashboards

---

## 1. Compliance Certifications

### SOC 2 (Service Organization Control 2)

**What it covers:**
- Security, availability, processing integrity, confidentiality, privacy
- Third-party attestation of security controls
- Annual audits by independent auditors

**Vertex AI SOC 2 compliance:**
```yaml
# Services covered under SOC 2:
services:
  - Vertex AI Training (Custom Jobs, Pipelines)
  - Vertex AI Prediction (Online, Batch)
  - Vertex AI Workbench
  - Vertex AI Model Registry
  - Vertex AI Feature Store
  - Vertex AI Matching Engine
  - Vertex AI Experiments
  - Vertex AI Metadata Store

# SOC 2 reports available:
reports:
  - SOC 2 Type I: Point-in-time assessment
  - SOC 2 Type II: Controls operating effectiveness over time
  - Available through Google Cloud Compliance Reports Manager
```

**Requesting SOC 2 reports:**
```bash
# Access via Google Cloud Console:
# 1. Navigate to: Security > Compliance Reports Manager
# 2. Download SOC 2 Type II report
# 3. Share with auditors under NDA

# Or request via support:
gcloud support cases create \
  --display-name="SOC 2 Report Request" \
  --description="Requesting latest SOC 2 Type II report for Vertex AI" \
  --severity=S3
```

From [Google Cloud Compliance](https://cloud.google.com/security/compliance) (accessed 2025-11-16):
- SOC 2 Type II reports available quarterly
- Covers all Vertex AI services in scope
- Independent auditor attestations

---

### ISO 27001 (Information Security Management)

**Certification scope:**
- International standard for information security management systems (ISMS)
- Risk assessment and treatment processes
- Continuous improvement framework

**Vertex AI ISO 27001:**
```yaml
# ISO 27001 certification:
scope:
  - All Vertex AI services
  - Global infrastructure
  - Annual surveillance audits

controls:
  - Access control (IAM, service accounts)
  - Cryptography (data at rest, in transit)
  - Physical security (data centers)
  - Operations security (change management)
  - Communications security (network isolation)
  - System acquisition and maintenance
  - Incident management

# Certificate details:
certificate_authority: "BSI (British Standards Institution)"
standard_version: "ISO/IEC 27001:2013"
validity: "3 years with annual surveillance"
```

**Accessing ISO 27001 certificate:**
```bash
# Download from Google Cloud Trust Center:
# https://cloud.google.com/security/compliance/iso-27001

# Certificate includes:
# - Scope of certification
# - Certified services (Vertex AI included)
# - Validity period
# - Certification body details
```

From [Google Cloud ISO 27001](https://cloud.google.com/security/compliance/iso-27001) (accessed 2025-11-16):
- Certificate updated annually
- Covers global infrastructure
- Vertex AI services in scope since 2021

---

### HIPAA (Health Insurance Portability and Accountability Act)

**HIPAA compliance for healthcare:**
- Protected Health Information (PHI) handling
- Business Associate Agreement (BAA) required
- Technical safeguards for ePHI

**Vertex AI HIPAA-eligible services:**
```yaml
# HIPAA-covered services:
eligible_services:
  - Vertex AI Training (Custom Jobs)
  - Vertex AI Prediction (Online, Batch)
  - Vertex AI Workbench (Managed Notebooks)
  - Vertex AI Pipelines
  - Cloud Storage (for datasets)
  - BigQuery (for data processing)

# NOT covered under BAA:
excluded_services:
  - AutoML (some features)
  - Pre-trained models (Gemini, PaLM via public endpoints)
  - Generative AI features (unless using private endpoints)

# BAA requirements:
baa:
  required: true
  process: "Sign BAA before processing PHI"
  contact: "Google Cloud sales team"
  scope: "Specific GCP services listed in BAA appendix"
```

**HIPAA deployment pattern:**
```python
# Example: HIPAA-compliant training job
from google.cloud import aiplatform

# Initialize with HIPAA-compliant region
aiplatform.init(
    project="healthcare-ml-project",
    location="us-central1",  # HIPAA-eligible region
    staging_bucket="gs://phi-datasets-bucket"  # BAA-covered bucket
)

# Custom training job with HIPAA safeguards
job = aiplatform.CustomTrainingJob(
    display_name="hipaa-compliant-training",
    container_uri="us-docker.pkg.dev/healthcare-ml/training:latest",
    # Requirements:
    # 1. BAA signed with Google Cloud
    # 2. Use HIPAA-eligible services only
    # 3. Enable encryption (CMEK optional but recommended)
    # 4. Audit logging enabled
    # 5. Access controls (IAM) configured
)

# Deploy with encryption and access controls
job.run(
    training_encryption_spec_key_name="projects/PROJECT/locations/us-central1/keyRings/KEYRING/cryptoKeys/KEY",
    service_account="hipaa-training-sa@PROJECT.iam.gserviceaccount.com"
)
```

**HIPAA checklist:**
```bash
# 1. Sign BAA
# Contact Google Cloud sales to execute BAA

# 2. Enable audit logging
gcloud logging sinks create hipaa-audit-sink \
  cloud-logging://logging.googleapis.com/projects/PROJECT/locations/us-central1/buckets/hipaa-audit-logs \
  --log-filter='resource.type="aiplatform.googleapis.com"'

# 3. Configure CMEK (Customer-Managed Encryption Keys)
gcloud kms keyrings create hipaa-keyring \
  --location=us-central1

gcloud kms keys create hipaa-key \
  --location=us-central1 \
  --keyring=hipaa-keyring \
  --purpose=encryption

# 4. Restrict access with IAM
gcloud projects add-iam-policy-binding PROJECT \
  --member="serviceAccount:hipaa-training-sa@PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user" \
  --condition="resource.name.startsWith('projects/PROJECT/locations/us-central1')"

# 5. Enable VPC Service Controls (recommended)
gcloud access-context-manager perimeters create hipaa-perimeter \
  --title="HIPAA Perimeter" \
  --resources="projects/PROJECT_NUMBER" \
  --restricted-services="aiplatform.googleapis.com"
```

From [Google Cloud HIPAA Compliance](https://cloud.google.com/security/compliance/hipaa) (accessed 2025-11-16):
- BAA required before processing PHI
- Specific Vertex AI services are HIPAA-eligible
- Customer responsible for HIPAA compliance implementation

---

### PCI-DSS (Payment Card Industry Data Security Standard)

**PCI-DSS for payment processing:**
- Cardholder data protection
- Level 1 Service Provider certification
- Annual attestation of compliance

**Vertex AI PCI-DSS considerations:**
```yaml
# PCI-DSS compliance scope:
vertex_ai_usage:
  # Vertex AI is PCI-DSS certified infrastructure
  # BUT: You cannot store cardholder data in Vertex AI

  allowed_use_cases:
    - Fraud detection models (using tokenized data)
    - Risk scoring (without CHD/SAD)
    - Pattern analysis (anonymized transactions)

  prohibited:
    - Storing Primary Account Numbers (PAN)
    - Storing Card Verification Values (CVV)
    - Storing unencrypted cardholder data

# Tokenization pattern:
recommended_approach:
  - Tokenize cardholder data before ML processing
  - Use token vaults (separate PCI-DSS compliant system)
  - Train models on tokens, not actual card numbers
  - Deploy models in PCI-DSS scoped environment
```

**PCI-DSS compliant ML workflow:**
```python
# Example: Fraud detection without cardholder data
from google.cloud import aiplatform
import hashlib

# Step 1: Tokenization (done in PCI-DSS vault, not shown)
# Step 2: Train model on tokenized features

def create_features(transaction):
    """Create ML features without CHD/SAD."""
    return {
        "amount": transaction["amount"],
        "merchant_id_hash": hashlib.sha256(transaction["merchant_id"].encode()).hexdigest(),
        "transaction_time": transaction["timestamp"],
        "location_hash": hashlib.sha256(transaction["location"].encode()).hexdigest(),
        # NO card numbers, CVV, or cardholder names
    }

# Train fraud detection model
job = aiplatform.CustomTrainingJob(
    display_name="fraud-detection-pci-compliant",
    container_uri="gcr.io/fraud-detection/trainer:latest"
)

# Model operates on tokenized/hashed features only
model = job.run(
    dataset="gs://tokenized-transactions/train.csv",
    # No cardholder data in training set
)

# Deploy in PCI-DSS scoped environment
endpoint = model.deploy(
    machine_type="n1-standard-4",
    service_account="pci-scoped-sa@PROJECT.iam.gserviceaccount.com"
)
```

**PCI-DSS compliance controls:**
```bash
# 1. Network segmentation (VPC isolation)
gcloud compute networks create pci-scoped-vpc \
  --subnet-mode=custom

gcloud compute networks subnets create pci-subnet \
  --network=pci-scoped-vpc \
  --region=us-central1 \
  --range=10.0.1.0/24

# 2. Firewall rules (deny all by default)
gcloud compute firewall-rules create deny-all-ingress \
  --network=pci-scoped-vpc \
  --action=deny \
  --rules=all \
  --direction=ingress \
  --priority=1000

# 3. Logging and monitoring
gcloud logging sinks create pci-audit-sink \
  cloud-logging://logging.googleapis.com/projects/PROJECT/locations/us-central1/buckets/pci-audit-logs \
  --log-filter='resource.type="aiplatform.googleapis.com" AND severity>=WARNING'

# 4. Quarterly vulnerability scans (use GCP Security Command Center)
gcloud scc findings list PROJECT \
  --source="SECURITY_COMMAND_CENTER_SOURCE" \
  --filter="category='Vulnerability'"
```

From [Google Cloud PCI-DSS](https://cloud.google.com/security/compliance/pci-dss) (accessed 2025-11-16):
- GCP infrastructure is PCI-DSS Level 1 certified
- Customers responsible for cardholder data handling
- Vertex AI can be used in PCI-scoped environments with proper controls

---

## 2. Data Residency and Regional Endpoints

### Data Residency Requirements

**Why data residency matters:**
- Legal requirements (GDPR, data sovereignty laws)
- Customer contractual obligations
- Industry regulations (financial services, healthcare)

**Vertex AI data residency guarantees:**
```yaml
# Data residency commitments:
guarantees:
  training_data:
    at_rest: "Stored only in specified region"
    in_transit: "May traverse Google's global network"
    processing: "Compute occurs in specified region"

  model_artifacts:
    location: "Stored in region-specific GCS bucket"
    replication: "Optional multi-region (customer controlled)"

  metadata:
    location: "Stored in region-specific Metadata Store"

  logs:
    location: "Cloud Logging (regional or multi-regional bucket)"

# Regional endpoints:
endpoints:
  us_central1: "us-central1-aiplatform.googleapis.com"
  europe_west4: "europe-west4-aiplatform.googleapis.com"
  asia_southeast1: "asia-southeast1-aiplatform.googleapis.com"
  # Global endpoint (routes to nearest region):
  global: "aiplatform.googleapis.com"  # NOT recommended for data residency
```

**Using regional endpoints:**
```python
# Example: EU data residency
from google.cloud import aiplatform

# CORRECT: Specify region explicitly
aiplatform.init(
    project="eu-ml-project",
    location="europe-west4",  # EU region
    staging_bucket="gs://eu-datasets-bucket"  # Bucket in europe-west4
)

# Use regional endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="eu-compliant-endpoint",
    # Data stays in EU
)

# INCORRECT: Using global endpoint
# This may route data through other regions!
# aiplatform.init(location="global")  # ❌ Violates data residency
```

**Regional endpoint configuration:**
```bash
# Set regional endpoint in gcloud
gcloud config set ai/region europe-west4

# Verify regional endpoint
gcloud ai endpoints list --region=europe-west4

# Training job with explicit region
gcloud ai custom-jobs create \
  --region=europe-west4 \
  --display-name=eu-training \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/image:latest

# Ensure logs stay in region
gcloud logging buckets create eu-logs \
  --location=europe-west4 \
  --retention-days=365
```

---

### EU Data Residency (GDPR Compliance)

**GDPR requirements:**
- Data must stay within EU/EEA
- Data processing agreements (DPAs)
- Right to erasure, data portability

**EU-compliant Vertex AI setup:**
```yaml
# EU regions for Vertex AI:
eu_regions:
  - europe-west1 (Belgium)
  - europe-west4 (Netherlands)
  - europe-west9 (Paris, France)
  - europe-north1 (Finland)

# GDPR-compliant configuration:
configuration:
  region: "europe-west4"
  data_storage: "EU-only buckets"
  processing: "EU compute zones"
  logs: "EU Cloud Logging buckets"

# DPA (Data Processing Agreement):
dpa:
  required: true
  google_terms: "Automatically covers GDPR requirements"
  controller_processor: "Customer is controller, Google is processor"
```

**GDPR compliance code:**
```python
# Example: GDPR-compliant training
from google.cloud import aiplatform, storage

# Initialize for EU
aiplatform.init(
    project="gdpr-ml-project",
    location="europe-west4"
)

# Training job with EU data residency
job = aiplatform.CustomTrainingJob(
    display_name="gdpr-compliant-training",
    container_uri="europe-docker.pkg.dev/gdpr-ml/trainer:latest"
)

model = job.run(
    dataset="gs://eu-gdpr-datasets/train/",
    replica_count=1,
    machine_type="n1-standard-4",
    # All data stays in EU
)

# Right to erasure implementation
def delete_user_data(user_id):
    """GDPR Article 17: Right to erasure."""
    # Delete training data
    storage_client = storage.Client()
    bucket = storage_client.bucket("eu-gdpr-datasets")
    blobs = bucket.list_blobs(prefix=f"users/{user_id}/")
    bucket.delete_blobs(blobs)

    # Delete predictions (if stored)
    # Delete from Feature Store
    # Delete from Metadata Store

    # Log deletion for audit trail
    print(f"User {user_id} data deleted per GDPR Article 17")
```

From [Vertex AI Data Residency](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/data-residency) (accessed 2025-11-16):
- Regional endpoints guarantee data residency
- EU regions support GDPR compliance
- Customer responsible for configuring correct regions

---

### Multi-Region vs Single-Region

**Trade-offs:**
```yaml
# Single-region deployment:
single_region:
  pros:
    - Data residency compliance
    - Predictable latency
    - Simpler compliance audits
  cons:
    - Single point of failure
    - No automatic failover
    - Limited disaster recovery

# Multi-region deployment:
multi_region:
  pros:
    - High availability
    - Disaster recovery
    - Lower latency (geo-distributed)
  cons:
    - Data residency challenges
    - Cross-region data transfer costs
    - Complex compliance (data may leave region)
```

**Hybrid approach (compliance + availability):**
```python
# Example: Multi-region with data residency controls
from google.cloud import aiplatform

# Primary region (EU for GDPR)
aiplatform.init(
    project="hybrid-ml-project",
    location="europe-west4"
)

# Train in EU (data residency)
model_eu = aiplatform.CustomTrainingJob(
    display_name="eu-primary-training",
    container_uri="europe-docker.pkg.dev/trainer:latest"
).run(dataset="gs://eu-datasets/train/")

# Replicate model (not data) to US for low-latency serving
model_eu.upload(
    display_name="eu-model-replicated-to-us",
    serving_container_image_uri="us-docker.pkg.dev/serve:latest"
)

# Deploy in US (model only, no training data)
aiplatform.init(location="us-central1")
endpoint_us = model_eu.deploy(
    machine_type="n1-standard-4",
    # Model weights replicated, no EU data stored in US
)
```

---

## 3. Model Approval Workflows

### Deployment Gates and Approval Processes

**Why approval workflows:**
- Prevent unauthorized model deployments
- Ensure model quality before production
- Meet regulatory requirements (e.g., FDA for medical devices)

**Approval workflow architecture:**
```yaml
# Multi-stage approval process:
stages:
  1_development:
    owner: "Data Science Team"
    environment: "dev"
    approval: "None (automatic)"

  2_validation:
    owner: "ML Engineering Team"
    environment: "staging"
    approval: "Automated tests + peer review"
    gates:
      - Model evaluation metrics > threshold
      - Bias/fairness checks passed
      - Performance tests passed

  3_production:
    owner: "ML Operations Team"
    environment: "prod"
    approval: "Manual approval (stakeholders)"
    gates:
      - Staging validation passed
      - Security review completed
      - Business stakeholder sign-off
      - Compliance check (if regulated)
```

**Implementing approval gates with Vertex AI Pipelines:**
```python
# Example: Multi-stage deployment with approval gates
from kfp.v2 import dsl, compiler
from google.cloud import aiplatform

@dsl.component
def evaluate_model(model_uri: str, test_dataset: str) -> dict:
    """Evaluate model and return metrics."""
    # Load model and test data
    # Compute metrics
    return {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "f1": 0.925
    }

@dsl.component
def check_approval_gate(metrics: dict, threshold: float) -> str:
    """Check if metrics pass approval gate."""
    if metrics["accuracy"] < threshold:
        raise ValueError(f"Model accuracy {metrics['accuracy']} below threshold {threshold}")
    return "APPROVED"

@dsl.component
def request_stakeholder_approval(model_uri: str) -> str:
    """Request manual approval from stakeholders."""
    # Send email/Slack notification
    # Wait for approval (via Cloud Functions or manual)
    # This would typically integrate with:
    # - Jira ticket creation
    # - Email approval workflow
    # - Slack bot approval
    return "PENDING_APPROVAL"  # Human reviews and approves

@dsl.component
def deploy_to_production(model_uri: str, approval_status: str):
    """Deploy model to production if approved."""
    if approval_status != "APPROVED":
        raise ValueError("Cannot deploy without approval")

    # Deploy model
    aiplatform.init(project="prod-project", location="us-central1")
    model = aiplatform.Model.upload(model_uri=model_uri)
    endpoint = model.deploy(machine_type="n1-standard-4")
    return endpoint.resource_name

@dsl.pipeline(name="model-deployment-with-approval")
def deployment_pipeline(
    model_uri: str,
    test_dataset: str,
    accuracy_threshold: float = 0.90
):
    """Pipeline with approval gates."""

    # Stage 1: Evaluate model
    eval_task = evaluate_model(
        model_uri=model_uri,
        test_dataset=test_dataset
    )

    # Gate 1: Automated metric check
    gate_task = check_approval_gate(
        metrics=eval_task.output,
        threshold=accuracy_threshold
    )

    # Gate 2: Manual stakeholder approval
    approval_task = request_stakeholder_approval(
        model_uri=model_uri
    ).after(gate_task)

    # Stage 2: Deploy to production
    deploy_task = deploy_to_production(
        model_uri=model_uri,
        approval_status=approval_task.output
    )

# Compile and run pipeline
compiler.Compiler().compile(
    pipeline_func=deployment_pipeline,
    package_path="deployment_with_approval.json"
)

# Execute pipeline
job = aiplatform.PipelineJob(
    display_name="model-deployment-approval",
    template_path="deployment_with_approval.json",
    parameter_values={
        "model_uri": "gs://models/fraud-detector/v2",
        "test_dataset": "gs://datasets/test.csv",
        "accuracy_threshold": 0.92
    }
)

job.run()
```

**Manual approval via Cloud Functions:**
```python
# Cloud Function for approval workflow
from google.cloud import firestore
import functions_framework
import smtplib

@functions_framework.http
def request_approval(request):
    """HTTP endpoint to request deployment approval."""
    model_uri = request.args.get("model_uri")

    # Create approval request in Firestore
    db = firestore.Client()
    approval_ref = db.collection("approvals").document()
    approval_ref.set({
        "model_uri": model_uri,
        "status": "PENDING",
        "requested_at": firestore.SERVER_TIMESTAMP,
        "requested_by": request.headers.get("X-User-Email")
    })

    # Send email to approvers
    send_approval_email(
        model_uri=model_uri,
        approval_link=f"https://approval-ui.example.com/approve/{approval_ref.id}"
    )

    return {"approval_id": approval_ref.id, "status": "PENDING"}

@functions_framework.http
def approve_deployment(request):
    """HTTP endpoint to approve deployment."""
    approval_id = request.args.get("approval_id")
    approver = request.headers.get("X-User-Email")

    # Update approval status
    db = firestore.Client()
    approval_ref = db.collection("approvals").document(approval_id)
    approval_ref.update({
        "status": "APPROVED",
        "approved_by": approver,
        "approved_at": firestore.SERVER_TIMESTAMP
    })

    # Trigger deployment (via Pub/Sub)
    from google.cloud import pubsub_v1
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path("PROJECT", "model-deployment-trigger")
    publisher.publish(
        topic_path,
        data=b"deploy",
        approval_id=approval_id
    )

    return {"status": "APPROVED", "deployment": "TRIGGERED"}
```

From [MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (accessed 2025-11-16):
- Approval gates prevent unauthorized deployments
- Multi-stage pipelines enforce quality checks
- Manual approval required for production

---

### Role-Based Approval

**Approval matrix:**
```yaml
# Role-based approval requirements:
approval_roles:
  dev_deployment:
    required_approvers: 0
    allowed_roles:
      - "Data Scientist"

  staging_deployment:
    required_approvers: 1
    allowed_roles:
      - "ML Engineer"
      - "Tech Lead"

  production_deployment:
    required_approvers: 2
    allowed_roles:
      - "ML Operations Manager"
      - "Engineering Director"
      - "Compliance Officer" (for regulated industries)

# IAM configuration:
iam_roles:
  developers:
    role: "roles/aiplatform.user"
    environments: ["dev"]

  ml_engineers:
    role: "roles/aiplatform.user"
    environments: ["dev", "staging"]

  ml_ops:
    role: "roles/aiplatform.admin"
    environments: ["dev", "staging", "prod"]
```

**IAM-based deployment gates:**
```bash
# Configure environment-specific service accounts
gcloud iam service-accounts create dev-deployment-sa \
  --display-name="Dev Deployment Service Account"

gcloud iam service-accounts create prod-deployment-sa \
  --display-name="Prod Deployment Service Account"

# Dev: Allow all ML team members
gcloud projects add-iam-policy-binding PROJECT \
  --member="group:ml-team@example.com" \
  --role="roles/iam.serviceAccountUser" \
  --condition="resource.name=='projects/PROJECT/serviceAccounts/dev-deployment-sa@PROJECT.iam.gserviceaccount.com'"

# Prod: Restrict to ML Ops only
gcloud projects add-iam-policy-binding PROJECT \
  --member="group:ml-ops@example.com" \
  --role="roles/iam.serviceAccountUser" \
  --condition="resource.name=='projects/PROJECT/serviceAccounts/prod-deployment-sa@PROJECT.iam.gserviceaccount.com'"

# Production deployments require service account impersonation
gcloud ai models deploy MODEL_ID \
  --region=us-central1 \
  --endpoint=ENDPOINT_ID \
  --impersonate-service-account=prod-deployment-sa@PROJECT.iam.gserviceaccount.com
  # This will fail unless user has permission to impersonate prod-deployment-sa
```

---

## 4. Metadata Lineage Tracking

### Model Provenance and Lineage

**Why lineage matters:**
- Reproducibility (recreate exact model)
- Debugging (trace issues to source data)
- Compliance (prove model training process)
- Auditing (track who created what, when)

**Vertex AI Metadata Store:**
```yaml
# Metadata Store tracks:
artifacts:
  - Datasets (location, schema, version)
  - Models (architecture, hyperparameters, checkpoints)
  - Metrics (training metrics, evaluation results)

contexts:
  - Experiments (grouping related runs)
  - Pipelines (multi-step workflows)

executions:
  - Training runs
  - Evaluation runs
  - Deployment events

# Lineage relationships:
relationships:
  - Dataset → Model (training lineage)
  - Model → Endpoint (deployment lineage)
  - Execution → Artifact (provenance)
```

**Tracking lineage with Vertex AI Metadata:**
```python
# Example: Complete lineage tracking
from google.cloud.aiplatform import metadata

# Initialize Metadata Store
metadata.init(
    project="lineage-tracking-project",
    location="us-central1"
)

# Create dataset artifact
dataset_artifact = metadata.Artifact.create(
    uri="gs://datasets/fraud-detection/v3/",
    display_name="fraud-detection-dataset-v3",
    schema_title="system.Dataset",
    schema_version="0.0.1",
    metadata={
        "num_examples": 1000000,
        "date_range": "2024-01-01 to 2024-12-31",
        "preprocessing_version": "v2.1"
    }
)

# Create model artifact
model_artifact = metadata.Artifact.create(
    uri="gs://models/fraud-detector/v3/model.pkl",
    display_name="fraud-detector-v3",
    schema_title="system.Model",
    schema_version="0.0.1",
    metadata={
        "framework": "scikit-learn",
        "algorithm": "RandomForest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
)

# Create training execution
training_execution = metadata.Execution.create(
    schema_title="system.ContainerExecution",
    display_name="fraud-detector-training-v3",
    metadata={
        "container_image": "gcr.io/fraud-detection/trainer:v2.1",
        "training_start": "2024-11-16T10:00:00Z",
        "training_end": "2024-11-16T12:30:00Z",
        "trained_by": "ml-engineer@example.com"
    }
)

# Link lineage: Dataset → Execution → Model
training_execution.add_artifact_and_event(
    artifact=dataset_artifact,
    event_type=metadata.constants.INPUT
)

training_execution.add_artifact_and_event(
    artifact=model_artifact,
    event_type=metadata.constants.OUTPUT
)

# Create metrics artifact
metrics_artifact = metadata.Artifact.create(
    schema_title="system.Metrics",
    display_name="fraud-detector-v3-metrics",
    metadata={
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "auc_roc": 0.97
    }
)

training_execution.add_artifact_and_event(
    artifact=metrics_artifact,
    event_type=metadata.constants.OUTPUT
)

# Query lineage later
def get_model_lineage(model_artifact_id):
    """Retrieve full lineage for a model."""
    model = metadata.Artifact.get(resource_name=model_artifact_id)

    # Get training execution
    executions = model.get_executions()

    # Get input datasets
    for execution in executions:
        input_artifacts = execution.get_input_artifacts()
        print(f"Model trained from: {[a.display_name for a in input_artifacts]}")

    return executions

# Compliance audit: "Show me everything about model v3"
get_model_lineage("projects/PROJECT/locations/us-central1/metadataStores/default/artifacts/123")
```

**Automated lineage with Vertex AI Pipelines:**
```python
# Pipelines automatically track lineage
from kfp.v2 import dsl, compiler
from google.cloud import aiplatform

@dsl.component
def preprocess_data(
    input_dataset: dsl.Input[dsl.Dataset],
    output_dataset: dsl.Output[dsl.Dataset],
):
    """Preprocess dataset (lineage tracked automatically)."""
    # KFP automatically records:
    # - Input dataset artifact
    # - Output dataset artifact
    # - Execution metadata
    pass

@dsl.component
def train_model(
    dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
):
    """Train model (lineage tracked automatically)."""
    # KFP automatically records:
    # - Input dataset → model lineage
    # - Output metrics
    # - Execution metadata
    pass

@dsl.pipeline(name="lineage-tracked-pipeline")
def ml_pipeline():
    # Each step automatically tracked in Metadata Store
    preprocess_task = preprocess_data(...)
    train_task = train_model(dataset=preprocess_task.outputs["output_dataset"])

    # Lineage graph created automatically

# Run pipeline
job = aiplatform.PipelineJob(
    display_name="lineage-example",
    template_path="pipeline.json"
)
job.run()  # Lineage tracked in Vertex AI Metadata Store
```

From [Vertex AI Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction) (accessed 2025-11-16):
- Metadata Store provides automatic lineage tracking
- Pipelines integrate seamlessly with Metadata API
- Full provenance from data → model → endpoint

---

### Lineage Visualization

**Viewing lineage in Console:**
```bash
# Access lineage graph:
# 1. Navigate to Vertex AI > Metadata
# 2. Select artifact (dataset, model, etc.)
# 3. Click "View lineage" tab
# 4. See visual graph of relationships

# Query lineage via gcloud:
gcloud ai metadata-stores artifacts list \
  --metadata-store=default \
  --region=us-central1 \
  --filter="displayName:fraud-detector"

# Get artifact lineage
gcloud ai metadata-stores artifacts describe ARTIFACT_ID \
  --metadata-store=default \
  --region=us-central1
```

**Programmatic lineage queries:**
```python
# Example: Audit trail query
from google.cloud.aiplatform import metadata

def audit_model_creation(model_name: str):
    """Generate audit report for model creation."""
    # Find model artifact
    artifacts = metadata.Artifact.list(
        filter=f'display_name="{model_name}"'
    )

    for artifact in artifacts:
        # Get creation execution
        executions = artifact.get_executions()

        for execution in executions:
            # Extract audit information
            print(f"Model: {artifact.display_name}")
            print(f"Created: {execution.create_time}")
            print(f"Container: {execution.metadata.get('container_image')}")
            print(f"User: {execution.metadata.get('trained_by')}")

            # Get input datasets
            inputs = execution.get_input_artifacts()
            print(f"Input datasets: {[i.uri for i in inputs]}")

            # Get metrics
            outputs = execution.get_output_artifacts()
            metrics = [o for o in outputs if o.schema_title == "system.Metrics"]
            print(f"Metrics: {metrics[0].metadata if metrics else 'None'}")

# Run audit
audit_model_creation("fraud-detector-v3")
```

---

## 5. Organization Policy Constraints

### Resource Location Restrictions

**Organization Policies for compliance:**
```yaml
# Common constraints:
constraints:
  # Restrict resource locations (data residency)
  gcp.resourceLocations:
    allowed_values:
      - "in:eu-locations"  # EU only
      - "in:us-locations"  # US only
      # Or specific regions:
      - "in:europe-west4-locations"

  # Disable external IP addresses
  compute.vmExternalIpAccess:
    allowed: false

  # Require VPC Service Controls
  gcp.restrictVpcPeering:
    enforced: true

  # Require CMEK encryption
  gcp.requireCmekForVertexAi:
    enforced: true (custom constraint)
```

**Configuring Organization Policies:**
```bash
# Set location restriction (EU only)
gcloud resource-manager org-policies set-policy \
  --organization=ORG_ID \
  policy.yaml

# policy.yaml:
cat > policy.yaml <<EOF
constraint: constraints/gcp.resourceLocations
listPolicy:
  allowedValues:
    - in:eu-locations
EOF

# Verify policy
gcloud resource-manager org-policies describe \
  constraints/gcp.resourceLocations \
  --organization=ORG_ID

# Test compliance (this will fail if outside EU):
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=test-job \
  --worker-pool-spec=...
# Error: Location us-central1 violates organization policy
```

**Custom Organization Policy for Vertex AI:**
```yaml
# Example: Require approval tags on production models
custom_constraint:
  name: "customConstraints/requireApprovalTag"
  resource_types:
    - "aiplatform.googleapis.com/Model"
  condition:
    expression: "resource.labels.has('approval-status') && resource.labels['approval-status'] == 'approved'"
    title: "Require approval tag on models"
    description: "All deployed models must have approval-status=approved label"

  action_type: DENY  # Block deployments without approval tag

# Apply custom constraint:
# 1. Create constraint at organization level
# 2. Create policy that enforces constraint
# 3. Models without "approval-status=approved" label cannot be deployed
```

**Enforcing custom constraints:**
```bash
# Create custom constraint
gcloud org-policies set-custom-constraint policy/custom-constraint.yaml \
  --organization=ORG_ID

# custom-constraint.yaml:
cat > custom-constraint.yaml <<EOF
name: organizations/ORG_ID/customConstraints/requireApprovalTag
resourceTypes:
  - aiplatform.googleapis.com/Model
methodTypes:
  - CREATE
  - UPDATE
condition: "resource.labels.has('approval-status') && resource.labels['approval-status'] == 'approved'"
actionType: DENY
displayName: "Require approval tag on Vertex AI models"
EOF

# Create policy that enforces constraint
gcloud org-policies set-policy org-policy.yaml \
  --organization=ORG_ID

# org-policy.yaml:
cat > org-policy.yaml <<EOF
name: organizations/ORG_ID/policies/customConstraints/requireApprovalTag
spec:
  rules:
    - enforce: true
EOF

# Now deployments require approval tag:
gcloud ai models deploy MODEL_ID \
  --endpoint=ENDPOINT_ID \
  --region=us-central1
# Error: Model must have approval-status=approved label
```

From [Organization Policy Constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-11-16):
- Location restrictions enforce data residency
- Custom constraints enable approval workflows
- Policies inherited from organization to projects

---

### Vertex AI-Specific Policies

**Restricting Vertex AI features:**
```yaml
# Organization policies for Vertex AI:
vertex_ai_policies:
  # Disable AutoML (force custom training)
  aiplatform.disableAutoML:
    enforced: true

  # Require private endpoints
  aiplatform.requirePrivateEndpoints:
    enforced: true

  # Restrict model types
  aiplatform.allowedModelTypes:
    allowed_values:
      - "CUSTOM_TRAINED"
      # Deny: "PRE_TRAINED", "AUTOML"

  # Require encryption
  aiplatform.requireCmek:
    enforced: true
```

**Policy enforcement example:**
```bash
# Example: Prevent use of public Gemini endpoints
gcloud org-policies set-policy prevent-public-genai.yaml \
  --organization=ORG_ID

# prevent-public-genai.yaml:
cat > prevent-public-genai.yaml <<EOF
constraint: constraints/aiplatform.restrictPublicGenerativeAI
listPolicy:
  deniedValues:
    - "publishers/google/models/gemini-pro"
    - "publishers/google/models/palm-2"
EOF

# Attempting to use public endpoint will fail:
# from vertexai.preview.generative_models import GenerativeModel
# model = GenerativeModel("gemini-pro")
# Error: Access to gemini-pro denied by organization policy
```

---

## 6. Compliance Reporting and Audit Dashboards

### Cloud Audit Logs for Vertex AI

**Audit log types:**
```yaml
# Three types of audit logs:
log_types:
  admin_activity:
    description: "Administrative actions (create/delete resources)"
    retention: "400 days (default)"
    cost: "Free"
    examples:
      - Create model
      - Deploy endpoint
      - Delete training job

  data_access:
    description: "Data reads and writes"
    retention: "30 days (default, configurable)"
    cost: "Paid (after free tier)"
    examples:
      - Read dataset
      - Prediction requests
      - Feature Store queries
    note: "Not logged by default (must enable)"

  system_event:
    description: "GCP-initiated actions"
    retention: "400 days"
    cost: "Free"
    examples:
      - Automatic scaling events
      - System maintenance
```

**Enabling audit logging:**
```bash
# Enable Data Access logs for Vertex AI
gcloud projects get-iam-policy PROJECT > policy.yaml

# Edit policy.yaml to add auditConfigs:
cat >> policy.yaml <<EOF
auditConfigs:
  - service: aiplatform.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ
      - logType: DATA_WRITE
EOF

gcloud projects set-iam-policy PROJECT policy.yaml

# Verify logging enabled
gcloud logging read "resource.type=aiplatform.googleapis.com" \
  --limit=10 \
  --format=json
```

**Querying audit logs:**
```bash
# Query model deployments
gcloud logging read '
  resource.type="aiplatform.googleapis.com/Endpoint"
  AND protoPayload.methodName="google.cloud.aiplatform.v1.EndpointService.DeployModel"
' \
  --limit=50 \
  --format=json

# Query training job creation
gcloud logging read '
  resource.type="aiplatform.googleapis.com/CustomJob"
  AND protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob"
  AND timestamp>="2024-11-01T00:00:00Z"
' \
  --format='table(timestamp, protoPayload.authenticationInfo.principalEmail, resource.labels.job_id)'

# Query prediction requests (Data Access logs)
gcloud logging read '
  resource.type="aiplatform.googleapis.com/Endpoint"
  AND protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"
  AND timestamp>="2024-11-16T00:00:00Z"
' \
  --limit=100 \
  --format=json
```

---

### Compliance Dashboards

**Creating audit dashboard in Cloud Monitoring:**
```python
# Example: Create compliance dashboard
from google.cloud import monitoring_v3
import json

client = monitoring_v3.DashboardsServiceClient()

dashboard = monitoring_v3.Dashboard(
    display_name="Vertex AI Compliance Dashboard",
    dashboard_filters=[
        monitoring_v3.DashboardFilter(
            label_key="resource.type",
            string_value="aiplatform.googleapis.com"
        )
    ],
    grid_layout=monitoring_v3.GridLayout(
        widgets=[
            # Widget 1: Model deployments over time
            monitoring_v3.GridLayout.Widget(
                title="Model Deployments (Last 30 Days)",
                xy_chart=monitoring_v3.XyChart(
                    data_sets=[
                        monitoring_v3.XyChart.DataSet(
                            time_series_query=monitoring_v3.TimeSeriesQuery(
                                time_series_filter=monitoring_v3.TimeSeriesFilter(
                                    filter='resource.type="aiplatform.googleapis.com/Endpoint" AND metric.type="logging.googleapis.com/user/deployments"'
                                )
                            )
                        )
                    ]
                )
            ),

            # Widget 2: Compliance violations
            monitoring_v3.GridLayout.Widget(
                title="Organization Policy Violations",
                xy_chart=monitoring_v3.XyChart(
                    data_sets=[
                        monitoring_v3.XyChart.DataSet(
                            time_series_query=monitoring_v3.TimeSeriesQuery(
                                time_series_filter=monitoring_v3.TimeSeriesFilter(
                                    filter='resource.type="audited_resource" AND log_name="cloudaudit.googleapis.com%2Fpolicy" AND severity="ERROR"'
                                )
                            )
                        )
                    ]
                )
            ),

            # Widget 3: Audit log volume
            monitoring_v3.GridLayout.Widget(
                title="Audit Log Events",
                scorecard=monitoring_v3.Scorecard(
                    time_series_query=monitoring_v3.TimeSeriesQuery(
                        time_series_filter=monitoring_v3.TimeSeriesFilter(
                            filter='resource.type="aiplatform.googleapis.com"'
                        )
                    )
                )
            )
        ]
    )
)

# Create dashboard
project_name = f"projects/PROJECT_ID"
response = client.create_dashboard(name=project_name, dashboard=dashboard)
print(f"Dashboard created: {response.name}")
```

**Compliance report generation:**
```python
# Example: Generate monthly compliance report
from google.cloud import logging_v2
from datetime import datetime, timedelta
import csv

def generate_compliance_report(start_date, end_date, output_file):
    """Generate CSV report of all Vertex AI activities."""
    client = logging_v2.Client()

    # Query audit logs
    filter_str = f'''
        resource.type="aiplatform.googleapis.com"
        AND timestamp >= "{start_date.isoformat()}Z"
        AND timestamp < "{end_date.isoformat()}Z"
    '''

    entries = client.list_entries(filter_=filter_str, page_size=1000)

    # Write CSV report
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Timestamp", "User", "Action", "Resource", "Status", "Region"
        ])

        for entry in entries:
            writer.writerow([
                entry.timestamp,
                entry.payload.get("authenticationInfo", {}).get("principalEmail"),
                entry.payload.get("methodName"),
                entry.resource.labels.get("resource_id"),
                entry.payload.get("status", {}).get("code"),
                entry.resource.labels.get("location")
            ])

    print(f"Compliance report saved to {output_file}")

# Generate report for last month
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
generate_compliance_report(start_date, end_date, "vertex_ai_compliance_report.csv")
```

---

### Alerting on Compliance Violations

**Setting up compliance alerts:**
```bash
# Create alert for unauthorized deployments
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Unauthorized Model Deployment Alert" \
  --condition-display-name="Deployment without approval tag" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-filter='
    resource.type="aiplatform.googleapis.com/Endpoint"
    AND protoPayload.methodName="google.cloud.aiplatform.v1.EndpointService.DeployModel"
    AND NOT protoPayload.request.deployedModel.labels.approval-status="approved"
  '

# Alert for data residency violations
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Data Residency Violation Alert" \
  --condition-display-name="Resource created outside allowed region" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-filter='
    resource.type="aiplatform.googleapis.com"
    AND NOT (resource.labels.location:"europe-west4" OR resource.labels.location:"europe-west1")
  '

# Alert for HIPAA compliance (BAA-covered services only)
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Non-HIPAA Service Usage Alert" \
  --condition-display-name="Use of non-BAA-covered service" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-filter='
    resource.type="aiplatform.googleapis.com/AutoMlModel"
  '  # AutoML not HIPAA-eligible
```

From [Cloud Audit Logs](https://cloud.google.com/logging/docs/audit) (accessed 2025-11-16):
- Admin Activity logs enabled by default
- Data Access logs must be explicitly enabled
- Logs retained for 400 days (Admin) or 30 days (Data Access, configurable)

---

## 7. arr-coc-0-1 Compliance Configuration

### arr-coc-0-1 Compliance Setup

**Project-specific compliance requirements:**
```yaml
# arr-coc-0-1 compliance profile:
project: "arr-coc-0-1"
compliance_requirements:
  - Data residency: US only (us-west2)
  - Audit logging: Full (Admin + Data Access)
  - Model approval: Required for production
  - Lineage tracking: Full provenance

configuration:
  region: "us-west2"
  audit_logs: "enabled"
  approval_workflow: "stakeholder-sign-off"
  metadata_tracking: "automatic"
```

**Implementation:**
```python
# arr-coc-0-1 training with compliance
from google.cloud import aiplatform
from google.cloud.aiplatform import metadata

# Initialize with compliance settings
aiplatform.init(
    project="arr-coc-0-1",
    location="us-west2",  # Data residency: US only
    staging_bucket="gs://arr-coc-datasets-us-west2"
)

# Enable metadata tracking
metadata.init(project="arr-coc-0-1", location="us-west2")

# Create dataset artifact for lineage
dataset_artifact = metadata.Artifact.create(
    uri="gs://arr-coc-datasets-us-west2/texture-arrays/v1/",
    display_name="arr-coc-texture-dataset-v1",
    schema_title="system.Dataset",
    metadata={
        "num_examples": 50000,
        "channels": 13,
        "format": "texture-array"
    }
)

# Training job with audit logging
job = aiplatform.CustomTrainingJob(
    display_name="arr-coc-training-v1",
    container_uri="us-west2-docker.pkg.dev/arr-coc-0-1/training:latest",
    # Compliance:
    # - Region locked to us-west2
    # - Audit logs capture all actions
    # - Metadata tracking enabled
)

model = job.run(
    dataset="gs://arr-coc-datasets-us-west2/train/",
    service_account="arr-coc-training-sa@arr-coc-0-1.iam.gserviceaccount.com",
    # Logged in Cloud Audit Logs
)

# Create model artifact for lineage
model_artifact = metadata.Artifact.create(
    uri=model.gca_resource.artifact_uri,
    display_name="arr-coc-model-v1",
    schema_title="system.Model",
    metadata={
        "architecture": "Vervaekean-relevance-realizer",
        "channels": 13,
        "token_budget": "64-400",
        "approval_status": "pending"  # Requires approval before deployment
    }
)

# Deployment with approval gate
def deploy_with_approval(model, approval_status):
    """Deploy only if approved."""
    if approval_status != "approved":
        raise ValueError("Model requires approval before production deployment")

    endpoint = model.deploy(
        deployed_model_display_name="arr-coc-v1-prod",
        machine_type="n1-standard-4",
        service_account="arr-coc-serving-sa@arr-coc-0-1.iam.gserviceaccount.com"
    )

    # Log deployment in Metadata Store
    deployment_execution = metadata.Execution.create(
        schema_title="system.Deployment",
        display_name="arr-coc-v1-deployment",
        metadata={
            "deployed_at": "2024-11-16T15:00:00Z",
            "deployed_by": "ml-ops@arr-coc-0-1.iam.gserviceaccount.com",
            "approval_status": approval_status,
            "endpoint_id": endpoint.resource_name
        }
    )

    return endpoint

# Manual approval process
# approval_status = request_approval("arr-coc-model-v1")  # External workflow
# endpoint = deploy_with_approval(model, approval_status)
```

**arr-coc-0-1 audit configuration:**
```bash
# Enable comprehensive audit logging
gcloud projects get-iam-policy arr-coc-0-1 > arr-coc-policy.yaml

cat >> arr-coc-policy.yaml <<EOF
auditConfigs:
  - service: aiplatform.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ
      - logType: DATA_WRITE
  - service: storage.googleapis.com
    auditLogConfigs:
      - logType: ADMIN_READ
      - logType: DATA_READ  # Log dataset access
EOF

gcloud projects set-iam-policy arr-coc-0-1 arr-coc-policy.yaml

# Create compliance dashboard
gcloud monitoring dashboards create --config-from-file=arr-coc-dashboard.yaml

# Set up alerting
gcloud alpha monitoring policies create \
  --notification-channels=arr-coc-alerts \
  --display-name="arr-coc-0-1 Compliance Alert" \
  --condition-display-name="Deployment outside us-west2" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-filter='
    resource.type="aiplatform.googleapis.com"
    AND NOT resource.labels.location="us-west2"
  '
```

---

## Summary

### Compliance Checklist

**Pre-deployment compliance verification:**
```yaml
compliance_checklist:
  certifications:
    - [ ] Verify SOC 2 coverage (request report)
    - [ ] Confirm ISO 27001 certificate (download from Trust Center)
    - [ ] Sign BAA if processing PHI (HIPAA)
    - [ ] Review PCI-DSS requirements if payment data

  data_residency:
    - [ ] Configure regional endpoints (not global)
    - [ ] Set Organization Policy location restrictions
    - [ ] Verify data storage buckets in correct region
    - [ ] Test cross-region data transfer blocking

  governance:
    - [ ] Define approval workflow (dev/staging/prod)
    - [ ] Configure IAM roles for approvals
    - [ ] Implement deployment gates
    - [ ] Test approval process

  lineage:
    - [ ] Enable Metadata Store
    - [ ] Configure artifact tracking
    - [ ] Test lineage queries
    - [ ] Create lineage visualization

  audit:
    - [ ] Enable Admin Activity logs (default)
    - [ ] Enable Data Access logs (manual)
    - [ ] Set up compliance dashboard
    - [ ] Configure alerting policies
    - [ ] Test audit log queries
```

**Key commands reference:**
```bash
# Compliance certifications
# - Download from: https://cloud.google.com/security/compliance

# Data residency
gcloud config set ai/region REGION
gcloud org-policies set-policy location-policy.yaml

# Audit logging
gcloud logging read "resource.type=aiplatform.googleapis.com" --limit=50

# Metadata lineage
gcloud ai metadata-stores artifacts list --region=REGION

# Organization Policy
gcloud org-policies describe CONSTRAINT --organization=ORG_ID

# Compliance alerts
gcloud alpha monitoring policies create --config-from-file=alert.yaml
```

---

## Sources

**Web Research:**
- [Google Cloud Compliance](https://cloud.google.com/security/compliance) - Certifications (SOC 2, ISO 27001, HIPAA, PCI-DSS)
- [Vertex AI Compliance Controls](https://cloud.google.com/generative-ai-app-builder/docs/compliance-security-controls) - Service coverage
- [Data Residency](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/data-residency) - Regional endpoints
- [Organization Policy Constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) - Location restrictions
- [Cloud Audit Logs](https://cloud.google.com/logging/docs/audit) - Audit capabilities
- All sources accessed 2025-11-16

**Related Documentation:**
- Internal: gcloud-iam/00-service-accounts-ml-security.md (IAM best practices)
- Internal: gcp-vertex/15-iam-service-accounts-security.md (Vertex AI IAM)

---

*This knowledge drop provides production-ready compliance and governance for Vertex AI deployments across regulatory frameworks (HIPAA, PCI-DSS, SOC 2, ISO 27001).*
