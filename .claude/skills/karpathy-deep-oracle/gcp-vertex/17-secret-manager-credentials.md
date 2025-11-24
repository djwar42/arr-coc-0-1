# Secret Manager & Credential Management: Secure Secrets for Vertex AI

**Knowledge File: Comprehensive guide to Google Cloud Secret Manager for ML workloads**

---

## Overview

Secret Manager is Google Cloud's fully-managed service for storing, managing, and accessing sensitive information such as API keys, passwords, certificates, and connection strings. When integrated with Vertex AI, it provides enterprise-grade security for ML training jobs, enabling teams to avoid hardcoding credentials while maintaining audit trails and access controls.

**Core Value Proposition:**
- Centralized secret storage with versioning
- IAM-based granular access control
- Automatic encryption at rest (AES-256)
- Audit logging for compliance
- Seamless integration with Vertex AI Custom Jobs and GKE

**When to Use Secret Manager:**
- Storing database credentials for training data access
- Managing API keys for third-party services (W&B, HuggingFace, etc.)
- Securing certificates and OAuth tokens
- Rotating passwords automatically without downtime
- Meeting compliance requirements (HIPAA, SOC 2, PCI-DSS)

**When NOT to Use Secret Manager:**
- Public configuration values (use environment variables)
- Non-sensitive hyperparameters (use pipeline configs)
- Large binary files (use GCS instead)
- Frequently changing values (cache locally to avoid API rate limits)

From [Secret Manager Overview](https://cloud.google.com/secret-manager/docs) (accessed 2025-11-16):
> "Secret Manager provides a central place and single source of truth to manage, access, and audit secrets across Google Cloud."

From [Vertex AI Fundamentals](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md):
- Service accounts require proper IAM roles for secret access
- Vertex AI Custom Jobs run with specified service account identity
- Environment variables can be injected from Secret Manager at runtime

---

## Section 1: Secret Manager API Fundamentals

### 1.1 Core Concepts

**Secret vs Secret Version:**
- **Secret**: Named container for sensitive data (e.g., `db-password`)
- **Secret Version**: Specific value at a point in time (e.g., `v1`, `v2`, `latest`)
- Versions are immutable once created
- Can reference by version number or aliases (`latest`, `v1`)

**Resource Hierarchy:**
```
projects/PROJECT_ID/secrets/SECRET_NAME/versions/VERSION
projects/my-project/secrets/wandb-api-key/versions/3
projects/my-project/secrets/wandb-api-key/versions/latest
```

**Secret Payload Limits:**
- Maximum size: 64 KiB per version
- Recommended: Keep secrets small (API keys, passwords)
- For larger data: Store GCS path in secret, retrieve file in training

### 1.2 Creating Secrets (API)

**Via gcloud CLI:**
```bash
# Create secret from literal value
echo -n "sk_wandb_abc123def456" | gcloud secrets create wandb-api-key \
    --data-file=- \
    --replication-policy="automatic"

# Create secret from file
gcloud secrets create db-credentials \
    --data-file=/path/to/creds.json \
    --replication-policy="automatic"

# Create secret with manual replication (specific regions)
gcloud secrets create gcp-service-account-key \
    --data-file=sa-key.json \
    --replication-policy="user-managed" \
    --locations="us-central1,us-east1"
```

**Via Python SDK:**
```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
project_id = "my-ml-project"

# Create secret
parent = f"projects/{project_id}"
secret = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "wandb-api-key",
        "secret": {
            "replication": {"automatic": {}},
        },
    }
)

# Add version with payload
payload = b"sk_wandb_abc123def456"
version = client.add_secret_version(
    request={
        "parent": secret.name,
        "payload": {"data": payload},
    }
)

print(f"Secret version created: {version.name}")
```

**Replication Policies:**
- **Automatic**: Replicated across all GCP regions (default)
- **User-managed**: Specify exact regions (better latency control)
- Consider: Co-locate with training region for low latency

### 1.3 Accessing Secrets (API)

**Via gcloud CLI:**
```bash
# Access latest version
gcloud secrets versions access latest --secret="wandb-api-key"

# Access specific version
gcloud secrets versions access 3 --secret="wandb-api-key"

# List all versions
gcloud secrets versions list wandb-api-key
```

**Via Python SDK (in training script):**
```python
from google.cloud import secretmanager

def access_secret(project_id, secret_id, version_id="latest"):
    """
    Access secret value from Secret Manager.

    Args:
        project_id: GCP project ID
        secret_id: Secret name
        version_id: Version number or "latest"

    Returns:
        Secret payload as string
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    return payload

# In training code (train.py)
import os
import wandb

project_id = os.environ.get("GCP_PROJECT_ID", "my-ml-project")
wandb_key = access_secret(project_id, "wandb-api-key")

wandb.login(key=wandb_key)
wandb.init(project="vit-training", name="experiment-001")
```

**Best Practice: Cache Secrets Locally**
```python
import functools

@functools.lru_cache(maxsize=32)
def access_secret_cached(project_id, secret_id, version_id="latest"):
    """Cached version to avoid repeated API calls."""
    return access_secret(project_id, secret_id, version_id)
```

### 1.4 Managing Secret Versions

**Add New Version:**
```bash
# Add new version (rotated password)
echo -n "new_password_xyz789" | gcloud secrets versions add db-password \
    --data-file=-
```

**Disable Old Version:**
```bash
# Disable compromised version
gcloud secrets versions disable 2 --secret="api-key"
```

**Destroy Version (Irreversible):**
```bash
# Schedule destruction (cannot be undone after 24 hours)
gcloud secrets versions destroy 1 --secret="old-api-key"
```

**List Versions with States:**
```bash
gcloud secrets versions list wandb-api-key --format="table(name,state,createTime)"
```

From [Automatic Password Rotation on Google Cloud](https://grigorkh.medium.com/get-started-with-automatic-password-rotation-on-google-cloud-fdf9243bfb44) (accessed 2025-11-16):
> "Rotating a password requires three steps: (1) Change the password in the underlying system, (2) Update Secret Manager with the new password, (3) Restart applications to load the latest password."

---

## Section 2: Environment Variable Injection for Vertex AI Custom Jobs

### 2.1 Injecting Secrets as Environment Variables

**Basic Pattern (NOT Recommended - Exposes Secrets in Logs):**
```python
# ❌ BAD: Hardcoded secret
custom_job = aiplatform.CustomJob(
    display_name="vit-training",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "a2-highgpu-1g"},
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/my-project/trainer:v1",
            "env": [
                {"name": "WANDB_API_KEY", "value": "sk_wandb_EXPOSED"}  # ❌ Visible in logs
            ]
        }
    }]
)
```

**Recommended Pattern: Access in Training Script**
```python
# ✅ GOOD: Access secret at runtime inside container
custom_job = aiplatform.CustomJob(
    display_name="vit-training",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "a2-highgpu-1g"},
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/my-project/trainer:v1",
            "env": [
                {"name": "GCP_PROJECT_ID", "value": "my-ml-project"},  # ✅ Non-sensitive
                {"name": "SECRET_IDS", "value": "wandb-api-key,hf-token"}  # ✅ Names only
            ]
        }
    }],
    # Service account with Secret Manager access
    service_account="vertex-training-sa@my-ml-project.iam.gserviceaccount.com"
)
```

**Training Script (train.py):**
```python
import os
from google.cloud import secretmanager

def load_secrets():
    """Load secrets from Secret Manager based on SECRET_IDS env var."""
    project_id = os.environ["GCP_PROJECT_ID"]
    secret_ids = os.environ.get("SECRET_IDS", "").split(",")

    secrets = {}
    client = secretmanager.SecretManagerServiceClient()

    for secret_id in secret_ids:
        if not secret_id:
            continue
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secrets[secret_id] = response.payload.data.decode("UTF-8")

    return secrets

# Load secrets at startup
secrets = load_secrets()
wandb_key = secrets.get("wandb-api-key")
hf_token = secrets.get("hf-token")

# Use secrets
import wandb
wandb.login(key=wandb_key)

from huggingface_hub import login
login(token=hf_token)
```

### 2.2 Service Account Permissions

**Required IAM Roles:**
```bash
PROJECT_ID="my-ml-project"
SA_EMAIL="vertex-training-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant Secret Manager Secret Accessor role
gcloud secrets add-iam-policy-binding wandb-api-key \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# Or grant project-wide access (less secure)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"
```

**Principle of Least Privilege:**
```bash
# ✅ BEST: Per-secret access
gcloud secrets add-iam-policy-binding wandb-api-key \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# ❌ AVOID: Project-wide access
# Only use if service account needs ALL secrets
```

**Custom Role for Fine-Grained Control:**
```bash
# Create custom role with only "access" permission (no list/create)
gcloud iam roles create customSecretAccessor \
    --project=${PROJECT_ID} \
    --title="Custom Secret Accessor" \
    --description="Access secrets but cannot list or modify" \
    --permissions=secretmanager.versions.access
```

### 2.3 Multi-Secret Management Pattern

**Organize Secrets by Environment:**
```bash
# Production secrets
gcloud secrets create prod-db-password --data-file=prod-creds.txt
gcloud secrets create prod-wandb-api-key --data-file=prod-wandb-key.txt

# Staging secrets
gcloud secrets create staging-db-password --data-file=staging-creds.txt
gcloud secrets create staging-wandb-api-key --data-file=staging-wandb-key.txt
```

**Training Script with Environment Selection:**
```python
import os
from google.cloud import secretmanager

def get_secret(secret_id, project_id=None):
    """Get secret value from Secret Manager."""
    if project_id is None:
        project_id = os.environ["GCP_PROJECT_ID"]

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Environment-aware secret loading
env = os.environ.get("TRAINING_ENV", "staging")  # prod, staging, dev
db_password = get_secret(f"{env}-db-password")
wandb_key = get_secret(f"{env}-wandb-api-key")

print(f"Running in {env} environment")
```

---

## Section 3: Kubernetes Secret Mounting (GKE Integration)

### 3.1 Secret Manager Add-On for GKE

**Enable Secret Manager Add-On on GKE Cluster:**
```bash
# Create GKE cluster with Secret Manager add-on
gcloud container clusters create ml-training-cluster \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --num-nodes=3 \
    --enable-ip-alias \
    --workload-pool=my-ml-project.svc.id.goog \
    --addons=GcpSecretManager

# Enable on existing cluster
gcloud container clusters update ml-training-cluster \
    --zone=us-central1-a \
    --update-addons=GcpSecretManager=ENABLED
```

**Verify Secret Manager Add-On:**
```bash
gcloud container clusters describe ml-training-cluster \
    --zone=us-central1-a \
    | grep secretManagerConfig -A 1
```

From [Manage Secrets in GKE Part 1](https://medium.com/ankercloud-engineering/manage-secrets-in-gke-part-1-using-secret-manager-add-on-6a8f0e5f5b2d) (accessed 2025-11-16):
> "The Secret Manager add-on uses the identity of the Pod to authenticate with the Secret Manager API through Workload Identity Federation for GKE."

### 3.2 Workload Identity Setup

**Create Kubernetes ServiceAccount:**
```yaml
# k8s-service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-training-sa
  namespace: default
```

```bash
kubectl apply -f k8s-service-account.yaml
```

**Bind Kubernetes SA to GCP Secret Manager:**
```bash
PROJECT_NUM="907234637394"  # Numerical project number
PROJECT_ID="my-ml-project"
K8S_NAMESPACE="default"
K8S_SA="ml-training-sa"
SECRET_NAME="wandb-api-key"

# Grant access to specific secret
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --role=roles/secretmanager.secretAccessor \
    --member="principal://iam.googleapis.com/projects/${PROJECT_NUM}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${K8S_NAMESPACE}/sa/${K8S_SA}"
```

**How Workload Identity Works:**
1. Pod runs with Kubernetes ServiceAccount `ml-training-sa`
2. GKE exchanges K8s token for GCP token
3. Pod authenticates to Secret Manager as the bound IAM principal
4. Secret Manager checks IAM policy on secret
5. If authorized, returns secret value

### 3.3 Mounting Secrets as Files in Pods

**Define SecretProviderClass:**
```yaml
# secret-provider-class.yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: ml-secrets
  namespace: default
spec:
  provider: gke
  parameters:
    secrets: |
      - resourceName: "projects/my-ml-project/secrets/wandb-api-key/versions/latest"
        path: "wandb-api-key.txt"
      - resourceName: "projects/my-ml-project/secrets/hf-token/versions/1"
        path: "hf-token.txt"
      - resourceName: "projects/my-ml-project/secrets/db-credentials/versions/latest"
        path: "db-creds.json"
```

```bash
kubectl apply -f secret-provider-class.yaml
```

**Mount Secrets in Pod:**
```yaml
# training-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: vit-training-pod
  namespace: default
spec:
  serviceAccountName: ml-training-sa
  containers:
  - name: trainer
    image: gcr.io/my-ml-project/vit-trainer:v1
    volumeMounts:
    - name: secrets-volume
      mountPath: "/var/secrets"
      readOnly: true
    env:
    - name: WANDB_API_KEY_FILE
      value: "/var/secrets/wandb-api-key.txt"
    - name: HF_TOKEN_FILE
      value: "/var/secrets/hf-token.txt"
  volumes:
  - name: secrets-volume
    csi:
      driver: secrets-store-gke.csi.k8s.io
      readOnly: true
      volumeAttributes:
        secretProviderClass: "ml-secrets"
```

```bash
kubectl apply -f training-pod.yaml
```

**Access Mounted Secrets in Training Script:**
```python
# train.py
import os

def load_secret_from_file(file_path):
    """Read secret from mounted file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

# Load secrets from mounted files
wandb_key = load_secret_from_file(os.environ["WANDB_API_KEY_FILE"])
hf_token = load_secret_from_file(os.environ["HF_TOKEN_FILE"])

import wandb
wandb.login(key=wandb_key)

from huggingface_hub import login
login(token=hf_token)
```

### 3.4 Automatic Secret Rotation in GKE

**Enable Rotation Polling:**
```yaml
# secret-provider-class-with-rotation.yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: ml-secrets-rotated
  namespace: default
spec:
  provider: gke
  parameters:
    secrets: |
      - resourceName: "projects/my-ml-project/secrets/db-password/versions/latest"
        path: "db-password.txt"
    rotation: |
      rotationPollInterval: "120s"  # Check every 2 minutes
```

**How Rotation Works:**
1. Create new secret version in Secret Manager
2. Update secret reference to `versions/latest`
3. CSI driver polls Secret Manager every `rotationPollInterval`
4. Detects new version and updates mounted file
5. Application reads updated file on next access

**Handling Rotation in Application:**
```python
import os
import time
from pathlib import Path

class SecretWatcher:
    """Watch secret file for changes and reload."""

    def __init__(self, secret_file):
        self.secret_file = Path(secret_file)
        self.last_mtime = self.secret_file.stat().st_mtime
        self.value = self._load()

    def _load(self):
        with open(self.secret_file, 'r') as f:
            return f.read().strip()

    def get(self):
        """Get secret value, reload if file changed."""
        current_mtime = self.secret_file.stat().st_mtime
        if current_mtime != self.last_mtime:
            print(f"Secret rotated, reloading from {self.secret_file}")
            self.value = self._load()
            self.last_mtime = current_mtime
        return self.value

# Use in training script
db_password_watcher = SecretWatcher("/var/secrets/db-password.txt")

# In training loop
for epoch in range(num_epochs):
    db_conn = connect_to_db(password=db_password_watcher.get())
    # Training code...
```

---

## Section 4: Automatic Rotation Policies

### 4.1 Rotation Schedule Configuration

**Create Rotation Schedule:**
```bash
# Rotate secret every 30 days
gcloud secrets create db-password \
    --replication-policy="automatic" \
    --rotation-period="2592000s" \  # 30 days in seconds
    --rotation-topic="projects/my-ml-project/topics/secret-rotation"
```

**Rotation Topic (Pub/Sub):**
```bash
# Create Pub/Sub topic for rotation events
gcloud pubsub topics create secret-rotation

# Subscribe to rotation events
gcloud pubsub subscriptions create secret-rotation-sub \
    --topic=secret-rotation
```

**Rotation Notification Message:**
```json
{
  "name": "projects/my-ml-project/secrets/db-password",
  "rotationTime": "2025-02-15T00:00:00Z"
}
```

### 4.2 Automated Rotation Workflow with Cloud Functions

**Architecture:**
```
Cloud Scheduler (monthly cron)
    ↓
Pub/Sub Topic (rotation-trigger)
    ↓
Cloud Function (rotate-password)
    ↓
1. Generate new password
2. Update database
3. Create new Secret Manager version
4. Notify applications (Pub/Sub)
```

**Cloud Function (Python):**
```python
# main.py
import base64
import json
import os
import secrets
from google.cloud import secretmanager
import pg8000  # PostgreSQL connector

def rotate_db_password(event, context):
    """
    Rotate Cloud SQL password and update Secret Manager.

    Triggered by Pub/Sub message from Cloud Scheduler.
    """
    # Parse Pub/Sub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    config = json.loads(pubsub_message)

    project_id = config['project_id']
    secret_id = config['secret_id']
    db_instance = config['db_instance']
    db_user = config['db_user']
    db_name = config['db_name']

    # Generate new password
    new_password = generate_secure_password()

    # Update database password
    update_database_password(db_instance, db_user, db_name, new_password)

    # Create new Secret Manager version
    sm_client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}/secrets/{secret_id}"
    payload = new_password.encode("UTF-8")

    version = sm_client.add_secret_version(
        request={
            "parent": parent,
            "payload": {"data": payload},
        }
    )

    print(f"Password rotated: {version.name}")

    # Publish notification (optional)
    from google.cloud import pubsub_v1
    publisher = pubsub_v1.PublisherClient()
    topic_path = f"projects/{project_id}/topics/password-rotated"

    message = json.dumps({
        "secret_id": secret_id,
        "version": version.name,
        "timestamp": version.create_time.isoformat()
    }).encode("utf-8")

    publisher.publish(topic_path, message)
    return "Password rotated successfully"

def generate_secure_password(length=32):
    """Generate cryptographically secure password."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def update_database_password(instance, user, db_name, new_password):
    """Update Cloud SQL user password."""
    import subprocess

    # Use gcloud to update password
    cmd = [
        "gcloud", "sql", "users", "set-password", user,
        "--instance", instance,
        "--password", new_password
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to update password: {result.stderr}")
```

**Deploy Cloud Function:**
```bash
gcloud functions deploy rotate-db-password \
    --runtime=python311 \
    --trigger-topic=rotation-trigger \
    --entry-point=rotate_db_password \
    --region=us-central1 \
    --service-account=rotation-function-sa@my-ml-project.iam.gserviceaccount.com
```

**Cloud Scheduler Job:**
```bash
# Rotate on 1st of every month
gcloud scheduler jobs create pubsub rotate-db-monthly \
    --schedule="0 0 1 * *" \
    --topic=rotation-trigger \
    --message-body='{
      "project_id": "my-ml-project",
      "secret_id": "db-password",
      "db_instance": "ml-training-db",
      "db_user": "training_user",
      "db_name": "experiments"
    }' \
    --location=us-central1
```

### 4.3 Rotation Best Practices

**Rotation Schedules:**
- **Critical secrets** (production DB): 30 days
- **API keys** (third-party): 90 days
- **Service account keys**: 90 days
- **Development secrets**: 180 days

**Grace Period Pattern:**
```python
# Keep previous version active during transition
def rotate_with_grace_period(secret_id, new_value, grace_hours=24):
    """
    Rotate secret while keeping old version enabled.

    Args:
        secret_id: Secret name
        new_value: New secret value
        grace_hours: Hours to keep old version active
    """
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}/secrets/{secret_id}"

    # Add new version
    new_version = client.add_secret_version(
        request={
            "parent": parent,
            "payload": {"data": new_value.encode("UTF-8")},
        }
    )

    # Schedule old version disablement
    import time
    from datetime import datetime, timedelta

    disable_time = datetime.now() + timedelta(hours=grace_hours)
    print(f"Old version will be disabled at {disable_time}")

    # In production: Use Cloud Scheduler to disable old version
    # after grace period
```

**Zero-Downtime Rotation:**
1. Create new secret version
2. Gradually roll out to pods (rolling update)
3. Monitor for errors
4. After 100% rollout, disable old version
5. After grace period, destroy old version

---

## Section 5: Customer-Managed Encryption Keys (CMEK)

### 5.1 CMEK Overview for Secret Manager

**Default Encryption:**
- Secrets encrypted at rest with Google-managed keys (AES-256)
- Automatic key rotation by Google
- No configuration required

**Customer-Managed Encryption Keys (CMEK):**
- You control encryption keys via Cloud KMS
- Additional layer of security and compliance
- Required for: HIPAA, PCI-DSS, SOC 2 Type II

**When to Use CMEK:**
- Regulatory compliance requirements
- Need to prove data deletion (destroy key)
- Requirement for key rotation control
- Multi-region data residency

From [Customer-Managed Encryption Keys (CMEK)](https://cloud.google.com/kms/docs/cmek) (accessed 2025-11-16):
> "Customer-managed encryption keys are encryption keys that you own. This capability lets you have greater control over the keys used to encrypt data at rest."

### 5.2 Setting Up CMEK for Secret Manager

**Create Cloud KMS Key Ring and Key:**
```bash
PROJECT_ID="my-ml-project"
REGION="us-central1"
KEYRING_NAME="secret-manager-keyring"
KEY_NAME="secret-encryption-key"

# Create key ring
gcloud kms keyrings create ${KEYRING_NAME} \
    --location=${REGION}

# Create encryption key
gcloud kms keys create ${KEY_NAME} \
    --location=${REGION} \
    --keyring=${KEYRING_NAME} \
    --purpose=encryption \
    --rotation-period=90d \
    --next-rotation-time=$(date -u --date="+90 days" +%Y-%m-%dT%H:%M:%SZ)
```

**Grant Secret Manager Access to KMS Key:**
```bash
PROJECT_NUM="907234637394"  # Numerical project number

# Get Secret Manager service agent email
SM_SA="service-${PROJECT_NUM}@gcp-sa-secretmanager.iam.gserviceaccount.com"

# Grant cloudkms.cryptoKeyEncrypterDecrypter role
gcloud kms keys add-iam-policy-binding ${KEY_NAME} \
    --location=${REGION} \
    --keyring=${KEYRING_NAME} \
    --member="serviceAccount:${SM_SA}" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**Create Secret with CMEK:**
```bash
KMS_KEY="projects/${PROJECT_ID}/locations/${REGION}/keyRings/${KEYRING_NAME}/cryptoKeys/${KEY_NAME}"

# Create CMEK-encrypted secret
echo -n "my_secure_api_key" | gcloud secrets create cmek-api-key \
    --data-file=- \
    --replication-policy="user-managed" \
    --locations=${REGION} \
    --kms-key-name=${KMS_KEY}
```

**Via Python SDK:**
```python
from google.cloud import secretmanager

def create_cmek_secret(project_id, secret_id, kms_key_name):
    """
    Create secret encrypted with customer-managed key.

    Args:
        project_id: GCP project ID
        secret_id: Secret name
        kms_key_name: Full KMS key resource name
    """
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project_id}"

    secret = client.create_secret(
        request={
            "parent": parent,
            "secret_id": secret_id,
            "secret": {
                "replication": {
                    "user_managed": {
                        "replicas": [
                            {
                                "location": "us-central1",
                                "customer_managed_encryption": {
                                    "kms_key_name": kms_key_name
                                }
                            }
                        ]
                    }
                },
            },
        }
    )

    print(f"Created CMEK secret: {secret.name}")
    return secret

# Usage
kms_key = "projects/my-ml-project/locations/us-central1/keyRings/secret-manager-keyring/cryptoKeys/secret-encryption-key"
create_cmek_secret("my-ml-project", "prod-db-password", kms_key)
```

### 5.3 CMEK Key Rotation

**Automatic Rotation:**
```bash
# Set rotation period when creating key
gcloud kms keys create secret-encryption-key \
    --location=us-central1 \
    --keyring=secret-manager-keyring \
    --purpose=encryption \
    --rotation-period=90d \
    --next-rotation-time=$(date -u --date="+90 days" +%Y-%m-%dT%H:%M:%SZ)
```

**Manual Rotation:**
```bash
# Create new key version
gcloud kms keys versions create \
    --location=us-central1 \
    --keyring=secret-manager-keyring \
    --key=secret-encryption-key \
    --primary

# Set new version as primary
gcloud kms keys set-primary-version secret-encryption-key \
    --location=us-central1 \
    --keyring=secret-manager-keyring \
    --version=2
```

**Impact on Secrets:**
- Existing secret versions remain encrypted with old key version
- New secret versions use new key version
- Re-encryption NOT required (Cloud KMS handles it transparently)

### 5.4 Destroying CMEK Keys (Data Deletion Proof)

**Schedule Key Destruction:**
```bash
# Schedule destruction (24-hour grace period)
gcloud kms keys versions destroy 1 \
    --location=us-central1 \
    --keyring=secret-manager-keyring \
    --key=secret-encryption-key
```

**Impact:**
- Secrets encrypted with destroyed key become **permanently inaccessible**
- Provides cryptographic proof of data deletion
- Required for GDPR "right to be forgotten" compliance

**Best Practice: Test Destruction in Staging First**
```bash
# Staging environment key destruction test
gcloud kms keys versions destroy 1 \
    --location=us-central1 \
    --keyring=staging-secret-keyring \
    --key=staging-encryption-key

# Verify secrets are inaccessible
gcloud secrets versions access latest --secret="staging-api-key"
# Should fail with "FAILED_PRECONDITION: The key version is not available"
```

---

## Section 6: Secret Access Audit Logs

### 6.1 Enabling Audit Logs

**Enable Data Access Audit Logs:**
```bash
# Via gcloud (enable for Secret Manager)
gcloud projects get-iam-policy ${PROJECT_ID} > policy.yaml

# Edit policy.yaml:
auditConfigs:
- service: secretmanager.googleapis.com
  auditLogConfigs:
  - logType: ADMIN_READ
  - logType: DATA_READ
  - logType: DATA_WRITE

# Update policy
gcloud projects set-iam-policy ${PROJECT_ID} policy.yaml
```

**What Gets Logged:**
- **ADMIN_READ**: List secrets, get secret metadata
- **DATA_READ**: Access secret versions (most important)
- **DATA_WRITE**: Create/update/delete secrets

### 6.2 Querying Access Logs

**View Secret Access Logs:**
```bash
# Last 7 days of secret access
gcloud logging read 'resource.type="secretmanager.googleapis.com/Secret"
    AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"
    AND timestamp >= "2025-11-09T00:00:00Z"' \
    --limit=50 \
    --format=json
```

**Filter by Secret Name:**
```bash
gcloud logging read 'resource.type="secretmanager.googleapis.com/Secret"
    AND resource.labels.secret_id="wandb-api-key"
    AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"' \
    --limit=20 \
    --format=json
```

**Filter by Service Account:**
```bash
gcloud logging read 'resource.type="secretmanager.googleapis.com/Secret"
    AND protoPayload.authenticationInfo.principalEmail="vertex-training-sa@my-ml-project.iam.gserviceaccount.com"
    AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"' \
    --limit=20
```

### 6.3 Audit Log Schema

**Sample Audit Log Entry:**
```json
{
  "protoPayload": {
    "methodName": "google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion",
    "resourceName": "projects/907234637394/secrets/wandb-api-key/versions/3",
    "authenticationInfo": {
      "principalEmail": "vertex-training-sa@my-ml-project.iam.gserviceaccount.com"
    },
    "requestMetadata": {
      "callerIp": "35.192.45.123",
      "callerSuppliedUserAgent": "google-cloud-sdk gcloud/450.0.0"
    },
    "status": {
      "code": 0,
      "message": "OK"
    }
  },
  "insertId": "abc123xyz",
  "resource": {
    "type": "secretmanager.googleapis.com/Secret",
    "labels": {
      "project_id": "my-ml-project",
      "secret_id": "wandb-api-key"
    }
  },
  "timestamp": "2025-11-16T10:23:45.678Z",
  "severity": "INFO"
}
```

**Key Fields:**
- `principalEmail`: Who accessed the secret
- `resourceName`: Which secret version
- `timestamp`: When
- `callerIp`: Source IP address
- `status.code`: Success (0) or error

### 6.4 Alerting on Suspicious Access

**Create Log Metric for Secret Access:**
```bash
gcloud logging metrics create secret-access-count \
    --description="Count of secret accesses" \
    --log-filter='resource.type="secretmanager.googleapis.com/Secret"
        AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"'
```

**Create Alerting Policy:**
```bash
# Alert if >100 accesses in 5 minutes (potential leak/attack)
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High Secret Access Rate" \
    --condition-display-name="Secret access rate > 100/5min" \
    --condition-threshold-value=100 \
    --condition-threshold-duration=300s \
    --condition-filter='metric.type="logging.googleapis.com/user/secret-access-count"'
```

**Alert on Unauthorized Access Attempts:**
```bash
gcloud logging metrics create secret-access-denied \
    --description="Failed secret access attempts" \
    --log-filter='resource.type="secretmanager.googleapis.com/Secret"
        AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"
        AND protoPayload.status.code!=0'
```

**Send Alerts to Pub/Sub for Custom Processing:**
```bash
# Create Pub/Sub topic
gcloud pubsub topics create security-alerts

# Create log sink
gcloud logging sinks create secret-access-sink \
    pubsub.googleapis.com/projects/my-ml-project/topics/security-alerts \
    --log-filter='resource.type="secretmanager.googleapis.com/Secret"
        AND protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"
        AND protoPayload.status.code!=0'
```

---

## Section 7: arr-coc-0-1 Credential Management

### 7.1 W&B Launch + Vertex AI Secret Management

**Architecture:**
```
Secret Manager (wandb-api-key)
    ↓
Vertex AI Custom Job (env var: SECRET_IDS=wandb-api-key)
    ↓
Training Container (train.py)
    ↓
Access Secret via Python SDK
    ↓
W&B Launch Agent (authenticate with key)
    ↓
Log metrics to W&B
```

**Create W&B API Key Secret:**
```bash
# Get W&B API key from https://wandb.ai/authorize
WANDB_KEY="<your-wandb-api-key>"

echo -n "${WANDB_KEY}" | gcloud secrets create wandb-api-key \
    --data-file=- \
    --replication-policy="automatic"

# Grant Vertex AI service account access
gcloud secrets add-iam-policy-binding wandb-api-key \
    --member="serviceAccount:vertex-training-sa@arr-coc-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**Training Script (arr-coc-0-1/training/train.py):**
```python
"""
arr-coc-0-1 Training Script with Secret Manager Integration
"""
import os
from google.cloud import secretmanager
import wandb

def get_secret(secret_id, project_id=None):
    """Get secret from Secret Manager."""
    if project_id is None:
        project_id = os.environ.get("GCP_PROJECT_ID", "arr-coc-project")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def init_wandb(config):
    """Initialize W&B with secret API key."""
    wandb_key = get_secret("wandb-api-key")
    wandb.login(key=wandb_key)

    run = wandb.init(
        project="arr-coc-training",
        name=config.get("run_name", "experiment-001"),
        config=config
    )
    return run

# Main training loop
def train():
    config = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_patches": 200,
        "lod_range": (64, 400)
    }

    run = init_wandb(config)

    # Training code...
    for epoch in range(num_epochs):
        loss = train_epoch()
        wandb.log({"epoch": epoch, "loss": loss})

    run.finish()

if __name__ == "__main__":
    train()
```

**Vertex AI Custom Job with Secret Access:**
```python
from google.cloud import aiplatform

aiplatform.init(project="arr-coc-project", location="us-west2")

custom_job = aiplatform.CustomJob(
    display_name="arr-coc-training",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-west2-docker.pkg.dev/arr-coc-project/ml-containers/arr-coc-trainer:v1",
            "command": ["python", "training/train.py"],
            "env": [
                {"name": "GCP_PROJECT_ID", "value": "arr-coc-project"},
                {"name": "SECRET_IDS", "value": "wandb-api-key"}
            ]
        }
    }],
    service_account="vertex-training-sa@arr-coc-project.iam.gserviceaccount.com"
)

custom_job.run(sync=True)
```

### 7.2 HuggingFace Token Management

**Store HuggingFace Token:**
```bash
# Get token from https://huggingface.co/settings/tokens
HF_TOKEN="hf_abc123def456..."

echo -n "${HF_TOKEN}" | gcloud secrets create hf-token \
    --data-file=- \
    --replication-policy="automatic"

gcloud secrets add-iam-policy-binding hf-token \
    --member="serviceAccount:vertex-training-sa@arr-coc-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**Use in Training (arr-coc-0-1/training/data_loader.py):**
```python
from google.cloud import secretmanager
from transformers import AutoTokenizer, AutoModel

def load_pretrained_model(model_name="Qwen/Qwen2-VL-7B"):
    """Load HuggingFace model with authenticated token."""
    # Get HF token
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/arr-coc-project/secrets/hf-token/versions/latest"
    response = client.access_secret_version(request={"name": name})
    hf_token = response.payload.data.decode("UTF-8")

    # Load model with auth
    from huggingface_hub import login
    login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model
```

### 7.3 Database Credentials for Training Data

**Store PostgreSQL Credentials:**
```bash
# Create JSON credentials file
cat > db-creds.json << EOF
{
  "host": "10.128.0.5",
  "port": 5432,
  "database": "training_data",
  "user": "ml_trainer",
  "password": "secure_password_xyz789"
}
EOF

# Store in Secret Manager
gcloud secrets create db-credentials \
    --data-file=db-creds.json \
    --replication-policy="automatic"

rm db-creds.json  # Remove local file

# Grant access
gcloud secrets add-iam-policy-binding db-credentials \
    --member="serviceAccount:vertex-training-sa@arr-coc-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**Use in Data Loading (arr-coc-0-1/training/dataset.py):**
```python
import json
import psycopg2
from google.cloud import secretmanager

def get_db_connection():
    """Connect to training data database."""
    # Get credentials
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/arr-coc-project/secrets/db-credentials/versions/latest"
    response = client.access_secret_version(request={"name": name})
    creds = json.loads(response.payload.data.decode("UTF-8"))

    # Connect
    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        user=creds["user"],
        password=creds["password"]
    )

    return conn

# Load training data
def load_training_samples(query):
    """Load training samples from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(query)
    samples = cursor.fetchall()

    cursor.close()
    conn.close()

    return samples
```

### 7.4 Multi-Environment Secret Management

**Organize Secrets by Environment:**
```bash
# Production secrets
echo -n "${PROD_WANDB_KEY}" | gcloud secrets create prod-wandb-api-key --data-file=-
echo -n "${PROD_DB_PASSWORD}" | gcloud secrets create prod-db-password --data-file=-

# Staging secrets
echo -n "${STAGING_WANDB_KEY}" | gcloud secrets create staging-wandb-api-key --data-file=-
echo -n "${STAGING_DB_PASSWORD}" | gcloud secrets create staging-db-password --data-file=-

# Development secrets
echo -n "${DEV_WANDB_KEY}" | gcloud secrets create dev-wandb-api-key --data-file=-
echo -n "${DEV_DB_PASSWORD}" | gcloud secrets create dev-db-password --data-file=-
```

**Environment-Aware Training Script:**
```python
import os
from google.cloud import secretmanager

class SecretManager:
    """Manage secrets for arr-coc-0-1 training."""

    def __init__(self, project_id="arr-coc-project", env="staging"):
        self.project_id = project_id
        self.env = env  # prod, staging, dev
        self.client = secretmanager.SecretManagerServiceClient()

    def get_secret(self, secret_name):
        """Get environment-specific secret."""
        secret_id = f"{self.env}-{secret_name}"
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/latest"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

    def get_all_secrets(self, secret_names):
        """Get multiple secrets at once."""
        return {name: self.get_secret(name) for name in secret_names}

# Usage
env = os.environ.get("TRAINING_ENV", "staging")
secrets = SecretManager(env=env)

wandb_key = secrets.get_secret("wandb-api-key")
db_password = secrets.get_secret("db-password")
hf_token = secrets.get_secret("hf-token")

print(f"Running in {env} environment")
```

---

## Key Takeaways

**Secret Manager Strengths:**
1. Centralized secret storage with versioning
2. IAM-based access control (per-secret granularity)
3. Automatic encryption at rest (AES-256)
4. Audit logging for compliance (who/what/when)
5. Seamless Vertex AI and GKE integration

**Secret Manager Limitations:**
1. API rate limits: 1,800 requests/minute/project
2. 64 KiB payload size limit
3. No built-in secret generation (use Cloud Functions)
4. Cost: $0.06 per 10,000 access operations

**When Secret Manager is the Right Choice:**
- Vertex AI Custom Jobs need credentials (W&B, HuggingFace, DBs)
- GKE workloads require secrets mounted as files
- Regulatory compliance (audit logs, CMEK encryption)
- Automated secret rotation workflows
- Multi-environment secret management (prod/staging/dev)

**When to Consider Alternatives:**
- Simple config values: Use environment variables
- Large files: Store in GCS, reference path in secret
- Extremely high access rate: Cache secrets locally
- Multi-cloud: Use HashiCorp Vault instead

**Next Steps:**
- Review VPC Service Controls for network isolation (file 16-vpc-service-controls-private.md)
- Understand Compliance & Governance (file 18-compliance-governance-audit.md)
- Explore Neural Architecture Search with secret hyperparameters (file 19-nas-hyperparameter-tuning.md)

---

## Sources

**Official Documentation:**
- [Secret Manager Overview](https://cloud.google.com/secret-manager/docs) (accessed 2025-11-16)
- [Secret Rotation Schedules](https://cloud.google.com/secret-manager/docs/secret-rotation) (accessed 2025-11-16)
- [Enable CMEK for Secret Manager](https://cloud.google.com/secret-manager/docs/cmek) (accessed 2025-11-16)
- [Customer-Managed Encryption Keys (CMEK)](https://cloud.google.com/kms/docs/cmek) (accessed 2025-11-16)
- [Use Secret Manager add-on with GKE](https://cloud.google.com/secret-manager/docs/secret-manager-managed-csi-component) (accessed 2025-11-16)

**Source Documents:**
- [30-vertex-ai-fundamentals.md](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md) - Vertex AI architecture, service accounts, IAM roles

**Web Research:**
- [How to use Google Cloud's automatic password rotation](https://cloud.google.com/blog/products/identity-security/how-to-use-google-clouds-automatic-password-rotation) - Google Cloud Blog (accessed 2025-11-16)
- [Get Started with Automatic Password Rotation on Google Cloud](https://grigorkh.medium.com/get-started-with-automatic-password-rotation-on-google-cloud-fdf9243bfb44) - Grigor Khachatryan, Medium (accessed 2025-11-16)
- [Manage Secrets in GKE Part 1: Using Secret Manager add-on](https://medium.com/ankercloud-engineering/manage-secrets-in-gke-part-1-using-secret-manager-add-on-6a8f0e5f5b2d) - Madhav Sake, Medium (accessed 2025-11-16)

**Additional References:**
- Search results: "Secret Manager Vertex AI Custom Jobs 2024" (accessed 2025-11-16)
- Search results: "automatic secret rotation GCP" (accessed 2025-11-16)
- Search results: "CMEK customer-managed encryption keys GCP" (accessed 2025-11-16)
- Search results: "Kubernetes secrets GKE integration" (accessed 2025-11-16)
