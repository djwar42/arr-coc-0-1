# W&B Launch Agent Setup for Vertex AI

**Complete guide to deploying and configuring W&B Launch agents for Google Cloud Vertex AI training jobs**

---

## Overview

W&B Launch enables automated ML training job execution on Vertex AI Custom Jobs. The Launch agent polls queues for job submissions and orchestrates container-based training on Google Cloud infrastructure. This provides production-grade ML training automation with full W&B experiment tracking integration.

**Key capabilities:**
- Queue-based job orchestration
- Container image management (Docker/Kaniko builders)
- GCP service account authentication
- Resource configuration via queue config
- Multi-region deployment support

From [W&B Launch Vertex AI Documentation](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
- Vertex AI integration via `google-cloud-aiplatform` SDK
- CustomJob class for training orchestration
- Queue config maps to CustomJob parameters

---

## Section 1: Launch Agent Installation

### Prerequisites

**GCP Requirements:**
1. Google Cloud project with Vertex AI API enabled
2. Google Cloud Artifact Registry repository for container images
3. GCS staging bucket (same region as Vertex AI workloads)
4. Service account with Vertex AI permissions

From [GCP IAM Documentation](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) (accessed 2025-01-31):
- Vertex AI Service Agent auto-created on API usage
- Service accounts require proper IAM role bindings

**W&B Requirements:**
- W&B account with Launch enabled
- Entity/team access for queue creation
- API key for agent authentication

### Agent Installation Options

**Option 1: Local Machine**
```bash
# Install wandb CLI with Launch support
pip install wandb[launch]

# Verify installation
wandb launch-agent --version
```

**Option 2: Google Compute Engine**
```bash
# Create Compute Engine instance
gcloud compute instances create wandb-launch-agent \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --service-account=<service-account-email> \
    --scopes=cloud-platform

# SSH and install
gcloud compute ssh wandb-launch-agent --zone=us-central1-a
pip install wandb[launch]
```

**Option 3: Cloud Run (Serverless)**
```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: wandb-launch-agent
spec:
  template:
    spec:
      serviceAccountName: <service-account>
      containers:
      - image: gcr.io/<project>/wandb-launch-agent:latest
        command: ["wandb", "launch-agent"]
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-api-key
              key: key
```

From [W&B Launch Agent Setup](https://docs.wandb.ai/platform/launch/setup-agent-advanced) (accessed 2025-01-31):
- Agent config file: `~/.config/wandb/launch-config.yaml` by default
- Can override with `--config` flag
- Supports Docker and Kaniko builders

### Basic Agent Configuration

**Minimal launch-config.yaml:**
```yaml
# ~/.config/wandb/launch-config.yaml
max_jobs: 3  # Concurrent job limit
queues:
  - vertex-ai-queue  # Queue name to poll

# No builder needed if using pre-built images
builder:
  type: noop
```

### Starting the Agent

**Basic start:**
```bash
# Use default config location
wandb launch-agent

# Specify config file
wandb launch-agent --config /path/to/launch-config.yaml

# Specify queue directly
wandb launch-agent --queue vertex-ai-queue --max-jobs 5
```

**Environment variables:**
```bash
export WANDB_API_KEY="your-api-key"
export WANDB_ENTITY="your-team"
export WANDB_BASE_URL="https://api.wandb.ai"  # Or self-hosted URL

wandb launch-agent
```

### Agent Health Monitoring

**Check agent status:**
```bash
# View agent logs
wandb launch-agent --verbose

# Monitor in W&B App
# Navigate to: https://wandb.ai/launch → Select queue → "Agents" tab
```

**Agent metrics:**
- Jobs processed count
- Success/failure rates
- Current job status
- Agent last heartbeat

From [W&B Launch Queue Monitoring](https://docs.wandb.ai/platform/launch/launch-queue-observability) (accessed 2025-01-31):
- Agent heartbeat tracking
- Job state transitions
- Error reporting

---

## Section 2: Queue Configuration

### Creating a Vertex AI Queue

**Via W&B UI:**
1. Navigate to [https://wandb.ai/launch](https://wandb.ai/launch)
2. Click "Create Queue"
3. Select entity
4. Choose "Google Cloud Vertex AI" as resource
5. Configure queue settings

### Queue Config Structure

**Config anatomy:**
```yaml
spec:
  # CustomJob constructor arguments
  worker_pool_specs:  # List of worker pool configurations
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
        accelerator_count: 0
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}  # Populated by Launch
  staging_bucket: gs://my-vertex-staging  # Required
  service_account: <optional-custom-sa>

run:
  # CustomJob.run() method arguments
  restart_job_on_worker_restart: false
```

From [W&B Vertex AI Setup](https://docs.wandb.ai/platform/launch/setup-vertex) (accessed 2025-01-31):
- `spec` maps to `CustomJob` constructor
- `run` maps to `CustomJob.run()` method
- Use snake_case (not camelCase) for all keys

**Required fields:**
- `spec.worker_pool_specs`: Non-empty list
- `spec.staging_bucket`: GCS bucket URI

### Worker Pool Specifications

**Single GPU configuration:**
```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-8
        accelerator_type: NVIDIA_TESLA_V100
        accelerator_count: 1
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: gs://my-staging-bucket
```

**Multi-GPU configuration (8x A100):**
```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: a2-highgpu-8g
        accelerator_type: NVIDIA_TESLA_A100
        accelerator_count: 8
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: gs://my-staging-bucket
```

**Distributed training (multi-node):**
```yaml
spec:
  worker_pool_specs:
    # Chief worker
    - machine_spec:
        machine_type: n1-standard-16
        accelerator_type: NVIDIA_TESLA_V100
        accelerator_count: 4
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
    # Worker pool
    - machine_spec:
        machine_type: n1-standard-16
        accelerator_type: NVIDIA_TESLA_V100
        accelerator_count: 4
      replica_count: 3
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: gs://my-staging-bucket
```

From [Vertex AI CustomJob Documentation](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) (accessed 2025-01-31):
- Worker pool spec defines compute resources
- First pool is "chief" in distributed training
- `replica_count` sets number of VMs per pool

### Environment Variables and Secrets

**Inject environment variables:**
```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
      container_spec:
        image_uri: ${image_uri}
        env:
          - name: MODEL_NAME
            value: "gpt-llm-trainer"
          - name: BATCH_SIZE
            value: "32"
          - name: WANDB_PROJECT
            value: ${project_name}
  staging_bucket: gs://my-staging-bucket
```

**Using Secret Manager:**
```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
      container_spec:
        image_uri: ${image_uri}
        env:
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: huggingface-token
                key: token
  staging_bucket: gs://my-staging-bucket
```

From [GCP Secret Manager](https://cloud.google.com/secret-manager/docs) (accessed 2025-01-31):
- Secrets encrypted at rest
- IAM-based access control
- Automatic secret rotation support

### Dynamic Macros

**Available macros:**
```yaml
spec:
  worker_pool_specs:
    - container_spec:
        image_uri: ${image_uri}  # Resolved at runtime
        env:
          - name: RUN_ID
            value: ${run_id}
          - name: RUN_NAME
            value: ${run_name}
          - name: PROJECT_NAME
            value: ${project_name}
          - name: ENTITY_NAME
            value: ${entity_name}
  staging_bucket: gs://my-staging-${entity_name}
```

From [W&B Queue Configuration](https://docs.wandb.ai/platform/launch/setup-queue-advanced) (accessed 2025-01-31):
- Macros evaluated when agent dequeues job
- Custom macros resolve from agent environment variables

### Queue Config Templates

**Define user-configurable parameters:**
```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: "{{machine_type}}"
        accelerator_type: "{{gpu_type}}"
        accelerator_count: {{gpu_count}}
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: gs://my-staging-bucket
```

**In W&B UI:**
- Parse configuration creates template tiles
- Set data type (string, integer, float)
- Define allowed values, min/max ranges
- Users can only modify within constraints

**Example template tile:**
- `machine_type` (string): ["n1-standard-4", "n1-standard-8", "a2-highgpu-8g"]
- `gpu_type` (string): ["NVIDIA_TESLA_V100", "NVIDIA_TESLA_A100"]
- `gpu_count` (integer): min=0, max=8, default=1

From [W&B Queue Templates](https://docs.wandb.ai/platform/launch/setup-queue-advanced#configure-queue-template) (accessed 2025-01-31):
- Templates enforce compute guardrails
- Admins control resource limits
- Users customize within boundaries

### Complete Queue Configuration Example

**Production-ready config:**
```yaml
spec:
  # Worker pool configuration
  worker_pool_specs:
    - machine_spec:
        machine_type: a2-highgpu-8g
        accelerator_type: NVIDIA_TESLA_A100
        accelerator_count: 8
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
        command: ["python", "train.py"]
        args:
          - "--config"
          - "/workspace/config.yaml"
        env:
          - name: WANDB_PROJECT
            value: ${project_name}
          - name: WANDB_RUN_ID
            value: ${run_id}
          - name: CUDA_VISIBLE_DEVICES
            value: "0,1,2,3,4,5,6,7"

  # GCS staging bucket (required)
  staging_bucket: gs://my-ml-staging

  # Optional: custom service account
  service_account: my-training-sa@my-project.iam.gserviceaccount.com

  # Optional: network configuration
  network: projects/my-project/global/networks/my-vpc

  # Optional: encryption
  encryption_spec:
    kms_key_name: projects/my-project/locations/us-central1/keyRings/my-ring/cryptoKeys/my-key

run:
  # Job execution parameters
  restart_job_on_worker_restart: false
  timeout: 86400  # 24 hours in seconds

  # Optional: enable TensorBoard
  tensorboard: projects/my-project/locations/us-central1/tensorboards/my-tensorboard
```

---

## Section 3: Authentication & Permissions

### Service Account Creation

**Create service account:**
```bash
# Create service account
gcloud iam service-accounts create wandb-launch-agent \
    --display-name="W&B Launch Agent" \
    --description="Service account for W&B Launch agent to manage Vertex AI jobs"

# Get service account email
SA_EMAIL=$(gcloud iam service-accounts list \
    --filter="displayName:W&B Launch Agent" \
    --format="value(email)")

echo $SA_EMAIL
# Output: wandb-launch-agent@my-project.iam.gserviceaccount.com
```

### Required IAM Roles

**Minimum Vertex AI permissions:**

From [Vertex AI IAM Permissions](https://docs.cloud.google.com/vertex-ai/docs/general/iam-permissions) (accessed 2025-01-31):

| Permission | Resource Scope | Description |
|-----------|---------------|-------------|
| `aiplatform.customJobs.create` | Project | Create Custom Jobs |
| `aiplatform.customJobs.list` | Project | List jobs for monitoring |
| `aiplatform.customJobs.get` | Project | Get job status |
| `aiplatform.customJobs.cancel` | Project | Cancel running jobs |

**Grant Vertex AI User role:**
```bash
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"
```

**Predefined role includes:**
- `aiplatform.customJobs.*`
- `aiplatform.models.list`
- `aiplatform.endpoints.list`

### Storage Permissions

**Cloud Storage access:**
```bash
# Grant Storage Object Admin for staging bucket
gcloud storage buckets add-iam-policy-binding gs://my-staging-bucket \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

# If using separate data bucket
gcloud storage buckets add-iam-policy-binding gs://my-training-data \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer"
```

From [Cloud Storage IAM](https://cloud.google.com/storage/docs/access-control/iam-roles) (accessed 2025-01-31):
- `storage.objectAdmin`: Read, write, delete objects
- `storage.objectViewer`: Read-only access
- Bucket-level or project-level bindings

### Container Registry Permissions

**Artifact Registry access:**
```bash
# Grant Artifact Registry Reader
gcloud artifacts repositories add-iam-policy-binding my-ml-images \
    --location=us-central1 \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.reader"
```

**If agent builds images (with Kaniko):**
```bash
# Grant Artifact Registry Writer
gcloud artifacts repositories add-iam-policy-binding my-ml-images \
    --location=us-central1 \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.writer"
```

### Authentication Methods

**Method 1: Application Default Credentials (ADC)**

For agents running on GCP (Compute Engine, Cloud Run, GKE):
```bash
# Service account automatically attached to VM
# No explicit credentials needed
wandb launch-agent
```

**Method 2: Service Account JSON Key**

For agents running outside GCP:
```bash
# Download service account key
gcloud iam service-accounts keys create sa-key.json \
    --iam-account=${SA_EMAIL}

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"

wandb launch-agent
```

**Security note:** Service account keys are long-lived credentials. Prefer Workload Identity Federation when possible.

### Workload Identity Federation

**For agents running in external clouds (AWS, Azure):**

From [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) (accessed 2025-01-31):
- Keyless authentication from external identity providers
- Short-lived tokens instead of service account keys
- OIDC or SAML 2.0 based

**Setup Workload Identity Pool:**
```bash
# Create workload identity pool
gcloud iam workload-identity-pools create external-pool \
    --location=global \
    --display-name="External Workload Pool"

# Create provider (example: AWS)
gcloud iam workload-identity-pools providers create-aws aws-provider \
    --location=global \
    --workload-identity-pool=external-pool \
    --account-id=123456789012

# Grant service account impersonation
gcloud iam service-accounts add-iam-policy-binding ${SA_EMAIL} \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/external-pool/*"
```

**Configure agent:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credential-config.json"
# credential-config.json contains workload identity provider details
```

### Secret Manager Integration

**Store W&B API key securely:**
```bash
# Create secret
echo -n "your-wandb-api-key" | gcloud secrets create wandb-api-key \
    --data-file=- \
    --replication-policy="automatic"

# Grant access to service account
gcloud secrets add-iam-policy-binding wandb-api-key \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"
```

**Agent retrieves secret:**
```python
# In agent startup script
from google.cloud import secretmanager
import os

client = secretmanager.SecretManagerServiceClient()
secret_name = f"projects/my-project/secrets/wandb-api-key/versions/latest"
response = client.access_secret_version(request={"name": secret_name})
os.environ["WANDB_API_KEY"] = response.payload.data.decode("UTF-8")
```

### Cross-Project Permissions

**If Vertex AI and storage are in different projects:**

```bash
# Grant service account access to remote project
gcloud projects add-iam-policy-binding remote-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

# Update queue config with full resource names
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
      container_spec:
        image_uri: us-central1-docker.pkg.dev/remote-project/repo/image:tag
  staging_bucket: gs://remote-project-staging
```

### Security Best Practices

From [GCP IAM Best Practices](https://cloud.google.com/iam/docs/best-practices-for-securing-service-accounts) (accessed 2025-01-31):

**Principle of least privilege:**
- Grant only required permissions
- Use custom roles for fine-grained control
- Avoid `roles/owner` or `roles/editor`

**Key management:**
- Rotate service account keys regularly
- Delete unused keys
- Prefer Workload Identity over keys

**Audit and monitoring:**
```bash
# View service account activity
gcloud logging read \
    "protoPayload.authenticationInfo.principalEmail=${SA_EMAIL}" \
    --limit=50 \
    --format=json

# Set up alerts for unusual activity
# Via Cloud Monitoring: Metrics → Create Alert Policy
```

### Complete Permission Policy Example

**IAM policy YAML:**
```yaml
bindings:
  # Vertex AI permissions
  - role: roles/aiplatform.user
    members:
      - serviceAccount:wandb-launch-agent@my-project.iam.gserviceaccount.com

  # Storage permissions
  - role: roles/storage.objectAdmin
    members:
      - serviceAccount:wandb-launch-agent@my-project.iam.gserviceaccount.com
    condition:
      title: "Staging bucket only"
      expression: "resource.name.startsWith('projects/_/buckets/my-staging-bucket')"

  # Artifact Registry
  - role: roles/artifactregistry.reader
    members:
      - serviceAccount:wandb-launch-agent@my-project.iam.gserviceaccount.com

  # Secret Manager
  - role: roles/secretmanager.secretAccessor
    members:
      - serviceAccount:wandb-launch-agent@my-project.iam.gserviceaccount.com
    condition:
      title: "W&B secrets only"
      expression: "resource.name.startsWith('projects/my-project/secrets/wandb-')"
```

**Apply policy:**
```bash
gcloud projects set-iam-policy my-project policy.yaml
```

### Troubleshooting Authentication Issues

**Common errors:**

**Error: Permission denied**
```
google.api_core.exceptions.PermissionDenied: 403 The caller does not have permission
```

**Solution:**
```bash
# Check current permissions
gcloud projects get-iam-policy my-project \
    --flatten="bindings[].members" \
    --filter="bindings.members:${SA_EMAIL}"

# Add missing role
gcloud projects add-iam-policy-binding my-project \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"
```

**Error: Credentials not found**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials
```

**Solution:**
```bash
# Set ADC explicitly
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"

# Or on Compute Engine, attach service account
gcloud compute instances set-service-account INSTANCE_NAME \
    --service-account=${SA_EMAIL} \
    --zone=us-central1-a
```

**Verify authentication:**
```bash
# Test as service account
gcloud auth activate-service-account --key-file=sa-key.json
gcloud auth list

# Test Vertex AI access
gcloud ai custom-jobs list --region=us-central1
```

---

## Sources

**W&B Launch Documentation:**
- [Tutorial: Set up W&B Launch on Vertex AI](https://docs.wandb.ai/platform/launch/setup-vertex) - Vertex AI integration guide (accessed 2025-01-31)
- [Set up launch agent](https://docs.wandb.ai/platform/launch/setup-agent-advanced) - Agent configuration and builders (accessed 2025-01-31)
- [Configure launch queue](https://docs.wandb.ai/platform/launch/setup-queue-advanced) - Queue config templates and macros (accessed 2025-01-31)

**Google Cloud Documentation:**
- [Vertex AI access control with IAM](https://docs.cloud.google.com/vertex-ai/docs/general/access-control) - IAM roles and permissions (accessed 2025-01-31)
- [Vertex AI Custom Jobs](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) - CustomJob API reference (accessed 2025-01-31)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) - Keyless authentication (accessed 2025-01-31)
- [Cloud Storage IAM](https://cloud.google.com/storage/docs/access-control/iam-roles) - Storage permissions (accessed 2025-01-31)
- [Secret Manager](https://cloud.google.com/secret-manager/docs) - Secrets management (accessed 2025-01-31)

**Additional References:**
- W&B Launch queue monitoring and observability
- GCP IAM best practices for service accounts
- Artifact Registry authentication methods
