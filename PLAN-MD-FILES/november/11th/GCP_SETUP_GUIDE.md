# GCP_SETUP_GUIDE.md
**Complete Manual Setup: W&B Launch + GCP Vertex AI for ARR-COC Training**

*Step-by-step guide with default project set from the start*

---

## Prerequisites

Before starting, you need:
- ✅ GCP account with billing enabled
- ✅ W&B account (free tier works)
- ✅ `gcloud` CLI installed ([install guide](https://cloud.google.com/sdk/docs/install))
- ✅ Docker installed locally (for image building)
- ✅ Git repository with your code (public or private with credentials)

---

## Part 1: GCP Project Setup (One-Time)

### 1.1 Authenticate and Set Default Project

```bash
# Authenticate with GCP
gcloud auth login

# Set your project ID as a variable (replace with your actual project)
export PROJECT_ID="your-gcp-project-id"

# Set default project (persists across sessions - no need for --project flags!)
gcloud config set project ${PROJECT_ID}

# Verify it's set
gcloud config get-value project
# Should output: your-gcp-project-id

# Set default region (optional but recommended)
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

**✅ From now on, all `gcloud` commands use this project by default!**

### 1.2 Enable Required APIs

```bash
# Enable Vertex AI API (for training jobs)
gcloud services enable aiplatform.googleapis.com

# Enable Artifact Registry API (for Docker images)
gcloud services enable artifactregistry.googleapis.com

# Enable Cloud Storage API (for data/checkpoints)
gcloud services enable storage.googleapis.com

# Enable Compute Engine API (for VMs)
gcloud services enable compute.googleapis.com

# Enable Container Registry API (legacy, for GCR compatibility)
gcloud services enable containerregistry.googleapis.com

# Verify enabled APIs
gcloud services list --enabled | grep -E "aiplatform|artifact|storage|compute"
```

**Wait 2-3 minutes** for APIs to propagate before continuing.

---

## Part 2: Artifact Registry Setup (Docker Images)

### 2.1 Create Artifact Registry Repository

```bash
# Create repository for Launch Docker images
gcloud artifacts repositories create wandb-launch-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="W&B Launch Docker images for ARR-COC training"

# Verify creation
gcloud artifacts repositories list --location=us-central1

# Configure Docker auth for this registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

**Your image URI format will be:**
```
us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/arr-coc-0-1:latest
```

### 2.2 Test Docker Push (Optional)

```bash
# Quick test to verify registry works
docker pull hello-world
docker tag hello-world us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/test:latest
docker push us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/test:latest

# Clean up test image
gcloud artifacts docker images delete \
    us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/test:latest \
    --quiet
```

---

## Part 3: Cloud Storage Setup (Staging + Checkpoints)

### 3.1 Create Staging Bucket (Required by Vertex AI)

```bash
# Staging bucket for Vertex AI job metadata
# MUST be in same region as training jobs (us-central1)
gsutil mb -l us-central1 gs://${PROJECT_ID}-vertex-staging

# Verify
gsutil ls gs://${PROJECT_ID}-vertex-staging
```

### 3.2 Create Checkpoints Bucket (For Model Outputs)

```bash
# Bucket for training checkpoints and artifacts
gsutil mb -l us-central1 gs://${PROJECT_ID}-arr-coc-checkpoints

# Set lifecycle policy (delete checkpoints older than 30 days to save costs)
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-arr-coc-checkpoints

# Clean up temp file
rm lifecycle.json

# Verify buckets
gsutil ls
```

---

## Part 4: Service Account Setup (For Launch Agent)

### 4.1 Create Service Account

```bash
# Create service account for W&B Launch agent
gcloud iam service-accounts create wandb-launch-sa \
    --description="W&B Launch Service Account for ARR-COC training" \
    --display-name="W&B Launch SA"

# Get service account email (save this!)
export SA_EMAIL="wandb-launch-sa@${PROJECT_ID}.iam.gserviceaccount.com"
echo "Service Account Email: ${SA_EMAIL}"
```

### 4.2 Grant Required IAM Permissions

```bash
# Grant Vertex AI User role (create/list/get custom jobs)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

# Grant Storage Object Admin (read/write GCS buckets)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

# Grant Artifact Registry Writer (push Docker images)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

# Grant Logs Writer (for Cloud Logging)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"

# Verify permissions
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --filter="bindings.members:${SA_EMAIL}"
```

### 4.3 Create and Download Service Account Key

```bash
# Create JSON key file
gcloud iam service-accounts keys create wandb-launch-key.json \
    --iam-account=${SA_EMAIL}

# Move to secure location
mkdir -p ~/.gcp-keys
mv wandb-launch-key.json ~/.gcp-keys/

# Set environment variable (add to ~/.bashrc or ~/.zshrc for persistence)
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.gcp-keys/wandb-launch-key.json"

# Verify key works
gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
gcloud auth list

# Switch back to your user account (for interactive use)
gcloud config set account your-email@gmail.com
```

**⚠️ SECURITY:** Keep `wandb-launch-key.json` secure! Add to `.gitignore`. This key has admin access to your GCP resources.

---

## Part 5: W&B Setup

### 5.1 Install W&B CLI and Authenticate

```bash
# Install W&B CLI (latest version)
pip install wandb>=0.17.1

# Login to W&B (opens browser for auth)
wandb login

# Or set API key directly (get from https://wandb.ai/authorize)
export WANDB_API_KEY="your-wandb-api-key-here"
wandb login --relogin
```

### 5.2 Set W&B Secrets (For Container Access)

Go to W&B UI: https://wandb.ai/settings/secrets

Add these secrets:

**1. HF_TOKEN**
- Name: `HF_TOKEN`
- Value: Your HuggingFace token (from https://huggingface.co/settings/tokens)
- Permissions: Read + Write (for checkpoint upload)

**2. WANDB_API_KEY**
- Name: `WANDB_API_KEY`
- Value: Your W&B API key (from https://wandb.ai/authorize)

**3. GCP_SERVICE_ACCOUNT_KEY** (optional, for GCS access from containers)
- Name: `GCP_SERVICE_ACCOUNT_KEY`
- Value: Contents of `~/.gcp-keys/wandb-launch-key.json`

```bash
# Get key contents (copy this to W&B secrets)
cat ~/.gcp-keys/wandb-launch-key.json
```

---

## Part 6: W&B Launch Queue Setup

### 6.1 Create Queue via W&B UI

**Option A: Via Web UI (Recommended for first-time)**

1. Go to https://wandb.ai/launch
2. Click **"Create Queue"**
3. Fill in:
   - **Entity:** Your W&B username (e.g., `newsofpeace2`)
   - **Name:** `vertex-arr-coc-queue`
   - **Resource:** Select **"Google Cloud Vertex AI"**
4. **Configuration (YAML):**

```yaml
# Vertex AI job configuration
spec:
  worker_pool_specs:
    - machine_spec:
        # Machine type: n1-standard-8 (8 vCPUs, 30GB RAM)
        # For GPU-heavy workloads, use n1-highmem-8 or a2-highgpu-1g
        machine_type: n1-standard-8

        # GPU configuration: 1x A100 (40GB VRAM)
        # Options: NVIDIA_TESLA_A100, NVIDIA_TESLA_V100, NVIDIA_TESLA_T4
        accelerator_type: NVIDIA_TESLA_A100
        accelerator_count: 1

        # CRITICAL: Use preemptible for 70% cost savings
        preemptible: true

      # Number of workers (replicas)
      replica_count: 1

      # Container spec (Launch injects image_uri automatically)
      container_spec:
        image_uri: ${image_uri}

        # Optional: Override entrypoint/args
        # command: ["python"]
        # args: ["training/train.py"]

  # Staging bucket (REQUIRED)
  staging_bucket: gs://your-project-id-vertex-staging/

  # Service account for job execution
  service_account: wandb-launch-sa@your-project-id.iam.gserviceaccount.com

  # Optional: Network config (if using VPC)
  # network: projects/your-project-id/global/networks/default

# Launch job config
run:
  restart_job_on_worker_restart: false
```

**Replace:**
- `your-project-id` with your actual project ID

5. Click **"Create Queue"**

**Option B: Via CLI (Advanced)**

```bash
# Create queue config file
cat > queue-config.yaml <<EOF
name: vertex-arr-coc-queue
resource: vertex-ai
config:
  spec:
    worker_pool_specs:
      - machine_spec:
          machine_type: n1-standard-8
          accelerator_type: NVIDIA_TESLA_A100
          accelerator_count: 1
          preemptible: true
        replica_count: 1
        container_spec:
          image_uri: \${image_uri}
    staging_bucket: gs://${PROJECT_ID}-vertex-staging/
    service_account: ${SA_EMAIL}
  run:
    restart_job_on_worker_restart: false
EOF

# Create queue
wandb launch-queue create \
    --entity newsofpeace2 \
    --config queue-config.yaml
```

### 6.2 Verify Queue Creation

```bash
# List queues
wandb launch-queue list --entity newsofpeace2

# Should show: vertex-arr-coc-queue (vertex-ai)
```

---

## Part 7: Launch Agent Setup

The Launch agent runs on your local machine (or a server) and:
1. Polls the W&B queue for jobs
2. Builds Docker images
3. Pushes images to Artifact Registry
4. Submits jobs to Vertex AI

### 7.1 Configure Launch Agent

```bash
# Create config directory
mkdir -p ~/.config/wandb

# Create launch agent config
cat > ~/.config/wandb/launch-config.yaml <<EOF
max_jobs: 1  # Run 1 job at a time
environment:
  type: gcp-vertex
  project: ${PROJECT_ID}
  region: us-central1
queues:
  - vertex-arr-coc-queue
EOF
```

### 7.2 Start Launch Agent

**Terminal 1 (keep running):**

```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.gcp-keys/wandb-launch-key.json"
export WANDB_API_KEY="your-wandb-api-key"

# Start agent
wandb launch-agent \
    --entity newsofpeace2 \
    --queue vertex-arr-coc-queue

# You should see:
# wandb: Starting launch agent for queue vertex-arr-coc-queue
# wandb: Polling for jobs...
```

**Keep this terminal running!** It will poll for jobs and execute them.

---

## Part 8: Prepare Your Training Code

### 8.1 Repository Structure

Your Git repo should look like:

```
your-repo/
├── Dockerfile.wandb          # Custom Dockerfile
├── requirements.txt          # Python dependencies
├── arr_coc/                  # Your ARR-COC code
│   ├── __init__.py
│   ├── texture.py
│   ├── knowing.py
│   ├── balancing.py
│   └── attending.py
├── training/
│   └── train.py             # Training script with W&B integration
└── README.md
```

### 8.2 Create Dockerfile.wandb

**File: `Dockerfile.wandb`**

```dockerfile
# Base on Vertex AI's pre-built PyTorch image (includes CUDA, cuDNN)
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for ARR-COC
RUN pip install --no-cache-dir \
    accelerate==0.25.0 \
    wandb==0.16.0 \
    huggingface-hub==0.19.0 \
    datasets==2.15.0 \
    transformers==4.36.0

# Entry point
ENTRYPOINT ["python", "training/train.py"]
```

### 8.3 Update train.py for W&B Launch

**Minimal changes required** (these work with Launch environment variables):

Add to top of `training/train.py`:

```python
import os
import wandb

# Initialize W&B (Launch injects run info)
wandb.init(
    project=os.getenv("WANDB_PROJECT", "arr-coc-0-1"),
    job_type="train",
    tags=["vertex-ai", "launch", "arr-coc"]
)
```

**That's it!** Your existing train.py already has environment variable support from BUILD_OUT_TRAINING.md.

---

## Part 9: Submit Your First Job

### 9.1 Quick Test (Recommended First)

```bash
# Test with minimal training (10 samples, 1 epoch)
wandb launch \
    --uri "https://github.com/djwar42/arr-coc-0-1.git" \
    --dockerfile Dockerfile.wandb \
    --entry-point "python training/train.py" \
    --project "arr-coc-0-1" \
    --queue "vertex-arr-coc-queue" \
    --name "test-run-quick" \
    -- \
    --max_train_samples 10 \
    --num_epochs 1
```

**What happens:**
1. Launch agent pulls your Git repo
2. Builds Docker image using `Dockerfile.wandb`
3. Pushes image to `us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/`
4. Submits Vertex AI CustomJob with your queue config
5. Job runs on 1x A100 spot instance
6. Logs stream to W&B dashboard

**Monitor:**
- W&B: https://wandb.ai/newsofpeace2/arr-coc-0-1/runs
- GCP: https://console.cloud.google.com/vertex-ai/training/custom-jobs

### 9.2 Full Training Job

After test passes:

```bash
# Full VQAv2 training (3 epochs, all data)
wandb launch \
    --uri "https://github.com/djwar42/arr-coc-0-1.git" \
    --dockerfile Dockerfile.wandb \
    --entry-point "python training/train.py" \
    --project "arr-coc-0-1" \
    --queue "vertex-arr-coc-queue" \
    --name "baseline-v0.1-full" \
    -- \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 1e-5 \
    --output_dir gs://${PROJECT_ID}-arr-coc-checkpoints/baseline-v0.1 \
    --hub_repo_id newsofpeace2/arr-coc-0-1
```

### 9.3 Alternative: Use Pre-Built Image

If you manually build/push image:

```bash
# Build image locally
cd your-repo/
docker build -f Dockerfile.wandb \
    -t us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/arr-coc-0-1:latest .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/arr-coc-0-1:latest

# Launch with pre-built image (no build step)
wandb launch \
    --docker-image us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/arr-coc-0-1:latest \
    --project "arr-coc-0-1" \
    --queue "vertex-arr-coc-queue" \
    --name "prebuilt-test"
```

---

## Part 10: Hyperparameter Sweeps

### 10.1 Create Sweep Configuration

**File: `sweep.yaml`**

```yaml
program: training/train.py
method: bayes
metric:
  name: val/accuracy
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4

  batch_size:
    values: [2, 4, 8]

  num_visual_tokens:
    values: [100, 200, 400]

# Launch queue config
queue: vertex-arr-coc-queue
entity: newsofpeace2
project: arr-coc-0-1
```

### 10.2 Initialize and Run Sweep

```bash
# Initialize sweep
wandb sweep sweep.yaml

# Output: wandb: Created sweep with ID: abcd1234
# Copy the sweep ID

# Start sweep agent (uses Launch queue)
wandb agent newsofpeace2/arr-coc-0-1/abcd1234

# Or launch multiple agents
for i in {1..5}; do
    wandb agent newsofpeace2/arr-coc-0-1/abcd1234 &
done
```

---

## Part 11: Monitoring and Debugging

### 11.1 Check Job Status

**W&B Dashboard:**
```
https://wandb.ai/newsofpeace2/arr-coc-0-1
```

**Vertex AI Console:**
```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}
```

**GCP CLI:**
```bash
# List all custom jobs
gcloud ai custom-jobs list --region=us-central1

# Get specific job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### 11.2 Check Artifact Registry Images

```bash
# List images
gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo

# Get image details
gcloud artifacts docker images describe \
    us-central1-docker.pkg.dev/${PROJECT_ID}/wandb-launch-repo/arr-coc-0-1:latest
```

### 11.3 Check Checkpoints in GCS

```bash
# List checkpoints
gsutil ls gs://${PROJECT_ID}-arr-coc-checkpoints/

# Download checkpoint
gsutil cp -r gs://${PROJECT_ID}-arr-coc-checkpoints/baseline-v0.1/final ./local-checkpoint/
```

---

## Part 12: Cost Optimization

### 12.1 Use Preemptible Instances (Already Configured)

Your queue config has `preemptible: true` which gives **70% cost savings**:

**Pricing (us-central1):**
- A100 on-demand: $3.67/hour
- A100 spot: $1.10/hour (**Save $2.57/hour**)

**For 12-hour training:**
- On-demand: $44.04
- Spot: $13.20
- **Savings: $30.84 (70%)**

### 12.2 Auto-Delete Old Checkpoints

Already configured via lifecycle policy (30 days):

```bash
# Verify lifecycle policy
gsutil lifecycle get gs://${PROJECT_ID}-arr-coc-checkpoints
```

### 12.3 Budget Alerts

Set up billing alerts in GCP Console:

```
https://console.cloud.google.com/billing/budgets
```

Create alert at $100/month (email notifications).

---

## Part 13: Troubleshooting

### Issue 1: "Permission denied" pushing to Artifact Registry

```bash
# Re-authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Verify service account has artifactregistry.writer role
gcloud projects get-iam-policy ${PROJECT_ID} \
    --flatten="bindings[].members" \
    --filter="bindings.members:${SA_EMAIL}"
```

### Issue 2: "Insufficient quota for A100"

```bash
# Check current quotas
gcloud compute project-info describe --project=${PROJECT_ID}

# Request quota increase:
# https://console.cloud.google.com/iam-admin/quotas
# Search: "NVIDIA_A100_GPUS"
# Request: 1-8 GPUs in us-central1
```

Or use T4 GPUs (cheaper, no quota issues):

```yaml
# In queue config
accelerator_type: NVIDIA_TESLA_T4
accelerator_count: 1
```

### Issue 3: Spot instance keeps getting preempted

```bash
# Try different region (less contention)
# Update queue config region to: us-east1 or europe-west4

# Or use on-demand for critical runs
preemptible: false
```

### Issue 4: W&B logs not appearing

```bash
# Check WANDB_API_KEY is set in container
# Verify in Launch agent logs

# Test W&B auth locally
python3 -c "import wandb; print('✓ Authenticated') if wandb.api.api_key else print('✗ Not authenticated')"
```

### Issue 5: Docker build fails

```bash
# Check agent logs
# Common issues:
# - requirements.txt missing packages
# - Base image not accessible
# - Network issues

# Test build locally first
docker build -f Dockerfile.wandb -t test-image .
```

---

## Part 14: Complete Checklist

Before running full training:

**GCP Setup:**
- [ ] Default project set (`gcloud config set project`)
- [ ] APIs enabled (aiplatform, artifactregistry, storage, compute)
- [ ] Artifact Registry repository created
- [ ] Staging bucket created (`gs://${PROJECT_ID}-vertex-staging`)
- [ ] Checkpoints bucket created (`gs://${PROJECT_ID}-arr-coc-checkpoints`)
- [ ] Service account created with correct permissions
- [ ] Service account key downloaded and env var set

**W&B Setup:**
- [ ] W&B CLI installed and authenticated
- [ ] W&B secrets configured (HF_TOKEN, WANDB_API_KEY)
- [ ] Launch queue created with correct config
- [ ] Launch agent running (terminal open)

**Code Setup:**
- [ ] Git repo accessible (public or agent has credentials)
- [ ] `Dockerfile.wandb` exists in repo root
- [ ] `requirements.txt` complete
- [ ] `train.py` has W&B integration
- [ ] Code tested locally

**Testing:**
- [ ] Quick test job submitted (10 samples)
- [ ] Job shows in Vertex AI console
- [ ] Logs appear in W&B dashboard
- [ ] Checkpoints saved to GCS
- [ ] No errors in agent logs

**Ready for full training!**

---

## Summary

**What we set up:**
1. ✅ GCP project with default project configured
2. ✅ Artifact Registry for Docker images
3. ✅ GCS buckets for staging + checkpoints
4. ✅ Service account with correct permissions
5. ✅ W&B Launch queue targeting Vertex AI
6. ✅ Launch agent polling queue
7. ✅ Training code with W&B integration
8. ✅ Dockerfile based on Vertex PyTorch image
9. ✅ Spot instances for 70% cost savings

**Next steps:**
1. Submit quick test job (10 samples)
2. Verify end-to-end flow works
3. Run full training (3 epochs VQAv2)
4. Start hyperparameter sweeps
5. Scale to multi-GPU if needed

**Cost estimate:**
- Quick test (10 samples, 5 min): **$0.09** (spot A100)
- Full training (3 epochs, 12 hours): **$13.20** (spot A100)

---

**Questions or issues?** Check the Troubleshooting section or GCP/W&B docs.

---

---

---

# AUTOMATED SETUP SCRIPT

**⚡ Quick Setup Alternative to Manual Steps Above**

Instead of following Parts 1-13 manually, you can use this automated script to set up GCP + W&B Launch in ~5 minutes.

**What this script automates:**
- ✅ Part 1: Enable all required GCP APIs
- ✅ Part 2: Create Artifact Registry repository
- ✅ Part 3: Create GCS buckets (staging + checkpoints)
- ✅ Part 4: Create service account with IAM permissions
- ✅ Part 5: Set W&B secrets (prompts for HF token)
- ✅ Part 6: Create W&B Launch queue

**What you must do first:**
1. Create GCP account with billing enabled
2. Run `gcloud auth login` (browser OAuth)
3. Run `gcloud config set project YOUR_PROJECT_ID`
4. Run `wandb login` (browser auth)

**What remains manual:**
- Running the Launch agent (persistent terminal process)
- Submitting jobs

---

## Automated Setup Script

**File: `setup-gcp.sh`**

```bash
#!/bin/bash
# setup-gcp.sh
# One-time automated GCP + W&B Launch setup for ARR-COC training
#
# Prerequisites:
# 1. gcloud auth login (already logged in)
# 2. gcloud config set project <PROJECT_ID> (default project set)
# 3. wandb login (already authenticated)
#
# This script automates Parts 1-6 of GCP_SETUP_GUIDE.md

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ARR-COC GCP + W&B Launch Setup (Automated)              ║${NC}"
echo -e "${BLUE}║  This script sets up everything for Vertex AI training   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# PART 0: Prerequisites Check
# ============================================================================

echo -e "${YELLOW}=== Prerequisites Check ===${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi
echo -e "${GREEN}✓ gcloud CLI installed${NC}"

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo -e "${RED}✗ wandb CLI not found. Install: pip install wandb${NC}"
    exit 1
fi
echo -e "${GREEN}✓ wandb CLI installed${NC}"

# Check gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${RED}✗ Not authenticated with gcloud. Run: gcloud auth login${NC}"
    exit 1
fi
GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1)
echo -e "${GREEN}✓ Authenticated with gcloud as: ${GCLOUD_ACCOUNT}${NC}"

# Check if default project is set
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo -e "${RED}✗ No default GCP project set.${NC}"
    echo -e "${YELLOW}  Run: gcloud config set project YOUR_PROJECT_ID${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GCP Project: ${PROJECT_ID}${NC}"

# Check wandb authentication
WANDB_CHECK=$(python3 -c "import wandb; print('ok') if wandb.api.api_key else print('fail')" 2>/dev/null || echo "fail")
if [ "$WANDB_CHECK" != "ok" ]; then
    echo -e "${RED}✗ Not authenticated with W&B. Run: wandb login${NC}"
    exit 1
fi
# Get W&B username
WANDB_USER=$(python3 -c "import wandb; api = wandb.Api(); print(api.viewer.username)" 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ Authenticated with W&B as: ${WANDB_USER}${NC}"

echo ""
echo -e "${BLUE}Prerequisites check complete!${NC}"
echo ""

# ============================================================================
# PART 0.5: User Inputs
# ============================================================================

echo -e "${YELLOW}=== Configuration ===${NC}"

# W&B Entity (default to username)
read -p "Enter your W&B username/entity (default: ${WANDB_USER}): " INPUT_ENTITY
WANDB_ENTITY="${INPUT_ENTITY:-$WANDB_USER}"
echo -e "${GREEN}W&B Entity: ${WANDB_ENTITY}${NC}"

# HuggingFace Token (check if already set in W&B secrets first)
echo ""
echo "Checking if HuggingFace token already set in W&B secrets..."
HF_TOKEN_EXISTS=$(wandb secret list 2>/dev/null | grep -c "HF_TOKEN" || echo "0")
if [ "$HF_TOKEN_EXISTS" -gt 0 ]; then
    echo -e "${GREEN}✓ HF_TOKEN already exists in W&B secrets${NC}"
    read -p "Update HF_TOKEN? (y/N): " UPDATE_HF
    if [[ "$UPDATE_HF" =~ ^[Yy]$ ]]; then
        read -sp "Enter your HuggingFace token (https://huggingface.co/settings/tokens): " HF_TOKEN
        echo ""
        SET_HF_TOKEN=true
    else
        SET_HF_TOKEN=false
    fi
else
    read -sp "Enter your HuggingFace token (https://huggingface.co/settings/tokens): " HF_TOKEN
    echo ""
    SET_HF_TOKEN=true
fi

# Region (default us-central1)
read -p "Enter GCP region (default: us-central1): " INPUT_REGION
REGION="${INPUT_REGION:-us-central1}"
echo -e "${GREEN}Region: ${REGION}${NC}"

# Derived variables
SA_NAME="wandb-launch-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
STAGING_BUCKET="gs://${PROJECT_ID}-vertex-staging"
CHECKPOINTS_BUCKET="gs://${PROJECT_ID}-arr-coc-checkpoints"
REGISTRY_REPO="wandb-launch-repo"
QUEUE_NAME="vertex-arr-coc-queue"

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Project ID:        ${PROJECT_ID}"
echo "  Region:            ${REGION}"
echo "  W&B Entity:        ${WANDB_ENTITY}"
echo "  Service Account:   ${SA_EMAIL}"
echo "  Staging Bucket:    ${STAGING_BUCKET}"
echo "  Checkpoints:       ${CHECKPOINTS_BUCKET}"
echo "  Registry:          ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY_REPO}"
echo "  Queue:             ${QUEUE_NAME}"
echo ""

read -p "Continue with setup? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""

# ============================================================================
# PART 1: Enable GCP APIs
# ============================================================================

echo -e "${YELLOW}=== Part 1: Enabling GCP APIs ===${NC}"

APIS=(
    "aiplatform.googleapis.com"
    "artifactregistry.googleapis.com"
    "storage.googleapis.com"
    "compute.googleapis.com"
    "containerregistry.googleapis.com"
)

for API in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:${API}" --format="value(name)" 2>/dev/null | grep -q "${API}"; then
        echo -e "${GREEN}✓ ${API} already enabled${NC}"
    else
        echo "  Enabling ${API}..."
        gcloud services enable "${API}" --quiet
        echo -e "${GREEN}✓ ${API} enabled${NC}"
    fi
done

echo -e "${GREEN}✓ All APIs enabled${NC}"
echo ""

# Wait for API propagation
echo "Waiting 10 seconds for APIs to propagate..."
sleep 10

# ============================================================================
# PART 2: Artifact Registry
# ============================================================================

echo -e "${YELLOW}=== Part 2: Setting up Artifact Registry ===${NC}"

# Check if repository exists
if gcloud artifacts repositories describe ${REGISTRY_REPO} \
    --location=${REGION} &>/dev/null; then
    echo -e "${GREEN}✓ Artifact Registry repository '${REGISTRY_REPO}' already exists${NC}"
else
    echo "  Creating Artifact Registry repository..."
    gcloud artifacts repositories create ${REGISTRY_REPO} \
        --repository-format=docker \
        --location=${REGION} \
        --description="W&B Launch Docker images for ARR-COC training"
    echo -e "${GREEN}✓ Created repository: ${REGISTRY_REPO}${NC}"
fi

# Configure Docker auth
echo "  Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
echo -e "${GREEN}✓ Docker auth configured${NC}"
echo ""

# ============================================================================
# PART 3: Cloud Storage Buckets
# ============================================================================

echo -e "${YELLOW}=== Part 3: Creating Cloud Storage Buckets ===${NC}"

# Staging bucket
if gsutil ls ${STAGING_BUCKET} &>/dev/null; then
    echo -e "${GREEN}✓ Staging bucket already exists: ${STAGING_BUCKET}${NC}"
else
    echo "  Creating staging bucket..."
    gsutil mb -l ${REGION} ${STAGING_BUCKET}
    echo -e "${GREEN}✓ Created staging bucket: ${STAGING_BUCKET}${NC}"
fi

# Checkpoints bucket
if gsutil ls ${CHECKPOINTS_BUCKET} &>/dev/null; then
    echo -e "${GREEN}✓ Checkpoints bucket already exists: ${CHECKPOINTS_BUCKET}${NC}"
else
    echo "  Creating checkpoints bucket..."
    gsutil mb -l ${REGION} ${CHECKPOINTS_BUCKET}

    # Set lifecycle policy (delete after 30 days)
    cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF
    gsutil lifecycle set /tmp/lifecycle.json ${CHECKPOINTS_BUCKET}
    rm /tmp/lifecycle.json

    echo -e "${GREEN}✓ Created checkpoints bucket with lifecycle policy (30 days)${NC}"
fi
echo ""

# ============================================================================
# PART 4: Service Account + IAM
# ============================================================================

echo -e "${YELLOW}=== Part 4: Setting up Service Account ===${NC}"

# Check if service account exists
if gcloud iam service-accounts describe ${SA_EMAIL} &>/dev/null; then
    echo -e "${GREEN}✓ Service account already exists: ${SA_EMAIL}${NC}"
else
    echo "  Creating service account..."
    gcloud iam service-accounts create ${SA_NAME} \
        --description="W&B Launch Service Account for ARR-COC training" \
        --display-name="W&B Launch SA"
    echo -e "${GREEN}✓ Created service account: ${SA_EMAIL}${NC}"
fi

# Grant IAM roles
echo "  Granting IAM permissions..."

ROLES=(
    "roles/aiplatform.user"
    "roles/storage.objectAdmin"
    "roles/artifactregistry.writer"
    "roles/logging.logWriter"
)

for ROLE in "${ROLES[@]}"; do
    # Check if role already granted
    if gcloud projects get-iam-policy ${PROJECT_ID} \
        --flatten="bindings[].members" \
        --filter="bindings.members:serviceAccount:${SA_EMAIL} AND bindings.role:${ROLE}" \
        --format="value(bindings.role)" 2>/dev/null | grep -q "${ROLE}"; then
        echo -e "${GREEN}  ✓ ${ROLE} already granted${NC}"
    else
        gcloud projects add-iam-policy-binding ${PROJECT_ID} \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="${ROLE}" \
            --quiet &>/dev/null
        echo -e "${GREEN}  ✓ Granted ${ROLE}${NC}"
    fi
done

# Create service account key
KEY_PATH="${HOME}/.gcp-keys/wandb-launch-key.json"
mkdir -p "${HOME}/.gcp-keys"

# Check if GOOGLE_APPLICATION_CREDENTIALS is already set and valid
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo -e "${GREEN}✓ GOOGLE_APPLICATION_CREDENTIALS already set: ${GOOGLE_APPLICATION_CREDENTIALS}${NC}"
    read -p "Use existing credentials or create new key? (Use existing/create new) [e/N]: " USE_EXISTING
    if [[ "$USE_EXISTING" =~ ^[Ee]$ ]]; then
        KEY_PATH="$GOOGLE_APPLICATION_CREDENTIALS"
        echo -e "${GREEN}✓ Using existing credentials${NC}"
    else
        # User wants new key
        if [ -f "${KEY_PATH}" ]; then
            echo -e "${YELLOW}⚠ Service account key already exists: ${KEY_PATH}${NC}"
            read -p "Recreate key? This will invalidate the old key! (y/N): " RECREATE_KEY
            if [[ "$RECREATE_KEY" =~ ^[Yy]$ ]]; then
                gcloud iam service-accounts keys create "${KEY_PATH}" \
                    --iam-account=${SA_EMAIL}
                echo -e "${GREEN}✓ Created new service account key: ${KEY_PATH}${NC}"
            fi
        else
            gcloud iam service-accounts keys create "${KEY_PATH}" \
                --iam-account=${SA_EMAIL}
            echo -e "${GREEN}✓ Created service account key: ${KEY_PATH}${NC}"
        fi
    fi
elif [ -f "${KEY_PATH}" ]; then
    echo -e "${GREEN}✓ Service account key already exists: ${KEY_PATH}${NC}"
    echo -e "${YELLOW}  Note: GOOGLE_APPLICATION_CREDENTIALS not set, but key file exists${NC}"
    read -p "Use existing key? (y/N): " USE_KEY
    if [[ ! "$USE_KEY" =~ ^[Yy]$ ]]; then
        read -p "Recreate key? This will invalidate the old key! (y/N): " RECREATE_KEY
        if [[ "$RECREATE_KEY" =~ ^[Yy]$ ]]; then
            gcloud iam service-accounts keys create "${KEY_PATH}" \
                --iam-account=${SA_EMAIL}
            echo -e "${GREEN}✓ Created new service account key: ${KEY_PATH}${NC}"
        fi
    fi
else
    gcloud iam service-accounts keys create "${KEY_PATH}" \
        --iam-account=${SA_EMAIL}
    echo -e "${GREEN}✓ Created service account key: ${KEY_PATH}${NC}"
fi

echo ""

# ============================================================================
# PART 5: W&B Secrets
# ============================================================================

echo -e "${YELLOW}=== Part 5: Setting W&B Secrets ===${NC}"

# Set HF_TOKEN if needed
if [ "$SET_HF_TOKEN" = true ]; then
    echo "  Setting HF_TOKEN in W&B secrets..."
    echo "${HF_TOKEN}" | wandb secret set HF_TOKEN
    echo -e "${GREEN}✓ HF_TOKEN set in W&B secrets${NC}"
fi

# Set WANDB_API_KEY (auto-detect from local config)
WANDB_API_KEY=$(python3 -c "import wandb; print(wandb.api.api_key)" 2>/dev/null || echo "")
if [ -n "$WANDB_API_KEY" ]; then
    echo "  Setting WANDB_API_KEY in W&B secrets..."
    echo "${WANDB_API_KEY}" | wandb secret set WANDB_API_KEY
    echo -e "${GREEN}✓ WANDB_API_KEY set in W&B secrets${NC}"
fi

echo ""

# ============================================================================
# PART 6: W&B Launch Queue
# ============================================================================

echo -e "${YELLOW}=== Part 6: Creating W&B Launch Queue ===${NC}"

# Check if queue already exists
QUEUE_EXISTS=$(wandb launch-queue list --entity ${WANDB_ENTITY} 2>/dev/null | grep -c "${QUEUE_NAME}" || echo "0")

if [ "$QUEUE_EXISTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Queue '${QUEUE_NAME}' already exists${NC}"
    read -p "Recreate queue? (y/N): " RECREATE_QUEUE
    if [[ ! "$RECREATE_QUEUE" =~ ^[Yy]$ ]]; then
        echo "  Skipping queue creation"
    else
        echo "  Note: Delete queue manually at https://wandb.ai/${WANDB_ENTITY}/launch"
        echo "  Then re-run this script"
        exit 1
    fi
else
    # Create queue config
    cat > /tmp/queue-config.yaml <<EOF
name: ${QUEUE_NAME}
resource: vertex-ai
config:
  spec:
    worker_pool_specs:
      - machine_spec:
          machine_type: a2-highgpu-1g
          accelerator_type: NVIDIA_TESLA_A100
          accelerator_count: 1
          preemptible: true
        replica_count: 1
        container_spec:
          image_uri: \${image_uri}
    staging_bucket: ${STAGING_BUCKET}/
    service_account: ${SA_EMAIL}
  run:
    restart_job_on_worker_restart: false
EOF

    echo "  Creating W&B Launch queue..."
    wandb launch-queue create \
        --entity ${WANDB_ENTITY} \
        --config /tmp/queue-config.yaml

    rm /tmp/queue-config.yaml
    echo -e "${GREEN}✓ Created queue: ${QUEUE_NAME}${NC}"
fi

echo ""

# ============================================================================
# PART 7: Summary & Next Steps
# ============================================================================

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup Complete! ✅                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}What was created:${NC}"
echo "  ✓ GCP APIs enabled (Vertex AI, Artifact Registry, Storage)"
echo "  ✓ Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REGISTRY_REPO}"
echo "  ✓ Staging bucket: ${STAGING_BUCKET}"
echo "  ✓ Checkpoints bucket: ${CHECKPOINTS_BUCKET}"
echo "  ✓ Service account: ${SA_EMAIL}"
echo "  ✓ Service account key: ${KEY_PATH}"
echo "  ✓ W&B secrets configured (HF_TOKEN, WANDB_API_KEY)"
echo "  ✓ W&B Launch queue: ${QUEUE_NAME}"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Add to your shell profile (~/.bashrc or ~/.zshrc):"
echo "   ${BLUE}export GOOGLE_APPLICATION_CREDENTIALS=\"${KEY_PATH}\"${NC}"
echo ""
echo "2. Start the W&B Launch agent (keep running in a terminal):"
echo "   ${BLUE}wandb launch-agent --entity ${WANDB_ENTITY} --queue ${QUEUE_NAME}${NC}"
echo ""
echo "3. Submit a test job:"
echo "   ${BLUE}wandb launch \\${NC}"
echo "   ${BLUE}  --uri https://github.com/your-user/arr-coc-0-1.git \\${NC}"
echo "   ${BLUE}  --queue ${QUEUE_NAME} \\${NC}"
echo "   ${BLUE}  --project arr-coc-0-1 \\${NC}"
echo "   ${BLUE}  --name test-run${NC}"
echo ""
echo -e "${YELLOW}Costs (spot instances):${NC}"
echo "  • 1x A100 40GB: \$1.10/hour"
echo "  • Quick test (5 min): ~\$0.09"
echo "  • Full training (12 hrs): ~\$13.20"
echo ""

echo -e "${GREEN}Setup complete! Ready to train ARR-COC on Vertex AI.${NC}"
echo ""
echo "¯\\_(ツ)_/¯ Happy training!"
```

---

## Usage Instructions

### 1. Save the Script

```bash
# Navigate to training directory
cd code/arr-coc-0-1/training/

# Script already exists: setup-gcp.sh
# Make executable if needed
chmod +x setup-gcp.sh
```

### 2. Complete Prerequisites

```bash
# Install gcloud CLI (if needed)
# macOS: brew install --cask google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# Install wandb
pip install wandb

# Authenticate with GCP (browser OAuth)
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Authenticate with W&B (browser OAuth)
wandb login
```

### 3. Run the Setup Script

```bash
# Run the automated setup
./setup-gcp.sh
```

**The script will:**
1. ✅ Check all prerequisites (gcloud/wandb auth, project set)
2. ✅ Prompt for W&B entity and HuggingFace token (if not already set)
3. ✅ Enable all required GCP APIs
4. ✅ Create Artifact Registry repository
5. ✅ Create GCS buckets (staging + checkpoints with lifecycle)
6. ✅ Create service account with all IAM permissions
7. ✅ Download service account key to `~/.gcp-keys/`
8. ✅ Set W&B secrets (HF_TOKEN, WANDB_API_KEY)
9. ✅ Create W&B Launch queue for Vertex AI
10. ✅ Display next steps (start agent, submit jobs)

**Time:** ~3-5 minutes (mostly waiting for API propagation)

### 4. After Script Completes

**Start the Launch Agent (keep terminal running):**

```bash
# Export service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.gcp-keys/wandb-launch-key.json"

# Start the agent
wandb launch-agent --entity YOUR_USERNAME --queue vertex-arr-coc-queue
```

**Submit a test job:**

```bash
wandb launch \
  --uri https://github.com/your-user/arr-coc-0-1.git \
  --queue vertex-arr-coc-queue \
  --project arr-coc-0-1 \
  --name test-run
```

---

## What This Automates

**Automated (Parts 1-6):**
- ✅ Enable 5 GCP APIs
- ✅ Create Artifact Registry repository
- ✅ Configure Docker auth
- ✅ Create 2 GCS buckets with lifecycle policy
- ✅ Create service account
- ✅ Grant 4 IAM roles
- ✅ Generate service account key
- ✅ Set W&B secrets
- ✅ Create W&B Launch queue

**Remains Manual:**
- Running `wandb launch-agent` (persistent process)
- Submitting jobs with `wandb launch`

---

## Troubleshooting

### Script fails at "Prerequisites Check"

**gcloud not authenticated:**
```bash
gcloud auth login
```

**No default project:**
```bash
gcloud config set project YOUR_PROJECT_ID
```

**wandb not authenticated:**
```bash
wandb login
```

### "APIs not enabled" errors

Wait 2-3 minutes after script completes, then retry the failing command.

### "Permission denied" errors

Ensure your GCP account has `roles/owner` or `roles/editor` on the project.

---

¯\\_(ツ)_/¯ Happy training!
