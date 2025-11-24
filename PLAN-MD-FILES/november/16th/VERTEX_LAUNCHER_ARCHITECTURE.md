# Vertex AI Launcher Architecture - Complete Analysis

**Project**: arr-coc-0-1
**Component**: arr-vertex-launcher (W&B Launch → Vertex AI bridge)
**Date**: 2025-11-16
**Version**: 2.0 (Semi-persistent runner design)

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Breakdown](#component-breakdown)
3. [Queue & Timeout Management](#queue--timeout-management)
4. [Complete Launch Flow](#complete-launch-flow)
5. [Fatal Error Detection](#fatal-error-detection)
6. [Single-Job Execution](#single-job-execution)
7. [Shutdown & Cleanup](#shutdown--cleanup)
8. [Code References](#code-references)

---

## Architecture Overview

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LAUNCH FLOW                             │
└─────────────────────────────────────────────────────────────────┘

User runs: python training/cli.py launch
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1-6: BUILD IMAGES                                          │
│ - arr-pytorch-base (PyTorch from source, 2-4 hours)             │
│ - arr-ml-stack (ML dependencies)                                │
│ - arr-trainer (Training code)                                   │
│ - arr-vertex-launcher (This component!)                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: SUBMIT TO W&B LAUNCH QUEUE                              │
│ - Job config as JSON                                            │
│ - Pre-built training image URI                                  │
│ - Queue: "vertex-ai-queue"                                      │
│ - State: PENDING (waiting for agent)                            │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8-10: CREATE CLOUD RUN JOB (arr-vertex-launcher)           │
│ - Container: arr-vertex-launcher:HASH                           │
│ - Args: --queue vertex-ai-queue --max-jobs 1                    │
│ - Timeout: 60 minutes (task-timeout)                            │
│ - Max retries: 0 (fail fast, no retries)                        │
│ - Resources: 2 CPU, 2Gi RAM                                     │
│ - Secrets: WANDB_API_KEY from Secret Manager                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 11: EXECUTE CLOUD RUN JOB                                  │
│ - gcloud run jobs execute vertex-ai-launcher                    │
│ - Starts container                                              │
│ - Runs entrypoint-wrapper.sh                                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ INSIDE CONTAINER: entrypoint-wrapper.sh                         │
│ - Starts W&B Launch agent (background process)                  │
│ - Monitors logs every 5 seconds for fatal errors                │
│ - Detects 11 error patterns (machine type, quota, 500/503, etc) │
│ - Exits immediately on fatal error (no 60min wait!)             │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ W&B LAUNCH AGENT (wandb launch-agent)                           │
│ - Polls queue: vertex-ai-queue                                  │
│ - Pulls FIRST job (FIFO order)                                  │
│ - Downloads training code                                       │
│ - Builds Vertex AI CustomJob spec                               │
│ - Submits to Vertex AI using gcloud/Python SDK                  │
│ - Max jobs: 1 (processes ONE job, then exits!)                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ VERTEX AI CUSTOM JOB                                            │
│ - Starts training container (arr-trainer)                       │
│ - Runs training code                                            │
│ - Logs to W&B run                                               │
│ - GPU: 1x NVIDIA L4 (or T4)                                     │
│ - Timeout: 24 hours (Vertex AI job timeout)                     │
└─────────────────────────────────────────────────────────────────┘
    ↓
Training completes → W&B run finishes → Agent exits → Cloud Run execution terminates
```

---

## Component Breakdown

### 1. Docker Image: `arr-vertex-launcher`

**Base Image**: `wandb/launch-agent:latest`
**Purpose**: Poll W&B queue and submit jobs to Vertex AI
**Location**: `training/images/arr-vertex-launcher/`

**Dockerfile Key Steps**:
```dockerfile
FROM wandb/launch-agent:latest

# Install gcloud CLI (195MB download)
RUN apk add python3 py3-pip curl bash && \
    curl -O https://dl.google.com/dl/cloudsdk/.../google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xf google-cloud-cli-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh

# Create W&B Launch config (noop builder - uses pre-built image)
RUN printf 'builder:\n  type: noop\n' > /home/launch_agent/.config/wandb/launch-config.yaml

# Copy fatal error detection wrapper
COPY entrypoint-wrapper.sh /usr/local/bin/entrypoint-wrapper.sh

# Override entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint-wrapper.sh"]
```

**Why `noop` builder?**
- The launcher doesn't build Docker images (already built by CLI!)
- It just uses the pre-built `arr-trainer:HASH` image from Artifact Registry
- Config: `builder: type: noop` tells W&B "don't build, use container_spec"

**File**: `training/images/arr-vertex-launcher/Dockerfile:1-148`

---

### 2. Entrypoint Wrapper: `entrypoint-wrapper.sh`

**Purpose**: Monitor W&B agent and exit immediately on fatal errors
**Version**: 1.1 (Enhanced 500/503 error logging - 2025-11-16)

**Why This Exists**:
- W&B agent retries errors for ~60 minutes by default
- Cloud Run job has 60min timeout
- Without wrapper: Job hangs for full 60min on quota/permission errors
- **With wrapper: Exits in ~5 seconds on fatal errors!**

**Monitoring Loop** (every 5 seconds):
```bash
while kill -0 "$AGENT_PID" 2>/dev/null; do
    sleep 5

    # Check for 11 fatal error patterns:
    # 1. Machine type not supported
    # 2. InvalidArgument: 400
    # 3. PermissionDenied: 403
    # 4. NotFound: 404
    # 5. QuotaExceeded / ResourceExhausted
    # 6. Repeated failures (3+ in 50 lines)
    # 7. Unhandled Python exceptions
    # 8. HTTP 4xx/5xx errors
    # 9. W&B initialization failures
    # 10. Container image pull failures
    # 11. (Generic 500/503 removed - caused false positives)

    # If fatal error detected:
    show_error_context "$pattern" "$description"  # Show full context
    kill "$AGENT_PID"  # Terminate agent
    exit 1  # Exit wrapper with failure
done
```

**Error Context Display** (v1.1 enhancement):
```bash
show_error_context() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 BAILOUT TRIGGER: $description"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Show ALL matching lines with line numbers
    tail -100 "$LOG_FILE" | grep -n -E "$pattern" | sed 's/^/  ► /'

    # Show full log context (last 100 lines)
    tail -100 "$LOG_FILE"
}
```

**File**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh:1-163`

---

### 3. W&B Launch Queue Submission

**Function**: `_submit_to_wandb()`
**Location**: `training/cli/launch/core.py:1029-1071`

**What It Does**:
1. Reads config from `.training` file
2. Creates W&B Launch queue item
3. Includes pre-built training image URI
4. Returns run ID

**Code**:
```python
def _submit_to_wandb(helper, config, status, training_image):
    entity = config.get("WANDB_ENTITY", "newsofpeace2")
    project = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    # Submit job to queue with pre-built image
    run_id, output = helper.submit_job(config, training_image)

    return run_id
```

**Queue Configuration**:
- **Entity**: `newsofpeace2`
- **Project**: `arr-coc-0-1`
- **Queue**: `vertex-ai-queue`
- **State**: `pending` (waiting for agent to pick up)

**W&B Launch Command** (exact):
```bash
wandb launch \
  --docker-image us-central1-docker.pkg.dev/.../arr-trainer:HASH \
  --project arr-coc-0-1 \
  --entity newsofpeace2 \
  --queue vertex-ai-queue \
  --resource gcp-vertex \
  --entry-point "python training/train.py" \
  --config launch_config.json \
  --name spazzy-forest-1762670954  # Generated cool name
```

**Job Payload** (launch_config.json):
```json
{
  "overrides": {
    "run_config": {
      "BASE_MODEL": "Qwen/Qwen2-VL-2B-Instruct",
      "NUM_VISUAL_TOKENS": "256",
      "LEARNING_RATE": "1e-5",
      "BATCH_SIZE": "2",
      "GRADIENT_ACCUMULATION_STEPS": "4",
      "NUM_EPOCHS": "3",
      "SAVE_EVERY_N_STEPS": "50",
      "SEED": "42",
      "WANDB_PROJECT": "arr-coc-0-1",
      "HF_HUB_REPO_ID": "NorthHead/arr-coc-0-1",
      "DATASET_NAME": "NorthHead/arr-coc-texture-dataset",
      "MAX_TRAIN_SAMPLES": "500"
    }
  },
  "resource_args": {
    "gcp-vertex": {
      "spec": {
        "worker_pool_specs": [{
          "machine_spec": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1
          },
          "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 200
          },
          "replica_count": 1,
          "container_spec": {
            "image_uri": "us-central1-docker.pkg.dev/.../arr-trainer:HASH"
          }
        }],
        "staging_bucket": "gs://PROJECT-ID-staging"
      },
      "run": {
        "restart_job_on_worker_restart": false
      }
    }
  }
}
```

**Critical Details**:
- `--docker-image`: Creates **Docker image job** (no building, uses pre-built image)
- `container_spec.image_uri`: Specified TWICE (CLI arg + config for Vertex AI)
- `boot_disk_size_gb: 200`: **200GB SSD** (fixes "low on disk" errors during image pull)
- `restart_job_on_worker_restart: false`: Don't auto-restart failed jobs

**Git Remote Trick** (forces local file packaging):
```python
# Temporarily rename 'origin' remote during submission
# This prevents W&B from storing GitHub URL (forces local artifact)
git remote rename origin origin-changed-for-wandb-launch
wandb launch ...  # Packages local files
git remote rename origin-changed-for-wandb-launch origin  # Restore
```

**Helper Implementation**:
`training/cli/shared/wandb_helper.py:185-399`

---

### 4. Cloud Run Job Creation

**Function**: `_create_cloud_run_job()`
**Location**: `training/cli/launch/core.py:3104-3315`

**Critical Parameters**:

```python
create_cmd = [
    "gcloud", "run", "jobs", "create", job_name,
    "--image", runner_image,  # arr-vertex-launcher:HASH
    "--region", region,  # us-central1

    # ⚡ CRITICAL: Single-job execution!
    f"--args=-q,{queue},-e,{entity},--max-jobs,1",

    # Secrets
    f"--set-secrets=WANDB_API_KEY={secret_name}:latest",

    # Service account
    f"--service-account={sa_email}",

    # Environment
    f"--set-env-vars=CLOUDSDK_COMPUTE_REGION={region}",

    # ⏱️ TIMEOUTS & RETRIES
    "--max-retries", "0",  # No retries! Fail fast!
    "--task-timeout", "60m",  # 1 hour max (was 40m, increased)

    # Resources
    "--memory", "2Gi",
    "--cpu", "2",
]
```

**Breakdown**:

| Parameter | Value | Why? |
|-----------|-------|------|
| `--args` | `-q vertex-ai-queue -e newsofpeace2 --max-jobs 1` | **Queue name + Single job limit** |
| `--max-retries` | `0` | **Fail fast, no retries** (wrapper exits on fatal errors) |
| `--task-timeout` | `60m` | **1 hour max** (increased from 40m after hitting timeout) |
| `--memory` | `2Gi` | **2GB RAM** (enough for gcloud CLI + agent) |
| `--cpu` | `2` | **2 vCPUs** (lightweight agent workload) |
| `--set-secrets` | `WANDB_API_KEY=wandb-api-key:latest` | **Inject W&B API key from Secret Manager** |

**Idempotency**:
- First launch: Creates job
- Subsequent launches: Checks config hash, only updates if changed
- Config unchanged: Skips update (~0.5s vs ~5s)

**File**: `training/cli/launch/core.py:3104-3315`

---

### 5. Cloud Run Job Execution

**Function**: `_execute_runner()`
**Location**: `training/cli/launch/core.py:3317-3425`

**What Happens**:
```python
def _execute_runner(config, region, job_name, status):
    # Start Cloud Run job execution
    execute_cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        "--region", region,
    ]

    execute_result = subprocess.run(
        execute_cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes to START the job
    )

    # Extract execution name from output
    # Format: "Executing job vertex-ai-launcher-XXXXXXX..."
    execution_name = parse_execution_name(execute_result.stdout)

    return execution_name
```

**Timeline**:
1. **0-30s**: Cloud Run allocates container resources
2. **30-60s**: Container starts, pulls image (if not cached)
3. **60-90s**: `entrypoint-wrapper.sh` starts W&B agent
4. **90-120s**: Agent connects to W&B queue, pulls job
5. **120-180s**: Agent submits job to Vertex AI
6. **180s+**: Vertex AI starts training container

**File**: `training/cli/launch/core.py:3317-3425`

---

## Queue & Timeout Management

### How Single-Job Execution Works

**1. W&B Launch Agent Args**:
```bash
wandb launch-agent -q vertex-ai-queue -e newsofpeace2 --max-jobs 1
```

| Arg | Value | Effect |
|-----|-------|--------|
| `-q` | `vertex-ai-queue` | **Queue to poll** |
| `-e` | `newsofpeace2` | **W&B entity** |
| `--max-jobs` | `1` | **🎯 Process ONLY 1 job, then exit!** |

**2. Queue Behavior**:
- Agent polls queue in FIFO order (first job submitted = first processed)
- Pulls ONE job from queue
- Downloads training code
- Submits to Vertex AI
- **After submitting: Agent exits immediately!**
- Cloud Run execution terminates (~5min total runtime)

**3. Why `--max-jobs 1`?**
- **Cost**: Each Cloud Run execution costs money (billed per second)
- **Simplicity**: One-shot execution, no persistent agent
- **Reliability**: Fresh container for each job (no state pollution)
- **Debugging**: Easy to trace specific execution → specific job

**Alternative (Not Used)**:
```bash
# Long-running agent (NOT what we do!)
wandb launch-agent -q vertex-ai-queue --max-jobs -1  # Infinite jobs
```
This would keep agent running for full 60min timeout, processing multiple jobs. We don't do this because:
- ❌ Higher Cloud Run costs (60min vs 5min billing)
- ❌ Harder to debug (which execution handled which job?)
- ❌ State pollution (one failed job could affect next job)

---

### Timeout Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ TIMEOUT LAYER 1: Cloud Run Task Timeout                     │
│ Value: 60 minutes (3600 seconds)                            │
│ Effect: Kills entire container after 60min                  │
│ Why: Prevent infinite hangs from agent errors               │
└─────────────────────────────────────────────────────────────┘
    ↓ (If agent hangs for 60min, Cloud Run kills it)
┌─────────────────────────────────────────────────────────────┐
│ TIMEOUT LAYER 2: Fatal Error Detection (5s checks)          │
│ Value: 5 seconds per check                                  │
│ Effect: Exits agent in ~5s on quota/permission errors       │
│ Why: Fail fast instead of retrying for 60min                │
└─────────────────────────────────────────────────────────────┘
    ↓ (If quota exceeded, wrapper kills agent in 5s)
┌─────────────────────────────────────────────────────────────┐
│ TIMEOUT LAYER 3: Vertex AI Job Timeout                      │
│ Value: 24 hours (86400 seconds)                             │
│ Effect: Kills training job after 24h                        │
│ Why: Prevent runaway training costs                         │
└─────────────────────────────────────────────────────────────┘
    ↓ (Training runs for max 24h)
┌─────────────────────────────────────────────────────────────┐
│ TIMEOUT LAYER 4: W&B Run Timeout (None - user cancels)      │
│ Value: None (manual cancellation via TUI/CLI)               │
│ Effect: User can cancel anytime via monitor screen          │
│ Why: User control over long-running jobs                    │
└─────────────────────────────────────────────────────────────┘
```

**Timeline Example** (Normal execution):
```
T+0s:     Cloud Run execution starts
T+30s:    Container pulls image
T+60s:    entrypoint-wrapper.sh starts
T+65s:    W&B agent starts (background)
T+70s:    Agent polls queue (finds job)
T+75s:    Agent downloads training code
T+80s:    Agent builds Vertex AI job spec
T+120s:   Agent submits to Vertex AI
T+125s:   Agent exits (max-jobs 1 reached)
T+130s:   Wrapper detects agent exit (graceful)
T+135s:   Cloud Run execution terminates
Total:    ~2 minutes
```

**Timeline Example** (Fatal error - quota exceeded):
```
T+0s:     Cloud Run execution starts
T+30s:    Container pulls image
T+60s:    entrypoint-wrapper.sh starts
T+65s:    W&B agent starts (background)
T+70s:    Agent polls queue (finds job)
T+75s:    Agent tries to submit to Vertex AI
T+76s:    GCP returns "QuotaExceeded" error
T+77s:    Error appears in agent logs
T+82s:    Wrapper detects "QuotaExceeded" (5s check)
T+83s:    Wrapper shows full error context (last 100 lines)
T+84s:    Wrapper kills agent (PID kill)
T+85s:    Wrapper exits with code 1
T+86s:    Cloud Run execution terminates (failed)
Total:    ~86 seconds (vs 60 minutes without wrapper!)
```

---

### Queue Polling & Job Pickup

**W&B Launch Queue Polling**:
```
┌─────────────────────────────────────────────────────────────┐
│ W&B Launch Queue: vertex-ai-queue                           │
│                                                              │
│ Job 1: [PENDING] → Agent picks up → [LEASED]                │
│ Job 2: [PENDING] → Waits for next agent execution           │
│ Job 3: [PENDING] → Waits for next agent execution           │
└─────────────────────────────────────────────────────────────┘
```

**States**:
- `pending`: Job waiting in queue (not yet picked up)
- `leased`: Job picked up by agent (being processed)
- `completed`: Job submitted to Vertex AI successfully
- `failed`: Job failed to submit (quota/permission error)

**Agent Behavior**:
1. Agent starts → Polls queue
2. Finds `pending` job → Marks as `leased`
3. Downloads training code
4. Submits to Vertex AI → Marks as `completed`
5. Exits (max-jobs 1)

**If Multiple Jobs Queued**:
- Job 1: Picked up by first execution → submitted
- Job 2: Waits ~5min for next execution (automatic retry by W&B)
- Job 3: Waits ~10min for next execution
- **NOT concurrent!** One job at a time!

**Why Not Concurrent?**
- Each Cloud Run execution processes ONE job
- Next execution starts when queue has pending jobs
- W&B Launch Queue handles retry logic automatically
- Simple, predictable, easy to debug

---

## Complete Launch Flow

### Step-by-Step Execution

**Phase 1: Local CLI (User's Machine)**
```
1. User runs: python training/cli.py launch
2. CLI reads: .training file (config)
3. CLI builds: 4 Docker images (arr-pytorch-base, arr-ml-stack, arr-trainer, arr-vertex-launcher)
   - Uses Cloud Build in MECHA-selected region (e.g., us-west2)
   - Pushes to Artifact Registry in us-central1
   - Hash-based caching (only rebuilds if Dockerfile/files changed)
4. CLI verifies: All 4 images exist in registry
5. CLI submits: Job to W&B Launch queue
   - Queue: vertex-ai-queue
   - State: PENDING
   - Payload: Training config + pre-built image URI
6. CLI creates: Secret Manager secrets (W&B API key, HF token)
7. CLI creates: Service account with Vertex AI IAM roles
8. CLI creates: Cloud Run job (vertex-ai-launcher)
   - Image: arr-vertex-launcher:HASH
   - Args: --max-jobs 1
   - Timeout: 60m
9. CLI executes: Cloud Run job
   - Command: gcloud run jobs execute vertex-ai-launcher
   - Returns: Execution name (vertex-ai-launcher-XXXXXXX)
10. CLI streams: Execution logs (live)
```

**Phase 2: Cloud Run Execution (GCP)**
```
11. Cloud Run: Allocates container resources
12. Cloud Run: Pulls arr-vertex-launcher:HASH image
13. Cloud Run: Starts container
14. Container: Runs entrypoint-wrapper.sh
15. Wrapper: Starts W&B agent (background process)
    - Command: wandb launch-agent -q vertex-ai-queue -e newsofpeace2 --max-jobs 1
    - PID: Stored for monitoring
16. Wrapper: Monitors agent logs (every 5 seconds)
    - Checks: 11 fatal error patterns
    - Action: Kills agent + exits on fatal error
```

**Phase 3: W&B Launch Agent (Inside Container)**
```
17. Agent: Connects to W&B API
18. Agent: Authenticates with API key (from Secret Manager)
19. Agent: Polls queue: vertex-ai-queue
20. Agent: Finds PENDING job (FIFO order)
21. Agent: Marks job as LEASED
22. Agent: Downloads training code from queue
23. Agent: Reads resource_args from job config:
    - container_spec.image_uri (pre-built arr-trainer:HASH)
    - machine_type (n1-standard-4)
    - accelerator_type (NVIDIA_L4)
    - accelerator_count (1)
24. Agent: Builds Vertex AI CustomJob spec:
    {
      "displayName": "arr-coc-training-TIMESTAMP",
      "jobSpec": {
        "workerPoolSpecs": [{
          "machineSpec": {
            "machineType": "n1-standard-4",
            "acceleratorType": "NVIDIA_L4",
            "acceleratorCount": 1
          },
          "replicaCount": 1,
          "containerSpec": {
            "imageUri": "us-central1-docker.pkg.dev/.../arr-trainer:HASH",
            "env": [
              {"name": "BASE_MODEL", "value": "Qwen/Qwen2-VL-2B-Instruct"},
              {"name": "NUM_VISUAL_TOKENS", "value": "256"},
              {"name": "LEARNING_RATE", "value": "1e-5"},
              ...
            ]
          }
        }]
      }
    }
25. Agent: Submits to Vertex AI using gcloud/Python SDK
    - Command: gcloud ai custom-jobs create ...
    - Region: us-central1
26. Agent: Marks job as COMPLETED in W&B queue
27. Agent: Checks max-jobs counter (1 job processed)
28. Agent: Exits gracefully (max-jobs reached)
```

**Phase 4: Wrapper Cleanup (Inside Container)**
```
29. Wrapper: Detects agent exit (kill -0 returns non-zero)
30. Wrapper: Checks agent exit code
    - Exit 0: Success (job submitted)
    - Exit 1: Failure (error during submission)
31. Wrapper: Exits with agent's exit code
32. Cloud Run: Detects container exit
33. Cloud Run: Terminates execution
34. Cloud Run: Reports execution state (SUCCEEDED or FAILED)
```

**Phase 5: Vertex AI Training (GCP)**
```
35. Vertex AI: Receives CustomJob submission
36. Vertex AI: Validates job spec
37. Vertex AI: Allocates GPU resources (1x L4)
38. Vertex AI: Pulls arr-trainer:HASH image
39. Vertex AI: Starts training container
40. Container: Runs training code
    - Initializes W&B run
    - Loads model (Qwen2-VL-2B)
    - Loads dataset
    - Trains for N epochs
    - Logs metrics to W&B
    - Saves checkpoints to GCS
41. Container: Exits (training complete)
42. Vertex AI: Marks job as SUCCEEDED
43. W&B: Marks run as FINISHED
```

**Total Time Breakdown**:
- CLI build images: 5-10 minutes (first time) OR 30 seconds (cached)
- CLI submit + setup: 30-60 seconds
- Cloud Run execution: 2-5 minutes
- Vertex AI job start: 3-5 minutes
- Training: 30 minutes - 24 hours (varies by config)
- **Total**: ~40 minutes minimum (for small training jobs)

---

## Fatal Error Detection

### Why It Exists

**Problem**: W&B Launch agent retries errors indefinitely
- Quota exceeded? → Retry for 60 minutes
- Permission denied? → Retry for 60 minutes
- Machine type not supported? → Retry for 60 minutes

**Result**: Wasted time + Cloud Run costs

**Solution**: Monitor agent logs, exit immediately on fatal errors

---

### Detected Error Patterns

**11 Fatal Error Patterns** (checked every 5 seconds):

| # | Pattern | Regex | Why Fatal? | Example |
|---|---------|-------|------------|---------|
| 1 | Machine type not supported | `Machine type.*is not supported` | GPU incompatible with machine type | `Machine type n1-standard-4 is not supported with NVIDIA_L4` |
| 2 | Invalid argument | `InvalidArgument: 400` | Bad job spec (won't fix with retry) | `InvalidArgument: 400 Invalid machine_type` |
| 3 | Permission denied | `PermissionDenied: 403` | Missing IAM role (needs manual fix) | `PermissionDenied: 403 Service account lacks aiplatform.jobs.create` |
| 4 | Resource not found | `NotFound: 404` | Bucket/registry doesn't exist | `NotFound: 404 Artifact Registry repository not found` |
| 5 | Quota exceeded | `QuotaExceeded\|ResourceExhausted` | Out of GPU quota (needs increase request) | `ResourceExhausted: Insufficient quota for NVIDIA_L4` |
| 6 | Repeated failures | `FAILED\|FAILURE\|FATAL` (3+ times) | Generic failures (persistent issue) | Agent crashes 3+ times in 50 lines |
| 7 | Unhandled exceptions | `Traceback (most recent call last)` + `Error:\|Exception:` | Python crash in agent code | `TypeError: 'NoneType' object is not subscriptable` |
| 8 | HTTP errors | `HttpError: <HttpError [45][0-9]{2}` | GCP API errors (4xx/5xx) | `HttpError: <HttpError 503 Service Unavailable>` |
| 9 | W&B initialization | `Failed to initialize\|wandb.*ERROR.*Failed` | Can't connect to W&B API | `Failed to initialize wandb: Connection timeout` |
| 10 | Image pull failures | `ImagePullBackOff\|ErrImagePull\|Failed to pull image` | Can't pull Docker image from registry | `ErrImagePull: denied: Permission denied for arr-trainer` |
| 11 | ~~500/503 errors~~ | ~~(Removed)~~ | ~~False positives~~ | ~~Matched config values like `MAX_TRAIN_SAMPLES: 500`~~ |

**Why Pattern 11 Removed?**
- **False positives**: Matched config values (`MAX_TRAIN_SAMPLES': '500'`)
- **Already covered**: HttpError check (pattern 8) catches structured 5xx errors
- **Repeated failures check**: Pattern 6 catches persistent unstructured errors

---

### Error Context Display (v1.1)

**Enhanced Logging** (added 2025-11-16):

When fatal error detected:
```bash
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 BAILOUT TRIGGER: GCP quota limit reached
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 All occurrences of pattern in last 100 lines:

  ► 47:ResourceExhausted: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-central1.
  ► 89:ResourceExhausted: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-central1.

━━━ Full Log Context (last 100 lines) ━━━
[Full agent log dump - 100 lines]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 FATAL ERROR DETECTED: Quota exceeded!
❌ Killing agent (PID: 42) - quota limit reached
```

**Before v1.1**:
```bash
🚨 FATAL ERROR DETECTED: Quota exceeded!
❌ Killing agent - quota limit reached
```
(No context! User had to manually check Cloud Run logs)

**After v1.1**:
- Shows ALL matching lines with line numbers
- Shows full 100-line context
- User sees EXACT error without digging through logs
- Saves ~5 minutes of debugging time per error

---

## Single-Job Execution

### Why Only 1 Job?

**Design Decision**: `--max-jobs 1`

**Pros**:
- ✅ **Simple**: One execution = one job (easy to trace)
- ✅ **Cheap**: Short-lived execution (~2-5 min vs 60 min)
- ✅ **Reliable**: Fresh container per job (no state pollution)
- ✅ **Debuggable**: Easy to correlate execution logs → job

**Cons**:
- ❌ **Overhead**: 2-5 min startup per job (could batch multiple)
- ❌ **Queue delay**: Multiple jobs queued → sequential processing

**Alternative Considered**: Long-running agent
```bash
# NOT used - would process multiple jobs
wandb launch-agent --max-jobs -1  # Infinite jobs
```

**Why Rejected**:
- Higher Cloud Run costs (60 min billing vs 5 min)
- Harder to debug (which execution handled which job?)
- Risk of state pollution (one job's error affects next)
- Complexity: Need to handle agent crashes/restarts

**Cost Analysis**:

| Scenario | Jobs/Hour | Cloud Run Time | Cost (est.) |
|----------|-----------|----------------|-------------|
| Single-job (current) | 12 | 5 min × 12 = 60 min | **$0.02** |
| Multi-job (alternative) | 12 | 60 min × 1 = 60 min | **$0.02** |

**Same cost!** But single-job is simpler and more debuggable.

---

### Job Queuing Behavior

**Scenario**: User launches 3 jobs in quick succession

```
T+0s:   Job 1 submitted to queue → State: PENDING
T+1s:   Job 2 submitted to queue → State: PENDING
T+2s:   Job 3 submitted to queue → State: PENDING

T+10s:  Cloud Run execution 1 starts
T+120s: Execution 1 picks up Job 1 → State: LEASED
T+180s: Execution 1 submits Job 1 to Vertex AI → State: COMPLETED
T+190s: Execution 1 exits (max-jobs 1)

T+200s: W&B detects pending jobs → Triggers new agent
T+210s: Cloud Run execution 2 starts
T+330s: Execution 2 picks up Job 2 → State: LEASED
T+390s: Execution 2 submits Job 2 to Vertex AI → State: COMPLETED
T+400s: Execution 2 exits

T+410s: W&B detects pending jobs → Triggers new agent
T+420s: Cloud Run execution 3 starts
T+540s: Execution 3 picks up Job 3 → State: LEASED
T+600s: Execution 3 submits Job 3 to Vertex AI → State: COMPLETED
T+610s: Execution 3 exits

Total time: ~10 minutes for 3 jobs (sequential)
```

**Key Points**:
- Jobs processed **sequentially** (one at a time)
- ~3-5 min delay between jobs (Cloud Run startup overhead)
- W&B Launch Queue handles retry/scheduling automatically
- User doesn't need to manually trigger each execution

---

## Shutdown & Cleanup

### Normal Shutdown (Graceful)

**Sequence**:
```
1. Agent finishes processing job
2. Agent checks max-jobs counter (1 job processed)
3. Agent exits with code 0 (success)
4. Wrapper detects agent exit (kill -0 returns non-zero)
5. Wrapper checks exit code (0 = success)
6. Wrapper exits with code 0
7. Cloud Run detects container exit
8. Cloud Run terminates execution (state: SUCCEEDED)
9. Cloud Run releases resources
```

**Timing**: ~2-5 minutes total

---

### Fatal Error Shutdown (Fast Bailout)

**Sequence**:
```
1. Agent encounters fatal error (e.g., quota exceeded)
2. Error appears in /tmp/wandb-agent.log
3. Wrapper detects error in 5s check
4. Wrapper shows full error context (100 lines)
5. Wrapper kills agent (kill $AGENT_PID)
6. Wrapper exits with code 1 (failure)
7. Cloud Run detects container exit (non-zero)
8. Cloud Run terminates execution (state: FAILED)
9. Cloud Run releases resources
```

**Timing**: ~5-10 seconds from error → exit

**Comparison**:
- **Without wrapper**: 60 minutes (Cloud Run timeout)
- **With wrapper**: 5-10 seconds
- **Speedup**: 360-720× faster failure detection!

---

### Timeout Shutdown (60min Hard Limit)

**Sequence**:
```
1. Cloud Run task-timeout reaches 60 minutes
2. Cloud Run sends SIGTERM to container
3. Container has 10 seconds to gracefully exit
4. If still running: Cloud Run sends SIGKILL
5. Cloud Run terminates execution (state: TIMEOUT)
6. Cloud Run releases resources
```

**When This Happens**:
- Agent hangs (infinite loop)
- Wrapper fails to detect error (bug in pattern matching)
- Network issues (can't reach W&B/GCP APIs)

**Prevention**:
- Wrapper's 5s checks should catch most hangs
- Fatal error patterns cover common issues
- Timeout is last-resort safety net

---

### Resource Cleanup

**Cloud Run Job** (persistent):
- Created once during launch
- Reused for subsequent executions
- Only updated if config changes (image, args, timeout)
- **NOT deleted during teardown** (manual cleanup needed)

**Cloud Run Execution** (ephemeral):
- Created per job submission
- Terminates after agent exits
- Automatically deleted after 30 days (GCP retention)
- **No manual cleanup needed**

**Logs**:
- Cloud Run logs: 30 day retention (automatic)
- Agent logs (/tmp/wandb-agent.log): Deleted when container terminates
- W&B logs: Permanent (until run deleted)

**Costs**:
- Cloud Run job: Free (no running containers)
- Cloud Run execution: Billed per second (2 CPU, 2Gi RAM)
- Typical cost: ~$0.001-0.002 per execution (~5 minutes)

---

## Code References

### Key Files

**Docker Image**:
- `training/images/arr-vertex-launcher/Dockerfile` (1-148)
  - Base: wandb/launch-agent
  - gcloud CLI installation
  - W&B config (noop builder)
  - Entrypoint: wrapper script

**Entrypoint Wrapper**:
- `training/images/arr-vertex-launcher/entrypoint-wrapper.sh` (1-163)
  - Version 1.1 (Enhanced error logging)
  - 11 fatal error patterns
  - 5-second monitoring loop
  - Full error context display

**CLI Launch Logic**:
- `training/cli/launch/core.py`
  - Line 1029-1071: `_submit_to_wandb()` - Queue submission
  - Line 3104-3315: `_create_cloud_run_job()` - Job creation
  - Line 3317-3425: `_execute_runner()` - Job execution
  - Line 3427+: `_stream_execution_logs()` - Log streaming

**W&B Helper**:
- `training/cli/shared/wandb_helper.py` (1-300)
  - Line 109-250: `WandBHelper` class
  - Line 125-183: `get_active_runs()` - Monitor queue + runs
  - Line 185-250: `submit_job()` - Submit to queue

**Image Manifest**:
- `training/images/arr-vertex-launcher/.image-manifest`
  - Lists all files affecting Docker build
  - Used for hash-based caching
  - Dockerfile changes trigger rebuild

---

### Configuration Files

**W&B Launch Config** (inside container):
```yaml
# /home/launch_agent/.config/wandb/launch-config.yaml
builder:
  type: noop
```
- **noop builder**: Don't build images, use pre-built ones
- Agent uses `container_spec.image_uri` from queue config

**Training Config** (`.training` file):
```bash
# Environment variables passed to training container
BASE_MODEL=Qwen/Qwen2-VL-2B-Instruct
NUM_VISUAL_TOKENS=256
LEARNING_RATE=1e-5
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=3
SAVE_EVERY_N_STEPS=50
SEED=42
WANDB_PROJECT=arr-coc-0-1
WANDB_ENTITY=newsofpeace2
WANDB_LAUNCH_QUEUE_NAME=vertex-ai-queue
HF_HUB_REPO_ID=NorthHead/arr-coc-0-1
DATASET_NAME=NorthHead/arr-coc-texture-dataset
MAX_TRAIN_SAMPLES=500
```

---

### Useful Commands

**Monitor Cloud Run Executions**:
```bash
# List recent executions
gcloud run jobs executions list \
  --job=vertex-ai-launcher \
  --region=us-central1 \
  --limit=5

# Get execution logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher" \
  --limit=100 \
  --format="table(timestamp,textPayload)"

# Stream execution logs (live)
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher" \
  --format="value(textPayload)" \
  --freshness=5m
```

**Monitor W&B Queue**:
```bash
# Via CLI monitor
python training/cli.py monitor --vertex-runner

# Via wandb CLI
wandb queue view vertex-ai-queue

# Via W&B web UI
https://wandb.ai/newsofpeace2/arr-coc-0-1/launch
```

**Monitor Vertex AI Jobs**:
```bash
# List recent jobs
gcloud ai custom-jobs list \
  --region=us-central1 \
  --limit=5

# Get job details
gcloud ai custom-jobs describe JOB_ID \
  --region=us-central1

# Stream job logs
gcloud ai custom-jobs stream-logs JOB_ID \
  --region=us-central1
```

**Debugging**:
```bash
# Check Cloud Run job config
gcloud run jobs describe vertex-ai-launcher \
  --region=us-central1 \
  --format=yaml

# Check last execution status
gcloud run jobs executions list \
  --job=vertex-ai-launcher \
  --region=us-central1 \
  --limit=1 \
  --format="value(name,status,startTime,completionTime)"

# Check for fatal errors in logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher" \
  --limit=200 \
  --format="value(textPayload)" | \
  grep -E "FATAL ERROR|QuotaExceeded|PermissionDenied|InvalidArgument"
```

---

## Summary

**arr-vertex-launcher** is a lightweight Cloud Run job that:

1. ✅ **Polls W&B Launch queue** for training jobs
2. ✅ **Submits jobs to Vertex AI** using gcloud/Python SDK
3. ✅ **Processes ONE job per execution** (--max-jobs 1)
4. ✅ **Exits in ~5 seconds on fatal errors** (wrapper monitoring)
5. ✅ **Times out after 60 minutes** (Cloud Run task-timeout)
6. ✅ **Costs ~$0.001-0.002 per execution** (~5 min runtime)

**Key Insight**: The system is designed for **simplicity and debuggability** over **performance**. Processing one job at a time adds ~3-5 min overhead per job, but makes debugging trivial and prevents state pollution between jobs.

**Next Steps**:
- Monitor executions via TUI: `python training/tui.py`
- View queue status: `python training/cli.py monitor --vertex-runner`
- Check Vertex AI jobs: `python training/cli.py monitor --vertex`
- Debug failures: Check Cloud Run logs for fatal error context

---

**Documentation Version**: 1.0
**Last Updated**: 2025-11-16
**Maintainer**: Claude + djwar42@gmail.com
