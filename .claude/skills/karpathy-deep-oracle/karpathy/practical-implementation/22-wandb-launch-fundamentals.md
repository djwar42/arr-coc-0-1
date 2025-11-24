# W&B Launch Fundamentals

**Automated ML Job Execution at Scale**

W&B Launch is an automation layer for executing machine learning workloads on shared compute resources. It enables teams to scale training runs from local development to production environments (Docker, Kubernetes, SageMaker, Vertex AI) with minimal configuration changes.

---

## Overview: What is W&B Launch?

**Core Purpose**: Decouple ML code from compute infrastructure, enabling asynchronous execution of jobs on shared resources.

**Key Benefit**: Execute the same training script locally, on a team GPU server, or in the cloud without code changes - just submit to different queues.

From [W&B Launch Documentation](https://docs.wandb.ai/platform/launch/) (accessed 2025-01-31):
- Launch automates the execution of containerized ML workflows
- Provides queue-based job scheduling for shared compute resources
- Integrates with existing W&B tracking and artifact management
- Supports multiple cloud providers and orchestration platforms

### Launch vs Manual Training Runs

**Manual W&B Runs** (Traditional):
```python
# Run on your local machine
import wandb
wandb.init(project="my-project")
# Training code runs here synchronously
```

**W&B Launch** (Automated):
```bash
# Submit job to queue - runs asynchronously on configured compute
wandb launch --queue gpu-queue --project my-project
```

**Comparison**:

| Aspect | Manual Runs | W&B Launch |
|--------|-------------|------------|
| Execution | Synchronous (blocks terminal) | Asynchronous (queued) |
| Compute | Local machine only | Any configured resource |
| Sharing | Manual setup required | Queue-based access for team |
| Scaling | Run one at a time | Parallel execution via agent pools |
| Reproducibility | Depends on local env | Containerized (guaranteed) |
| Resource Management | Manual coordination | Queue priorities, resource limits |

---

## Section 1: Launch Architecture

### Three Core Components

W&B Launch consists of three fundamental building blocks that work together to automate ML job execution:

**1. Launch Jobs** - Blueprints for ML tasks
**2. Launch Queues** - FIFO job lists for specific compute targets
**3. Launch Agents** - Workers that execute queued jobs

From [W&B Launch Walkthrough](https://docs.wandb.ai/platform/launch/walkthrough) (accessed 2025-01-31):

> "A launch job is a blueprint for configuring and running tasks in your ML workflow. Once you have a launch job, you can add it to a launch queue. A launch queue is a first-in, first-out (FIFO) queue where you can configure and submit your jobs to a particular compute target resource, such as Amazon SageMaker or a Kubernetes cluster. As jobs are added to the queue, launch agents poll that queue and execute the job on the system targeted by the queue."

### Architecture Diagram Explanation

```
┌─────────────────────────────────────────────────────────────
│ W&B Cloud / Self-Hosted Server
│
│  ┌──────────────────────────────────────────
│  │ Launch Jobs (Job Registry)
│  │  ├─ job-training-llm-v1
│  │  ├─ job-eval-vqa-v2
│  │  └─ job-deploy-triton-v3
│  └──────────────────────────────────────────
│
│  ┌──────────────────────────────────────────
│  │ Launch Queues (Job Scheduling)
│  │  ├─ queue-local-gpus    (Docker)
│  │  ├─ queue-k8s-cluster   (Kubernetes)
│  │  └─ queue-sagemaker     (AWS SageMaker)
│  └──────────────────────────────────────────
│
└─────────────────────────────────────────────────────────────

         ↓ Jobs submitted to queues
         ↓ Agents poll queues

┌─────────────────────────────────────────────────────────────
│ Your Infrastructure (Agent Environment)
│
│  ┌────────────────────────────────────
│  │ Launch Agent 1
│  │  - Polls: queue-local-gpus
│  │  - Executes: Docker containers
│  │  - Resources: 4x A100 GPUs
│  └────────────────────────────────────
│
│  ┌────────────────────────────────────
│  │ Launch Agent 2
│  │  - Polls: queue-k8s-cluster
│  │  - Executes: Kubernetes pods
│  │  - Resources: Auto-scaling GPU nodes
│  └────────────────────────────────────
│
└─────────────────────────────────────────────────────────────
```

**Workflow**:
1. Create a launch job (from code, Docker image, or git repo)
2. Submit job to queue via UI, CLI, or Python API
3. Agent polls queue, detects new job
4. Agent builds/pulls container image
5. Agent executes job on target resource
6. Job logs metrics to W&B (creates a run)
7. Agent marks job complete, polls for next job

### When to Use Launch

**Use Launch when:**
- Multiple team members need access to shared GPU servers
- Training jobs need to run overnight or on weekends
- You want to automate hyperparameter sweeps across many runs
- Production workflows require reproducible, containerized execution
- Different projects need different compute resources (CPU, GPU, TPU)
- Cost optimization matters (spot instances, auto-scaling)

**Don't need Launch if:**
- Working solo with dedicated local compute
- Running quick experiments interactively
- No need for containerization or reproducibility guarantees
- Manual job submission is acceptable

### Launch in the ML Workflow

```
Development                    Staging                     Production
────────────────────────────────────────────────────────────────────

Local iteration      →    Queue-based training    →    Automated deployment
(wandb.init)              (wandb launch)               (Launch job trigger)

Manual experiments   →    Hyperparameter sweeps   →    Scheduled retraining
Small datasets            Full datasets                Live data pipelines
CPU/single GPU           Multi-GPU/distributed         Optimized inference
```

Launch bridges the gap between local experimentation and production-scale training by providing:
- Consistent execution environment (containers)
- Resource isolation and allocation
- Queue-based scheduling and prioritization
- Integration with existing W&B tracking

From [W&B Launch Documentation](https://docs.wandb.ai/platform/launch/) (accessed 2025-01-31):
- Launch is designed for teams building workflows around shared compute
- Enables asynchronous execution of jobs with advanced features like prioritization
- Supports hyperparameter optimization at scale via Sweeps integration

---

## Section 2: Jobs and Queues

### Launch Jobs: Task Blueprints

A **launch job** is a W&B Artifact that contains everything needed to execute an ML task:
- Python code and file assets
- Runnable entrypoint (e.g., `train.py`, `app.py`)
- Environment specification (`requirements.txt`, `Dockerfile`)
- Input parameters (config) and expected outputs (metrics)

From [W&B Launch Terminology](https://docs.wandb.ai/platform/launch/launch-terminology) (accessed 2025-01-31):

> "A launch job is a specific type of W&B Artifact that represents a task to complete. Job definitions include: Python code and other file assets, including at least one runnable entrypoint; Information about the input (config parameter) and output (metrics logged); Information about the environment."

### Three Types of Launch Jobs

**1. Artifact-Based (Code-Based) Jobs**
- Code saved as W&B artifact
- Launch agent builds container from code + dependencies
- Best for: Active development, code versioning via W&B

```bash
# Create job from local code
wandb job create --name "train-job" \
  --project my-project \
  --entry-point "train.py" \
  --requirements requirements.txt
```

**2. Git-Based Jobs**
- Code cloned from specific commit/branch/tag
- Launch agent builds container from git repo
- Best for: CI/CD integration, team collaboration

```bash
# Create job from git repository
wandb launch \
  --uri https://github.com/user/repo.git \
  --entry-point train.py \
  --queue gpu-queue
```

**3. Image-Based Jobs**
- Code baked into pre-built Docker image
- Launch agent pulls image and runs
- Best for: Production deployment, complex dependencies

```bash
# Create job from Docker image
wandb launch \
  --docker-image myrepo/training:v1.0 \
  --queue gpu-queue
```

### Creating Jobs from Existing Runs

Every W&B run can be converted to a launch job:

```python
# After training completes
import wandb
run = wandb.init(project="my-project", job_type="train")
# ... training code ...
run.finish()

# Job automatically created from run
# Find it in W&B UI: Project → Jobs tab
```

Or via CLI:
```bash
# Create job from specific run
wandb job create --from-run <run-id>
```

### Job Configuration

Jobs accept configuration parameters that modify behavior without changing code:

```yaml
# Job configuration override
hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

resource:
  gpu_count: 4
  gpu_type: "a100"
  memory: "64GB"
```

From [W&B Launch Job Inputs](https://docs.wandb.ai/platform/launch/job-inputs) (accessed 2025-01-31):
- Jobs accept input parameters via config overrides
- Parameters can be modified when submitting to queue
- Enables same job to run with different hyperparameters

### Launch Queues: FIFO Job Scheduling

A **launch queue** is an ordered list (FIFO) of jobs targeting a specific compute resource.

From [W&B Launch Terminology](https://docs.wandb.ai/platform/launch/launch-terminology) (accessed 2025-01-31):

> "Launch queues are ordered lists of jobs to execute on a specific target resource. Launch queues are first-in, first-out (FIFO). There is no practical limit to the number of queues you can have, but a good guideline is one queue per target resource."

### Queue Creation

**Via W&B UI**:
1. Navigate to [wandb.ai/launch](https://wandb.ai/launch)
2. Click "Create a queue"
3. Select entity (user or team)
4. Enter queue name
5. Select resource type (Docker, Kubernetes, SageMaker, Vertex AI)
6. Configure resource parameters
7. Click "Create queue"

**Via CLI**:
```bash
wandb launch-queue create \
  --entity my-team \
  --queue-name gpu-training \
  --resource docker \
  --config config.yaml
```

### Queue Configuration

Each queue has resource-specific configuration:

**Docker Queue Example**:
```yaml
# docker-queue-config.yaml
resource: docker
resource_args:
  gpus: "all"  # Use all available GPUs
  shm_size: "16gb"  # Shared memory for data loading
  network: "host"  # Network mode
```

**Kubernetes Queue Example**:
```yaml
# k8s-queue-config.yaml
resource: kubernetes
resource_args:
  namespace: "ml-training"
  gpu_type: "nvidia.com/gpu"
  node_selector:
    node-type: "gpu-node"
```

### Queue Priorities and Routing

**Priority Levels** (higher number = higher priority):
```python
# Submit job with priority
wandb.launch_add(
    job="job-artifact:v1",
    queue="gpu-queue",
    priority=10  # Higher priority jobs run first
)
```

**Queue Routing Patterns**:
- **Development queue**: Low-priority, spot instances, smaller GPUs
- **Production queue**: High-priority, on-demand instances, larger GPUs
- **Sweep queue**: Medium-priority, auto-scaling, many parallel agents

### Submitting Jobs to Queues

**Via UI**:
1. Navigate to project → Jobs tab
2. Click on job
3. Click "Launch" button
4. Select queue
5. Override config if needed
6. Click "Launch"

**Via CLI**:
```bash
# Submit job to queue
wandb launch \
  --job job-artifact:v1 \
  --queue gpu-queue \
  --project my-project \
  --config config.yaml
```

**Via Python API**:
```python
import wandb

# Submit job programmatically
wandb.launch_add(
    job="my-team/my-project/job-artifact:v1",
    queue="gpu-queue",
    config={"learning_rate": 0.001, "batch_size": 64},
    priority=5
)
```

### Job Monitoring and Logs

**Job Lifecycle States**:
1. **Queued** - Job submitted to queue, waiting for agent
2. **Running** - Agent executing job on target resource
3. **Finished** - Job completed successfully
4. **Failed** - Job encountered error
5. **Stopped** - Job manually stopped by user

**Monitoring**:
- View queue status at [wandb.ai/launch](https://wandb.ai/launch)
- See job position in queue, estimated start time
- Monitor resource utilization per job
- View real-time logs from running jobs

**Log Access**:
```bash
# View logs for running job (via agent terminal)
# Agent outputs job logs in real-time

# Or access via W&B UI:
# Project → Runs → Select run → Logs tab
```

From [W&B Launch Queue Observability](https://docs.wandb.ai/platform/launch/launch-queue-observability) (accessed 2025-01-31):
- Queue monitoring shows job status, position, and resource usage
- Real-time logs available for debugging failed jobs
- Queue metrics help optimize agent allocation

---

## Section 3: Launch Agents

A **launch agent** is a lightweight, persistent program that:
1. Polls one or more launch queues
2. Pulls jobs from the queue (FIFO order)
3. Builds or pulls container images
4. Executes jobs on target resources
5. Reports progress back to W&B
6. Repeats indefinitely

From [W&B Launch Terminology](https://docs.wandb.ai/platform/launch/launch-terminology) (accessed 2025-01-31):

> "Launch agents are lightweight, persistent programs that periodically check Launch queues for jobs to execute. When a launch agent receives a job, it first builds or pulls the image from the job definition then runs it on the target resource."

### Agent Types by Target Resource

**1. Docker Agent** (Local/VM execution)
- Runs Docker containers on local machine or VM
- Requires Docker engine installed
- Best for: Team GPU servers, development environments

**2. Kubernetes Agent**
- Submits jobs as Kubernetes pods
- Requires kubectl access to cluster
- Best for: Production clusters, auto-scaling

**3. SageMaker Agent**
- Submits jobs as SageMaker Training Jobs
- Requires AWS credentials
- Best for: AWS-native workflows, managed compute

**4. Vertex AI Agent**
- Submits jobs as Vertex AI Custom Jobs
- Requires GCP credentials
- Best for: GCP-native workflows, TPU access

From [W&B Launch Target Resources](https://docs.wandb.ai/platform/launch/launch-terminology#target-resources) (accessed 2025-01-31):
- Each target resource accepts different configuration parameters
- Resource configurations have queue defaults, can be overridden per job
- Single agent can support multiple target resources if configured properly

### Setting Up a Local Docker Agent

**Prerequisites**:
- Docker installed and running
- W&B account and API key
- Access to launch queue

**Basic Setup**:

```bash
# 1. Authenticate with W&B
wandb login

# 2. Start agent polling a queue
wandb launch-agent \
  --queue gpu-queue \
  --entity my-team
```

The agent will:
- Poll `gpu-queue` every few seconds
- Execute jobs using local Docker
- Log output to terminal
- Run indefinitely (Ctrl+C to stop)

**Multi-Queue Agent**:
```bash
# Poll multiple queues (useful for load balancing)
wandb launch-agent \
  --queue high-priority \
  --queue low-priority \
  --entity my-team
```

Agent prioritizes jobs from first queue, then second queue.

### Resource Configuration for Agents

**GPU Access**:
```bash
# Agent config for GPU jobs
wandb launch-agent \
  --queue gpu-queue \
  --entity my-team \
  --config agent-config.yaml
```

`agent-config.yaml`:
```yaml
resource_args:
  docker:
    gpus: "all"  # Use all GPUs
    shm_size: "32gb"  # Shared memory for data loaders
    environment:
      - "CUDA_VISIBLE_DEVICES=0,1,2,3"
```

**CPU and Memory Limits**:
```yaml
resource_args:
  docker:
    cpus: "16"  # CPU core limit
    memory: "64g"  # RAM limit
    memswap: "64g"  # Swap limit
```

**Environment Variables**:
```yaml
resource_args:
  docker:
    environment:
      - "HF_HOME=/cache/huggingface"  # HuggingFace cache
      - "WANDB_CACHE_DIR=/cache/wandb"  # W&B artifact cache
      - "TORCH_HOME=/cache/torch"  # PyTorch model cache
```

### Agent Pools for Load Balancing

**Pattern**: Multiple agents polling same queue = parallel job execution

**Setup**:
```bash
# Terminal 1: Agent 1
wandb launch-agent --queue shared-queue --entity team

# Terminal 2: Agent 2
wandb launch-agent --queue shared-queue --entity team

# Terminal 3: Agent 3
wandb launch-agent --queue shared-queue --entity team
```

**Result**:
- 3 jobs from queue run in parallel
- Queue processes 3x faster
- Each agent handles 1 job at a time

**Use Cases**:
- Hyperparameter sweeps (many parallel runs)
- Team training queue (multiple experiments simultaneously)
- Production pipeline (parallel evaluation jobs)

### Agent Configuration Files

**Advanced agent configuration**:

```yaml
# launch-agent-config.yaml
max_jobs: -1  # Run indefinitely (-1) or stop after N jobs
entity: my-team

queues:
  - name: high-priority
    max_jobs: 10  # Limit jobs from this queue
  - name: low-priority
    max_jobs: -1

builder:
  type: docker  # or noop (for pre-built images)
  accelerator:
    base_image: "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"

registry:
  url: "https://index.docker.io/v1/"
  username: "myuser"
  password_secret: "DOCKER_PASSWORD"  # From env var

environment:
  HF_HOME: "/mnt/cache/huggingface"
  WANDB_CACHE_DIR: "/mnt/cache/wandb"
```

Launch agent with config:
```bash
wandb launch-agent --config launch-agent-config.yaml
```

### Agent Monitoring and Health Checks

**Agent Status**:
- View active agents in W&B UI: [wandb.ai/launch](https://wandb.ai/launch) → Queue → Agents tab
- Shows: Agent ID, last poll time, current job, jobs completed

**Health Checks**:
```bash
# Agent logs show health status
wandb launch-agent --queue gpu-queue --entity team

# Output:
# [INFO] Polling queue 'gpu-queue'...
# [INFO] No jobs in queue, waiting...
# [INFO] Job received: job-artifact:v1
# [INFO] Building Docker image...
# [INFO] Running container...
# [INFO] Job completed successfully
# [INFO] Polling queue 'gpu-queue'...
```

**Failure Handling**:
- Job fails → agent logs error, marks job failed, continues polling
- Agent crashes → queue remains intact, restart agent to resume
- Docker failure → agent logs error, skips job, continues polling

**Production Patterns**:
- Run agents as systemd services (Linux) or launchd (macOS)
- Use process managers (supervisord, pm2) for auto-restart
- Monitor agent logs for errors
- Alert if agent stops polling (no heartbeat)

### Agent vs Agent Environment

From [W&B Launch Terminology](https://docs.wandb.ai/platform/launch/launch-terminology#launch-agent-environment) (accessed 2025-01-31):

> "The agent environment is the environment where a launch agent is running, polling for jobs. The agent's runtime environment is independent of a queue's target resource. In other words, agents can be deployed anywhere as long as they are configured sufficiently to access the required target resources."

**Examples**:
- Docker agent running on local laptop → executes jobs in Docker containers on same laptop
- Kubernetes agent running in pod → submits jobs as separate pods in cluster
- SageMaker agent running on EC2 → submits jobs as SageMaker Training Jobs

**Key Insight**: Agent location ≠ job execution location. Agents are coordinators, not executors.

---

## Complete Example: End-to-End Launch Workflow

### Scenario: Team Training Queue

**Goal**: 3 researchers share 4-GPU server for LLM fine-tuning experiments.

**Setup**:

**Step 1: Create Queue (Admin)**
```bash
wandb launch-queue create \
  --entity ml-team \
  --queue-name team-gpus \
  --resource docker
```

**Step 2: Start Agent on GPU Server (Admin)**
```bash
# SSH into GPU server
ssh gpu-server.company.com

# Start agent
wandb launch-agent \
  --queue team-gpus \
  --entity ml-team \
  --config gpu-config.yaml
```

`gpu-config.yaml`:
```yaml
resource_args:
  docker:
    gpus: "all"
    shm_size: "32gb"
```

**Step 3: Researcher 1 Submits Job**
```bash
# From local laptop
wandb launch \
  --uri https://github.com/ml-team/llm-training.git \
  --entry-point train.py \
  --queue team-gpus \
  --config '{"learning_rate": 0.0001, "batch_size": 8}'
```

**Step 4: Researcher 2 Submits Job**
```bash
wandb launch \
  --uri https://github.com/ml-team/llm-training.git \
  --entry-point train.py \
  --queue team-gpus \
  --config '{"learning_rate": 0.0003, "batch_size": 16}'
```

**Step 5: Jobs Execute in FIFO Order**
- Researcher 1's job runs first (uses 4 GPUs)
- Researcher 2's job queued
- When job 1 finishes, job 2 starts automatically
- Both researchers monitor progress in W&B UI

**Benefits**:
- No SSH coordination needed
- Fair FIFO scheduling
- Reproducible containerized execution
- Automatic logging to W&B
- Queue visible to entire team

---

## Best Practices

### Job Design
1. **Always call `wandb.init()`** - Required for Launch jobs to complete successfully
2. **Use config parameters** - Make hyperparameters configurable without code changes
3. **Log artifacts** - Save checkpoints, models as W&B artifacts for resumption
4. **Handle preemption** - Save checkpoints frequently for spot instance jobs

### Queue Management
1. **One queue per resource type** - Separate queues for different GPUs, cloud providers
2. **Use priority levels** - High-priority for urgent jobs, low for experiments
3. **Monitor queue depth** - Add more agents if queue grows too long
4. **Name queues clearly** - `gpu-a100-prod` not `queue-1`

### Agent Operations
1. **Run agents as services** - Use systemd, supervisord for auto-restart
2. **Configure resource limits** - Prevent single job from using all resources
3. **Set up logging** - Capture agent logs for debugging
4. **Use agent pools** - Multiple agents for parallel execution
5. **Monitor agent health** - Alert if agent stops polling

### Security
1. **Protect API keys** - Use environment variables, not hardcoded
2. **Use private queues** - Restrict access to team/organization
3. **Review job code** - Jobs run arbitrary code, verify before queuing
4. **Isolate agents** - Run agents in separate namespaces/VMs for multi-tenancy

---

## Common Patterns

### Development → Production Queue Promotion

```python
# Development: Quick iteration on small queue
wandb.launch_add(
    job="dev-job:v1",
    queue="dev-cpu-queue",  # Fast, cheap CPU instances
    config={"epochs": 1, "dataset_size": 1000}  # Small test
)

# After validation: Production training
wandb.launch_add(
    job="dev-job:v1",  # Same job
    queue="prod-gpu-queue",  # Large GPU instances
    config={"epochs": 100, "dataset_size": 1000000},  # Full dataset
    priority=10  # High priority
)
```

### Scheduled Retraining

```bash
# Cron job to retrain model daily
0 2 * * * wandb launch --job retrain-job:latest --queue prod-queue
```

### Multi-Stage Training Pipeline

```python
# Stage 1: Preprocess data
preprocess_job = wandb.launch_add(
    job="preprocess:v1",
    queue="cpu-queue"
)

# Wait for completion, get output artifact
# (Requires polling or webhook integration)

# Stage 2: Train model
train_job = wandb.launch_add(
    job="train:v1",
    queue="gpu-queue",
    config={"data_artifact": "preprocessed-data:v1"}
)

# Stage 3: Evaluate model
eval_job = wandb.launch_add(
    job="eval:v1",
    queue="cpu-queue",
    config={"model_artifact": "trained-model:v1"}
)
```

---

## Troubleshooting

### Job Stuck in Queue
- **Check agent status**: Is agent running and polling?
- **Check agent logs**: Look for errors (permissions, Docker daemon)
- **Verify queue name**: Job queue matches agent queue?

### Job Failed
- **View job logs**: W&B UI → Project → Runs → Failed run → Logs
- **Check container build**: Did image build succeed?
- **Verify dependencies**: All packages in `requirements.txt`?
- **Resource limits**: Enough GPU memory, disk space?

### Agent Not Polling
- **Check authentication**: `wandb login` successful?
- **Check network**: Agent can reach api.wandb.ai?
- **Check Docker daemon**: `docker ps` works?
- **Check queue permissions**: Agent has access to queue?

---

## Summary

**W&B Launch** provides queue-based automation for ML job execution:

**Core Components**:
1. **Jobs** - Task blueprints (code + environment)
2. **Queues** - FIFO job lists for specific resources
3. **Agents** - Workers that execute queued jobs

**Key Benefits**:
- Decouple code from infrastructure
- Share compute resources across team
- Reproducible containerized execution
- Scale from local to cloud seamlessly
- Integrate with W&B tracking and artifacts

**Typical Workflow**:
1. Create job from code/image/git
2. Submit job to queue
3. Agent polls queue, executes job
4. Monitor progress in W&B UI
5. Results logged automatically

**Next Steps**:
- See `23-wandb-launch-llm-training.md` for LLM/VLM training automation patterns
- See `24-wandb-launch-job-config.md` for advanced job configuration
- See `25-wandb-launch-sweeps.md` for hyperparameter optimization at scale

---

## Sources

**W&B Launch Documentation:**
- [W&B Launch Overview](https://docs.wandb.ai/platform/launch/) (accessed 2025-01-31)
- [W&B Launch Walkthrough](https://docs.wandb.ai/platform/launch/walkthrough) (accessed 2025-01-31)
- [Launch Terms and Concepts](https://docs.wandb.ai/platform/launch/launch-terminology) (accessed 2025-01-31)

**Additional References:**
- [Set up Launch](https://docs.wandb.ai/platform/launch/set-up-launch)
- [Create Launch Jobs](https://docs.wandb.ai/platform/launch/create-launch-job)
- [Configure Launch Queue](https://docs.wandb.ai/platform/launch/setup-queue-advanced)
- [Set up Launch Agent](https://docs.wandb.ai/platform/launch/setup-agent-advanced)
- [Launch Queue Observability](https://docs.wandb.ai/platform/launch/launch-queue-observability)
- [Manage Job Inputs](https://docs.wandb.ai/platform/launch/job-inputs)
