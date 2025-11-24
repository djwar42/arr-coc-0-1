# W&B Launch + Sweeps Integration: Scalable Hyperparameter Tuning

## Overview

W&B Launch + Sweeps integration automates hyperparameter optimization at scale by combining Launch's job orchestration with Sweeps' intelligent search algorithms. A sweep scheduler runs as a Launch job, spawning training runs with different hyperparameter configurations onto Launch queues, enabling parallel execution across distributed compute resources.

**Key benefits:**
- Automated parallel sweep execution (10s-100s of concurrent runs)
- Dynamic resource allocation based on queue availability
- Centralized sweep management across teams and clusters
- Resume interrupted sweeps without losing progress
- Cost tracking and optimization across sweep runs

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Sweep scheduler pushed to Launch Queue with hyperparameters
- Scheduler launches training jobs onto same queue
- Continues until sweep completes or stopped
- Standard (bayes/grid/random) or custom schedulers

## Architecture: How Launch + Sweeps Work Together

### Standard Sweep Flow (Without Launch)

```
Manual Process:
1. Developer: wandb sweep config.yaml → Sweep ID
2. Developer: Start agent manually → wandb agent <sweep_id>
3. Agent: Polls for hyperparameter configs
4. Agent: Runs training locally with config
5. Developer: Monitor and manually stop
```

**Limitations:**
- Single machine execution (limited parallelism)
- Manual agent management
- No automatic resource allocation
- Interrupted if machine fails

### Launch + Sweeps Flow (Automated)

```
Automated Orchestration:
1. wandb launch-sweep config.yaml → Sweep Scheduler Job
2. Launch Queue: Scheduler job queued
3. Launch Agent: Picks up scheduler job
4. Scheduler: Generates hyperparameter configs
5. Scheduler: Pushes training jobs to queue (parallel)
6. Multiple Agents: Execute training jobs concurrently
7. Scheduler: Monitors results, generates new configs
8. Auto-complete when sweep criteria met
```

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Sweep scheduler is itself a Launch job
- Scheduler launches sweep runs onto same queue
- Multiple agents can execute training runs in parallel
- Scales to 100s of concurrent runs

**Benefits:**
- Automatic parallelization across available compute
- Fault-tolerant (resume on failure)
- Multi-GPU/multi-node support
- Queue-based load balancing

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────
│ W&B Cloud
│
│ Sweep Scheduler Job (Launch Job)
│ ├─ Bayesian optimization logic
│ ├─ Generates hyperparameter configs
│ └─ Pushes training jobs to queue
│
│ Launch Queue: "gpu-training"
│ ├─ Scheduler Job (priority: high)
│ ├─ Training Job 1 (lr=0.001, batch=32)
│ ├─ Training Job 2 (lr=0.01, batch=64)
│ ├─ Training Job 3 (lr=0.0001, batch=128)
│ └─ ... (parallel jobs)
└─────────────────────────────────────────────────
          ↓ Pull jobs
┌─────────────────────────────────────────────────
│ Launch Agents (Compute Resources)
│
│ Agent 1 (GPU cluster node 1)
│ └─ Executes: Training Job 1
│
│ Agent 2 (GPU cluster node 2)
│ └─ Executes: Training Job 2
│
│ Agent 3 (GPU cluster node 3)
│ └─ Executes: Training Job 3
│
│ Agent 4-N (auto-scale based on queue depth)
└─────────────────────────────────────────────────
```

## Section 1: Launch + Sweeps Integration Fundamentals

### Why Combine Launch with Sweeps?

**Sweeps alone:**
- Intelligent hyperparameter search (bayes, grid, random)
- Single-machine execution limitation
- Manual agent management
- No automatic resource allocation

**Launch alone:**
- Job orchestration and queueing
- Multi-compute execution
- No built-in hyperparameter optimization

**Launch + Sweeps together:**
- Intelligent search + distributed execution
- Automatic parallelization
- Resource pooling across sweeps
- Cost-effective scaling

From [W&B Sweeps Tutorial](https://docs.wandb.ai/models/sweeps/walkthrough) (accessed 2025-01-31):
- Sweeps define search space and optimization method
- Sweep controller manages hyperparameter selection
- Agents execute training with assigned configs

### Creating Sweeps with Launch: Two Approaches

#### 1. Standard Scheduler (Recommended for Most Use Cases)

Uses W&B's built-in Bayesian/grid/random search algorithms.

**Via W&B App UI:**
```
Steps (from W&B Launch Sweeps docs):
1. Navigate to project → Sweeps icon (broom)
2. Click "Create Sweep"
3. Click "Configure Launch" button
4. Select:
   - Job: Training job to sweep over
   - Queue: Launch queue to run on
   - Priority: Job priority (low/medium/high)
5. Configure scheduler overrides:
   - num_workers: Concurrent runs (default: 1)
6. Select destination project
7. Click "Launch Sweep"
```

**Via CLI:**
```bash
# 1. Create sweep config
cat > launch-sweep-config.yaml <<EOF
description: LLM fine-tuning hyperparameter sweep
method: bayes  # or grid, random
metric:
  name: val_loss
  goal: minimize

# Training job that executes each sweep run
job: my-project/llm-training-job:v3

# Queue to run on
queue: gpu-a100-queue

parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
    distribution: log_uniform
  batch_size:
    values: [16, 32, 64]
  warmup_steps:
    min: 100
    max: 1000
EOF

# 2. Launch sweep
wandb launch-sweep launch-sweep-config.yaml \
  --queue gpu-a100-queue \
  --project my-llm-project
```

#### 2. Custom Scheduler (Advanced)

Implement your own sweep logic (Optuna, custom Bayesian optimization, etc.).

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Custom scheduler runs as a Launch job
- Full control over hyperparameter selection
- Can integrate external optimization libraries

**Example with W&B Scheduler Job:**
```yaml
# Custom scheduler sweep config
description: Launch sweep with custom scheduler
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8 concurrent training runs

# Training job that sweep runs will execute
job: my-project/training-job:latest

method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```

Launch:
```bash
wandb launch-sweep custom-sweep-config.yaml \
  --queue my-queue \
  --project my-project
```

### Monitoring Sweep Progress

**Real-time monitoring:**
- W&B App: Sweeps dashboard shows parallel runs
- Parallel coordinates plot: Visualize hyperparameter impact
- Queue observability: See job status (queued/running/finished)

**Key metrics to track:**
- Sweep progress: X/N runs completed
- Best run so far: Current optimal hyperparameters
- Queue depth: Jobs waiting vs running
- Resource utilization: GPU/CPU usage across agents

### Best Run Selection Automation

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Sweep automatically tracks best performing config
- Metric defined in sweep config determines "best"
- Access best run: W&B App → Sweep → "Best Run" highlight

**Automated model selection:**
```python
import wandb

# Get best run from completed sweep
api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")

# Best run based on metric defined in sweep config
best_run = sweep.best_run()
print(f"Best hyperparameters: {best_run.config}")
print(f"Best validation loss: {best_run.summary['val_loss']}")

# Download best model artifact
best_model = best_run.use_artifact('model:best')
best_model.download()
```

## Section 2: Scalable Hyperparameter Tuning Patterns

### Parallel Sweep Execution (10s-100s of Runs)

**Challenge:** Single-agent sweeps are slow (sequential execution).

**Solution:** Launch + Sweeps enables massive parallelization.

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Configure `num_workers` in scheduler config
- Each worker executes training job concurrently
- Limited only by available agents/compute

**Example: 50 Parallel Runs**
```yaml
# Sweep config for massive parallelization
description: Large-scale LLM hyperparameter search
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 50  # 50 concurrent training jobs

job: my-project/llm-training:v5
queue: gpu-cluster-queue

method: random
run_cap: 500  # Total runs to execute

metric:
  name: val_perplexity
  goal: minimize

parameters:
  learning_rate:
    distribution: log_uniform
    min: 1e-6
    max: 1e-3
  weight_decay:
    distribution: uniform
    min: 0.0
    max: 0.1
  dropout:
    values: [0.0, 0.1, 0.2, 0.3]
```

**Resource requirements:**
- 50 num_workers → Need 50+ Launch agents
- Each agent: 1-8 GPUs depending on training job
- Queue depth monitoring: Ensure jobs don't starve

### Resource Pooling Strategies

#### Strategy 1: Dedicated Sweep Queue
```yaml
# Sweep-specific queue with dedicated resources
queue_config:
  name: sweep-gpu-queue
  resource_args:
    gpu_type: A100
    num_gpus: 8
  max_concurrent_jobs: 50
```

**Use when:**
- Large sweep with known resource needs
- Want guaranteed resources for sweep completion
- Cost is less important than speed

#### Strategy 2: Shared Queue with Priorities
```yaml
# Sweep uses shared queue but with priority
scheduler:
  num_workers: 20
  priority: medium  # Sweep jobs get medium priority

job_overrides:
  priority: low  # Individual training jobs can be preempted
```

**Use when:**
- Share resources with other workloads
- Want cost efficiency over max speed
- Okay with some queueing delays

### GPU Allocation Per Sweep Run

**Specify GPU requirements in training job:**
```python
# training_job.py - Training code
import wandb

# This config will be overridden by sweep
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    # GPU resources defined at job level, not sweep level
}

with wandb.init(config=config) as run:
    # Training code that uses run.config values
    model = train_model(
        lr=run.config.learning_rate,
        batch_size=run.config.batch_size
    )
```

**Job resource specification:**
```yaml
# job-config.yaml for training job
resource_args:
  num_gpus: 2  # Each sweep run uses 2 GPUs
  gpu_type: A100
  memory: 64GB
```

**Scaling considerations:**
- num_workers=50, num_gpus=2 per run → Need 100 GPUs total
- Agent auto-scaling: Provision agents based on queue depth
- Cost optimization: Use smaller GPUs for smaller models

### Queue-Based Load Balancing

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Sweep scheduler pushes jobs to queue
- Multiple agents pull from same queue
- Automatic load distribution

**Multi-queue strategy for heterogeneous resources:**
```yaml
# Sweep config with multi-queue routing
description: Multi-GPU sweep with smart routing
scheduler:
  num_workers: 30

# Different jobs for different resource tiers
job: my-project/training-job:latest

# Sweep creates jobs that can route to different queues
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
  batch_size:
    values: [16, 32, 64, 128]
```

**Agent setup:**
```bash
# Agent 1: High-end GPUs for large batches
wandb launch-agent \
  --queue gpu-a100-queue \
  --max-jobs 4

# Agent 2-5: Mid-tier GPUs for small batches
wandb launch-agent \
  --queue gpu-t4-queue \
  --max-jobs 8
```

### Adaptive Resource Allocation

**Dynamic agent scaling based on queue depth:**

From [W&B Launch Kubernetes Integration](https://docs.wandb.ai/platform/launch/setup-agent-advanced) (referenced in Launch docs):
- Kubernetes HPA (Horizontal Pod Autoscaler) scales agents
- Scales up when queue depth > threshold
- Scales down when queue empty

**Example: Kubernetes Auto-scaling**
```yaml
# k8s-agent-autoscale.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wandb-launch-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wandb-launch-agent
  minReplicas: 2
  maxReplicas: 50  # Scale up to 50 agents for large sweeps
  metrics:
  - type: External
    external:
      metric:
        name: wandb_queue_depth
        selector:
          matchLabels:
            queue_name: gpu-sweep-queue
      target:
        type: AverageValue
        averageValue: "5"  # 5 jobs per agent
```

### Cost Tracking Across Sweep

**Monitor sweep costs:**
```python
import wandb

api = wandb.Api()
sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")

total_cost = 0
for run in sweep.runs:
    # Assuming cost logged during training
    if 'gpu_cost_usd' in run.summary:
        total_cost += run.summary['gpu_cost_usd']

print(f"Total sweep cost: ${total_cost:.2f}")
print(f"Runs completed: {len(list(sweep.runs))}")
print(f"Cost per run: ${total_cost / len(list(sweep.runs)):.2f}")
```

**Cost optimization strategies:**
- Start with random search (cheaper than Bayesian early on)
- Use early termination (Hyperband) to kill bad runs fast
- Spot instances for cost savings (70-90% reduction)
- Smaller GPU types for initial exploration phase

## Section 3: Advanced Sweep Patterns with Launch

### Multi-Stage Sweeps (Coarse → Fine)

**Pattern:** Broad exploration → Narrow exploitation

**Stage 1: Coarse sweep (wide search)**
```yaml
# stage1-coarse-sweep.yaml
description: Coarse search over wide hyperparameter space
method: random
run_cap: 100
scheduler:
  num_workers: 20

job: my-project/training-job:v1
queue: gpu-t4-queue  # Cheaper GPUs for exploration

metric:
  name: val_loss
  goal: minimize

parameters:
  learning_rate:
    min: 1e-6
    max: 1e-2
    distribution: log_uniform
  batch_size:
    values: [8, 16, 32, 64, 128]
  warmup_ratio:
    min: 0.0
    max: 0.2
```

**Stage 2: Fine sweep (narrow search)**
```yaml
# stage2-fine-sweep.yaml
description: Fine-tuning around best region from stage 1
method: bayes
run_cap: 50
scheduler:
  num_workers: 10

job: my-project/training-job:v1
queue: gpu-a100-queue  # Better GPUs for fine-tuning

metric:
  name: val_loss
  goal: minimize

# Narrow ranges based on stage 1 best run
parameters:
  learning_rate:
    min: 5e-5  # Narrowed from stage 1 best
    max: 2e-4
    distribution: log_uniform
  batch_size:
    value: 32  # Fixed to stage 1 best
  warmup_ratio:
    min: 0.05
    max: 0.15
```

**Workflow:**
```bash
# 1. Run coarse sweep
wandb launch-sweep stage1-coarse-sweep.yaml \
  --queue gpu-t4-queue \
  --project my-project

# Wait for completion, analyze results

# 2. Adjust stage2 config based on best run

# 3. Run fine sweep
wandb launch-sweep stage2-fine-sweep.yaml \
  --queue gpu-a100-queue \
  --project my-project
```

### Conditional Sweeps (Based on Results)

**Pattern:** Adjust search strategy based on intermediate results

From [W&B Sweeps Overview](https://docs.wandb.ai/models/sweeps) (referenced):
- Custom sweep scheduler enables conditional logic
- Can implement multi-armed bandit algorithms
- Adaptive search based on performance

**Example: Custom conditional scheduler (pseudocode)**
```python
# custom_conditional_scheduler.py
import wandb
from wandb.apis.public import Api

def schedule_next_runs(sweep_id, num_workers):
    api = Api()
    sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")

    # Analyze current results
    runs = list(sweep.runs)
    best_runs = sorted(runs, key=lambda r: r.summary.get('val_loss', float('inf')))[:5]

    # Conditional logic
    if len(runs) < 50:
        # Early phase: Random exploration
        search_method = 'random'
        param_ranges = WIDE_RANGES
    elif best_runs[0].summary['val_loss'] < THRESHOLD:
        # Found good region: Exploit
        search_method = 'bayes'
        param_ranges = narrow_around_best(best_runs)
    else:
        # Haven't found good region: Keep exploring
        search_method = 'random'
        param_ranges = ALTERNATIVE_RANGES

    # Generate next configs
    for i in range(num_workers):
        config = generate_config(search_method, param_ranges)
        launch_training_job(config, sweep_id)
```

### Early Termination with Hyperband

From [W&B Sweeps Early Termination](https://docs.wandb.ai/models/sweeps/define-sweep-configuration#stopping-criteria) (referenced):
- Hyperband algorithm kills underperforming runs early
- Saves compute by not wasting time on bad hyperparameters
- Aggressive pruning → faster sweep completion

**Hyperband sweep configuration:**
```yaml
description: Sweep with Hyperband early termination
method: random
run_cap: 200

# Hyperband configuration
early_terminate:
  type: hyperband
  min_iter: 3  # Minimum epochs before termination
  eta: 3  # Reduction factor (keep 1/3 of runs)
  s: 2  # Number of brackets

job: my-project/training-job:latest
queue: gpu-queue

scheduler:
  num_workers: 40  # High parallelism with early termination

metric:
  name: val_loss
  goal: minimize

parameters:
  learning_rate:
    min: 1e-6
    max: 1e-3
    distribution: log_uniform
  num_layers:
    values: [4, 8, 12, 16, 24]
```

**How Hyperband works with Launch:**
```
Bracket 0 (s=2):
├─ 40 runs start (num_workers=40)
├─ After min_iter=3 epochs: Keep best 13 runs, terminate 27
├─ After 9 epochs: Keep best 4 runs, terminate 9
└─ After 27 epochs: Keep best 1 run, terminate 3

Bracket 1:
├─ 13 runs start
├─ After 9 epochs: Keep best 4, terminate 9
└─ After 27 epochs: Keep best 1, terminate 3

Result: Found best config with ~50% compute of full sweep
```

### Warm-Starting from Previous Sweeps

From [W&B Launch Sweeps Documentation](https://docs.wandb.ai/platform/launch/sweeps-on-launch) (accessed 2025-01-31):
- Resume sweep from previous sweep ID
- Cannot change hyperparameters or job
- Can change scheduler parameters (num_workers, queue)

**Resume existing sweep:**
```bash
# Resume previously stopped sweep
wandb launch-sweep \
  --resume_id <sweep_id> \
  --queue gpu-a100-queue \
  --project my-project

# With updated scheduler config
wandb launch-sweep updated-config.yaml \
  --resume_id <sweep_id> \
  --queue gpu-a100-queue
```

**Warm-start new sweep with knowledge from old sweep:**
```python
# Custom scheduler that uses previous sweep results
import wandb

def warm_start_scheduler(previous_sweep_id, new_sweep_id):
    api = wandb.Api()

    # Get previous sweep best configs
    prev_sweep = api.sweep(f"{entity}/{project}/sweeps/{previous_sweep_id}")
    best_configs = [run.config for run in prev_sweep.best_runs(5)]

    # New sweep: Start with variations of best configs
    for config in best_configs:
        # Small perturbations around best
        new_config = perturb_config(config, noise=0.1)
        launch_training_job(new_config, new_sweep_id)
```

### Nested Sweeps (Architecture + Hyperparameters)

**Pattern:** Sweep over model architectures first, then hyperparameters

**Outer sweep: Model architecture**
```yaml
# architecture-sweep.yaml
description: Sweep over model architectures
method: grid
run_cap: 12

job: my-project/training-job:latest
queue: gpu-queue

scheduler:
  num_workers: 4

metric:
  name: val_f1_score
  goal: maximize

parameters:
  model_architecture:
    values: ['transformer', 'lstm', 'cnn', 'hybrid']
  num_layers:
    values: [4, 8, 12]
  # Use default hyperparameters
  learning_rate:
    value: 1e-4
  batch_size:
    value: 32
```

**Inner sweep: Hyperparameters for best architecture**
```yaml
# hyperparameter-sweep.yaml (created after arch sweep)
description: Hyperparameter tuning for best architecture
method: bayes
run_cap: 100

job: my-project/training-job:latest
queue: gpu-a100-queue

scheduler:
  num_workers: 20

metric:
  name: val_f1_score
  goal: maximize

parameters:
  model_architecture:
    value: 'transformer'  # Best from architecture sweep
  num_layers:
    value: 8  # Best from architecture sweep

  # Now sweep hyperparameters
  learning_rate:
    min: 1e-5
    max: 1e-3
    distribution: log_uniform
  weight_decay:
    min: 0.0
    max: 0.1
  dropout:
    min: 0.0
    max: 0.5
  warmup_steps:
    min: 100
    max: 2000
```

### Complete ARR-COC Sweep Example

**ARR-COC hyperparameter sweep configuration:**
```yaml
# arr-coc-sweep.yaml
description: ARR-COC relevance realization hyperparameter sweep
method: bayes
run_cap: 150

job: arr-coc-project/arr-coc-training:v1
queue: gpu-a100-queue

scheduler:
  num_workers: 30  # 30 parallel runs

# Hyperband early termination (save compute on bad configs)
early_terminate:
  type: hyperband
  min_iter: 5  # Min 5 epochs before termination
  eta: 3

metric:
  name: vqa_accuracy
  goal: maximize

parameters:
  # Core learning hyperparameters
  learning_rate:
    min: 5e-6
    max: 5e-4
    distribution: log_uniform

  weight_decay:
    min: 0.0
    max: 0.1
    distribution: uniform

  batch_size:
    values: [16, 32, 64]

  warmup_ratio:
    min: 0.05
    max: 0.15

  # ARR-COC specific: Relevance realization weights
  propositional_weight:
    min: 0.1
    max: 1.0
    distribution: uniform

  perspectival_weight:
    min: 0.1
    max: 1.0
    distribution: uniform

  participatory_weight:
    min: 0.1
    max: 1.0
    distribution: uniform

  # LOD allocation parameters
  min_tokens_per_patch:
    values: [64, 96, 128]

  max_tokens_per_patch:
    values: [256, 384, 512]

  # Opponent processing tension weight
  compression_particularize_weight:
    min: 0.3
    max: 0.7
```

**Launch ARR-COC sweep:**
```bash
wandb launch-sweep arr-coc-sweep.yaml \
  --queue gpu-a100-queue \
  --project arr-coc-validation
```

**Expected outcomes:**
- 30 parallel training runs on A100 GPUs
- Hyperband terminates poor configs after 5 epochs
- Bayesian optimization focuses on high-impact hyperparameters
- Total ~150 runs exploring ARR-COC hyperparameter space
- Best config: Optimal balance of 3 ways of knowing + LOD allocation

## Sources

**W&B Launch Documentation:**
- [Create sweeps with W&B Launch](https://docs.wandb.ai/platform/launch/sweeps-on-launch) - Launch + Sweeps integration guide (accessed 2025-01-31)
- [W&B Launch Overview](https://docs.wandb.ai/platform/launch) - Launch fundamentals (referenced)
- [Set up launch agent](https://docs.wandb.ai/platform/launch/setup-agent-advanced) - Agent configuration (referenced)

**W&B Sweeps Documentation:**
- [Tutorial: Define, initialize, and run a sweep](https://docs.wandb.ai/models/sweeps/walkthrough) - Basic sweeps workflow (accessed 2025-01-31)
- [Sweeps overview](https://docs.wandb.ai/models/sweeps) - Sweeps fundamentals (referenced)
- [Define sweep configuration](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) - Config options (referenced)

**Web Research:**
- Google Search: "W&B Launch sweeps integration" (accessed 2025-01-31)
- Google Search: "automated hyperparameter tuning wandb launch" (accessed 2025-01-31)
- Google Search: "W&B Launch + Sweeps scale 2024 2025" (accessed 2025-01-31)
- Google Search: "wandb launch sweep agent orchestration" (accessed 2025-01-31)

**Additional References:**
- [wandb/launch-jobs GitHub](https://github.com/wandb/launch-jobs) - Custom scheduler examples (referenced in docs)
- W&B Launch Kubernetes Integration - Agent auto-scaling (referenced)
