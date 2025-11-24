# W&B Launch for LLM Training Automation

## Overview

W&B Launch enables automated, scalable LLM/VLM training workflows by transforming manual training scripts into containerized jobs that can be executed across diverse compute resources. Launch orchestrates the complete training lifecycle: from job submission to execution on GPUs, to checkpoint management and evaluation—all with minimal manual intervention.

**Core value proposition**: Convert `python train.py` into production-grade, reproducible training pipelines deployable to any compute environment (local, Kubernetes, cloud) with automated resource management, queue-based execution, and built-in checkpoint/artifact handling.

This guide covers practical LLM training automation patterns, multi-GPU/distributed training orchestration, and cost optimization strategies for production ML workflows.

---

## Section 1: LLM Training Automation with Launch (~170 lines)

### Launch Job Creation for LLM Fine-Tuning

**From a training run to automated job:**

```python
# Step 1: Run your LLM training script normally with wandb tracking
import wandb
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

def train_llm(config):
    # Initialize W&B run
    run = wandb.init(
        project="llm-training",
        config=config,
        job_type="train"
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    # Training setup
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_strategy="epoch",
        logging_steps=100,
        report_to="wandb",  # Automatic W&B integration
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train and save checkpoint as artifact
    trainer.train()

    # Save final checkpoint to W&B Artifacts
    artifact = wandb.Artifact(
        name=f"{config['model_name']}-finetuned",
        type="model",
        metadata={"epochs": config["epochs"], "loss": trainer.state.log_history[-1]["loss"]}
    )
    artifact.add_dir("./checkpoints")
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    config = {
        "model_name": "gpt2",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5
    }
    train_llm(config)
```

**Step 2: Create Launch job from successful run**

From W&B UI:
1. Navigate to successful training run
2. Click "Launch" tab → "Create Job"
3. Launch automatically captures:
   - Python environment (requirements.txt or conda.yml)
   - Git repository state (commit hash, branch)
   - Training script entry point
   - W&B config parameters

From CLI:
```bash
# Create job from existing run
wandb launch --uri https://wandb.ai/entity/project/runs/run_id --create-job

# Or create job from git repo + docker image
wandb job create \
  --name "llm-training-job" \
  --project "llm-training" \
  --git-repo "https://github.com/org/llm-training.git" \
  --entry-point "train.py" \
  --docker-image "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"
```

**Step 3: Submit job to queue for automated execution**

```bash
# Submit job with modified hyperparameters
wandb launch \
  --job entity/project/llm-training-job:latest \
  --queue gpu-training \
  --config '{"epochs": 5, "batch_size": 16}'
```

### HuggingFace Transformers + Launch Integration

**Automatic checkpoint versioning pattern:**

```python
from transformers import TrainingArguments
import wandb

# TrainingArguments with W&B integration
training_args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",  # Enables automatic logging
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=3,  # Keep only last 3 checkpoints locally
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# W&B will automatically:
# - Log training metrics (loss, learning_rate, etc.)
# - Track gradients and weights (if enabled)
# - Version checkpoints as artifacts
# - Link artifacts to training run
```

**Environment Setup for Launch Jobs:**

Create `requirements.txt` for reproducibility:
```
transformers==4.35.0
torch==2.1.0
wandb>=0.17.1
datasets==2.14.0
accelerate==0.24.0
```

Or use Docker for complex dependencies:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install training dependencies
RUN pip install transformers==4.35.0 wandb>=0.17.1 datasets accelerate

# Copy training code
COPY train.py /workspace/
WORKDIR /workspace

# Entry point
ENTRYPOINT ["python", "train.py"]
```

### Resume from Checkpoint Patterns

**Automatic checkpoint resumption in Launch jobs:**

```python
import wandb
import os

def train_with_resumption(config):
    run = wandb.init(project="llm-training", config=config, resume="allow")

    # Check for previous checkpoint artifact
    checkpoint_artifact = None
    if run.resumed:
        try:
            # Fetch last checkpoint from previous run
            api = wandb.Api()
            previous_run = api.run(f"{run.entity}/{run.project}/{run.id}")

            # Get latest model artifact
            artifacts = previous_run.logged_artifacts()
            checkpoint_artifacts = [a for a in artifacts if a.type == "model"]
            if checkpoint_artifacts:
                checkpoint_artifact = checkpoint_artifacts[-1]
                artifact_dir = checkpoint_artifact.download()
                print(f"Resuming from checkpoint: {artifact_dir}")
        except Exception as e:
            print(f"No checkpoint found, starting fresh: {e}")

    # Initialize model from checkpoint or fresh
    if checkpoint_artifact:
        model = AutoModelForCausalLM.from_pretrained(artifact_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    # Continue training...
    trainer = Trainer(model=model, args=training_args)
    trainer.train(resume_from_checkpoint=artifact_dir if checkpoint_artifact else None)

    # Save new checkpoint
    artifact = wandb.Artifact(name="checkpoint", type="model")
    artifact.add_dir("./checkpoints")
    run.log_artifact(artifact)
```

**Launch queue with preemption handling:**

When using spot instances (see Section 3), jobs may be interrupted. Launch automatically retries failed jobs:

```bash
# Queue configuration with retry policy
wandb launch-queue create \
  --name "spot-gpu-training" \
  --resource kubernetes \
  --config '{
    "max_retries": 3,
    "retry_on": ["preempted", "evicted"],
    "checkpoint_frequency": "epoch"
  }'
```

From [W&B Launch Tutorial: Basics](https://docs.wandb.ai/platform/launch/walkthrough) (accessed 2025-01-31):
- Launch jobs are blueprints containing code, environment, and configuration
- Jobs can be created from runs, git repos, or Docker images
- Queue-based execution enables asynchronous, multi-user workflows

From [W&B Distributed Training Guide](https://docs.wandb.ai/models/track/log/distributed-training) (accessed 2025-01-31):
- HuggingFace Trainer automatically integrates with W&B when `report_to="wandb"`
- Checkpoint artifacts enable automatic versioning and reproducibility

---

## Section 2: Multi-GPU and Distributed Training with Launch (~170 lines)

### Single-Node Multi-GPU (DataParallel, DDP)

**PyTorch DistributedDataParallel (DDP) + Launch:**

```python
# train_ddp.py - Multi-GPU training script
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import os

def setup_distributed():
    """Initialize distributed training environment"""
    # Launch sets these environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size

def train_ddp(config):
    local_rank, world_size = setup_distributed()

    # Only rank 0 initializes wandb (single-process logging)
    if local_rank == 0:
        run = wandb.init(project="llm-ddp", config=config)

    # Load model and move to correct GPU
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = model.to(local_rank)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Training loop
    for epoch in range(config["epochs"]):
        for batch in train_dataloader:
            # Forward pass (data automatically distributed across GPUs)
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass (gradients synchronized across GPUs)
            loss.backward()
            optimizer.step()

            # Log only from rank 0
            if local_rank == 0:
                wandb.log({"loss": loss.item(), "epoch": epoch})

    if local_rank == 0:
        run.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train_ddp(config)
```

**Launch job resource specification for multi-GPU:**

```yaml
# launch_config.yaml - Specify GPU requirements
resource: kubernetes  # or "sagemaker", "vertex-ai"
resource_args:
  kubernetes:
    spec:
      containers:
        - name: training
          resources:
            limits:
              nvidia.com/gpu: 4  # Request 4 GPUs
            requests:
              nvidia.com/gpu: 4
          env:
            - name: NCCL_DEBUG
              value: "INFO"  # Debug distributed training
```

**Submit multi-GPU job:**

```bash
wandb launch \
  --job entity/project/llm-ddp-job:latest \
  --queue gpu-4x \
  --resource-args launch_config.yaml
```

### Multi-Node Distributed Training

**Multi-node DDP with Launch + Kubernetes:**

```python
# train_multi_node.py
import torch.distributed as dist
import os

def setup_multi_node():
    """Setup for multi-node distributed training"""
    # Launch/K8s sets these automatically
    rank = int(os.environ.get("RANK", 0))  # Global rank across all nodes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Rank within node
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # Total processes
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU communication
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def train_multi_node(config):
    rank, local_rank, world_size = setup_multi_node()

    # Initialize wandb on all ranks (track each process separately)
    run = wandb.init(
        project="llm-multinode",
        config=config,
        group="multinode-experiment",  # Group all nodes together
        job_type="worker",
        name=f"rank-{rank}",
        settings=wandb.Settings(
            x_label=f"rank_{rank}",  # Label for system metrics
            mode="shared",  # Share run across processes (v0.19.9+)
            x_primary=(rank == 0),  # Rank 0 is primary
        ),
        id=os.environ.get("WANDB_RUN_ID") if rank > 0 else None,  # Share run ID
    )

    # Model setup with DDP
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Training (data automatically sharded across nodes)
    for epoch in range(config["epochs"]):
        # Use DistributedSampler to shard data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config["batch_size"])

        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # All ranks log to shared run
            wandb.log({"loss": loss.item(), "rank": rank, "epoch": epoch})

    run.finish()
    dist.destroy_process_group()
```

**Kubernetes Launch configuration for multi-node:**

```yaml
# k8s_multinode_config.yaml
resource: kubernetes
resource_args:
  kubernetes:
    # Use Volcano for multi-node job scheduling
    job_spec:
      apiVersion: batch.volcano.sh/v1alpha1
      kind: Job
      spec:
        minAvailable: 8  # Minimum pods needed (2 nodes × 4 GPUs)
        tasks:
          - replicas: 2  # 2 nodes
            name: worker
            template:
              spec:
                containers:
                  - name: training
                    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
                    command: ["python", "train_multi_node.py"]
                    resources:
                      limits:
                        nvidia.com/gpu: 4  # 4 GPUs per node
                    env:
                      - name: NCCL_DEBUG
                        value: INFO
                      - name: NCCL_IB_DISABLE
                        value: "1"  # Disable InfiniBand if not available
```

Submit multi-node job:
```bash
wandb launch \
  --job entity/project/multinode-job:latest \
  --queue k8s-multinode \
  --resource-args k8s_multinode_config.yaml \
  --config '{"epochs": 10, "batch_size": 32}'
```

### DeepSpeed Integration with Launch

**DeepSpeed ZeRO for training large LLMs:**

```python
# train_deepspeed.py
import deepspeed
import wandb

def train_with_deepspeed(config):
    run = wandb.init(project="llm-deepspeed", config=config)

    # DeepSpeed config for ZeRO-3 (partition optimizer states, gradients, and parameters)
    ds_config = {
        "train_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,  # ZeRO-3: full parameter sharding
            "offload_optimizer": {"device": "cpu"},  # Offload to CPU
            "offload_param": {"device": "cpu"},
        },
        "wandb": {"enabled": True, "project": "llm-deepspeed"},  # W&B integration
    }

    # Initialize model and DeepSpeed engine
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # Training loop
    for epoch in range(config["epochs"]):
        for step, batch in enumerate(train_dataloader):
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            # DeepSpeed automatically logs to W&B if enabled in config
            if step % 100 == 0:
                wandb.log({"loss": loss.item(), "step": step})

    # Save DeepSpeed checkpoint
    model_engine.save_checkpoint("./checkpoints")
    run.finish()
```

**Launch job for DeepSpeed:**

```bash
# DeepSpeed requires specific launch command
wandb launch \
  --job entity/project/deepspeed-job:latest \
  --queue deepspeed-gpu \
  --entry-point "deepspeed --num_gpus=4 train_deepspeed.py"
```

### FSDP (Fully Sharded Data Parallel)

**PyTorch FSDP for memory-efficient training:**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def train_with_fsdp(config):
    setup_distributed()
    run = wandb.init(project="llm-fsdp", config=config)

    # FSDP wrapping policy
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1e6  # Wrap layers with >1M params
    )

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
    )

    # Training with memory-efficient parameter sharding
    # (Similar training loop as DDP)
```

From [W&B Launch Multi-GPU Guide](https://wandb.ai/wandb_fc/articles/reports/Multi-GPU-Training-Using-PyTorch-Lightning--VmlldzozMTk3NTk) (accessed 2025-01-31):
- DDP is the recommended approach for multi-GPU training in PyTorch
- NCCL backend provides optimal GPU-to-GPU communication
- DistributedSampler ensures data is evenly sharded across processes

From [W&B Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) (accessed 2025-01-31):
- Shared mode (`x_primary=True`) enables multi-process logging to single run
- System metrics from all nodes aggregated in W&B UI
- Console logs can be filtered by rank label

---

## Section 3: Cost Optimization with Launch (~160 lines)

### Spot Instance Patterns

**Spot instances provide 70-90% cost savings but can be preempted:**

```python
# train_with_spot.py - Checkpoint-aware training for spot instances
import wandb
import os
import signal
import sys

class SpotInstanceHandler:
    """Handle spot instance interruptions gracefully"""

    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.interrupted = False

        # Register signal handlers for spot termination
        signal.signal(signal.SIGTERM, self.handle_interruption)
        signal.signal(signal.SIGINT, self.handle_interruption)

    def handle_interruption(self, signum, frame):
        """Save checkpoint when spot instance is terminated"""
        print("Spot instance termination signal received. Saving checkpoint...")
        self.interrupted = True
        # Checkpoint saving handled in training loop

def train_with_spot_resilience(config):
    handler = SpotInstanceHandler()
    run = wandb.init(project="llm-spot", config=config, resume="allow")

    # Resume from last checkpoint if available
    checkpoint_artifact = load_latest_checkpoint(run)
    start_epoch = checkpoint_artifact.metadata.get("epoch", 0) if checkpoint_artifact else 0

    model = load_model_from_checkpoint_or_fresh(config, checkpoint_artifact)
    trainer = Trainer(model=model, args=training_args)

    for epoch in range(start_epoch, config["epochs"]):
        # Check for interruption signal before each epoch
        if handler.interrupted:
            print(f"Saving checkpoint at epoch {epoch} before termination")
            save_checkpoint(model, epoch, run)
            sys.exit(0)

        # Train for one epoch
        trainer.train()

        # Save checkpoint after each epoch (frequent checkpointing for spot)
        save_checkpoint(model, epoch, run)

        # Log progress
        wandb.log({"epoch_completed": epoch})

    run.finish()

def save_checkpoint(model, epoch, run):
    """Save checkpoint and upload to W&B Artifacts"""
    checkpoint_path = f"./checkpoints/epoch_{epoch}"
    model.save_pretrained(checkpoint_path)

    artifact = wandb.Artifact(
        name="checkpoint-spot",
        type="model",
        metadata={"epoch": epoch, "spot_resilient": True}
    )
    artifact.add_dir(checkpoint_path)
    run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch}"])
```

**Launch queue configuration for spot instances:**

```yaml
# spot_queue_config.yaml
resource: kubernetes
resource_args:
  kubernetes:
    spec:
      # Use spot/preemptible node pool
      nodeSelector:
        cloud.google.com/gke-preemptible: "true"  # GKE
        # Or for AWS: eks.amazonaws.com/capacityType: SPOT
      tolerations:
        - key: "cloud.google.com/gke-preemptible"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      # Enable automatic restart on preemption
      restartPolicy: OnFailure
```

**Queue-level retry policy:**

```bash
# Create queue with spot-aware retry policy
wandb launch-queue create \
  --name "spot-gpu-training" \
  --resource kubernetes \
  --config '{
    "max_retries": 5,
    "retry_on": ["preempted", "evicted", "node-failure"],
    "backoff_multiplier": 1.5,
    "checkpoint_on_preemption": true
  }'
```

### Auto-Scaling Based on Queue Depth

**Dynamic resource allocation based on pending jobs:**

```python
# queue_autoscaler.py - Monitor queue and scale resources
import wandb
from kubernetes import client, config

def monitor_and_scale_queue(queue_name, min_nodes=1, max_nodes=10):
    """Scale Kubernetes node pool based on Launch queue depth"""
    api = wandb.Api()
    k8s_apps = client.AppsV1Api()

    while True:
        # Get queue depth
        queue = api.queue(queue_name)
        pending_jobs = len([j for j in queue.jobs if j.state == "pending"])

        # Calculate desired nodes (1 node per 4 jobs)
        desired_nodes = min(max_nodes, max(min_nodes, (pending_jobs + 3) // 4))

        # Scale node pool
        scale_node_pool(desired_nodes)

        print(f"Queue depth: {pending_jobs}, Scaled to {desired_nodes} nodes")
        time.sleep(60)  # Check every minute

# Run as sidecar service alongside Launch agent
```

**Kubernetes Horizontal Pod Autoscaler for Launch agents:**

```yaml
# agent_autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: launch-agent-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: launch-agent
  minReplicas: 1
  maxReplicas: 20
  metrics:
    - type: External
      external:
        metric:
          name: wandb_queue_depth
          selector:
            matchLabels:
              queue: gpu-training
        target:
          type: AverageValue
          averageValue: "4"  # 4 jobs per agent
```

### Resource Utilization Monitoring

**Track GPU utilization and cost per job:**

```python
# Log GPU metrics and compute costs
import wandb
import pynvml

def log_gpu_metrics():
    """Log detailed GPU utilization to W&B"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    metrics = {}
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics[f"gpu_{i}_utilization"] = util.gpu
        metrics[f"gpu_{i}_memory_utilization"] = util.memory

        # Memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics[f"gpu_{i}_memory_used_gb"] = mem_info.used / 1e9
        metrics[f"gpu_{i}_memory_total_gb"] = mem_info.total / 1e9

        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        metrics[f"gpu_{i}_temperature_c"] = temp

    wandb.log(metrics)

# Call periodically during training
for step in range(training_steps):
    if step % 100 == 0:
        log_gpu_metrics()
```

**Cost tracking with cloud provider metadata:**

```python
# Track compute costs per job
def log_training_cost(run, instance_type, duration_hours):
    """Calculate and log training cost"""
    # Cloud provider pricing (example rates)
    pricing = {
        "p3.2xlarge": 3.06,  # AWS, 1x V100
        "p3.8xlarge": 12.24,  # AWS, 4x V100
        "n1-highmem-8-t4": 0.95,  # GCP, 1x T4 (spot)
        "n1-highmem-32-v100": 2.48,  # GCP, 1x V100 (spot)
    }

    cost_per_hour = pricing.get(instance_type, 0)
    total_cost = cost_per_hour * duration_hours

    run.summary["compute_cost_usd"] = total_cost
    run.summary["instance_type"] = instance_type
    run.summary["training_duration_hours"] = duration_hours

    wandb.log({
        "cost_per_hour": cost_per_hour,
        "cumulative_cost": total_cost,
    })
```

### Training Time Estimation

**Estimate training time for queue scheduling:**

```python
def estimate_training_time(config, dataset_size):
    """Estimate training time based on model size and dataset"""
    # Rough estimates (adjust based on benchmarks)
    model_size_params = {
        "gpt2": 124e6,
        "gpt2-medium": 355e6,
        "gpt2-large": 774e6,
        "llama-7b": 7e9,
        "llama-13b": 13e9,
    }

    params = model_size_params.get(config["model_name"], 1e9)
    batch_size = config["batch_size"]
    num_gpus = config.get("num_gpus", 1)

    # Estimate throughput (tokens/sec) - adjust based on hardware
    throughput_per_gpu = {
        "V100": 5000,
        "A100": 15000,
        "H100": 30000,
    }[config.get("gpu_type", "V100")]

    total_throughput = throughput_per_gpu * num_gpus
    total_tokens = dataset_size * config["epochs"]

    estimated_seconds = total_tokens / total_throughput
    estimated_hours = estimated_seconds / 3600

    return {
        "estimated_hours": estimated_hours,
        "estimated_cost_usd": estimated_hours * config.get("cost_per_hour", 3.0),
        "estimated_tokens_processed": total_tokens,
    }

# Use in job submission
estimate = estimate_training_time(config, len(train_dataset))
print(f"Estimated training time: {estimate['estimated_hours']:.2f} hours")
print(f"Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
```

### Queue Priority for Important Jobs

**Priority-based job scheduling:**

```bash
# Submit high-priority job (e.g., production model retraining)
wandb launch \
  --job entity/project/critical-training:latest \
  --queue gpu-training \
  --priority 10 \  # Higher priority (0-10 scale)
  --config '{"epochs": 5}'

# Submit low-priority exploratory job
wandb launch \
  --job entity/project/exploratory:latest \
  --queue gpu-training \
  --priority 1 \  # Lower priority
  --config '{"epochs": 1}'
```

**Queue configuration with priority levels:**

```yaml
# priority_queue_config.yaml
priority_levels:
  critical: 10  # Production model updates
  high: 7       # Validated experiments
  normal: 5     # Standard training runs
  low: 2        # Exploratory/ablation studies

# Jobs are executed in priority order within FIFO
```

### Preemption Handling and Recovery

**Automatic job recovery after preemption:**

```python
# Launch automatically retries preempted jobs
# Configure retry behavior in queue
queue_config = {
    "max_retries": 3,
    "retry_on": ["preempted", "evicted"],
    "retry_delay_seconds": 60,  # Wait before retry
    "exponential_backoff": True,  # Increase delay on repeated failures
}

# Jobs resume from last checkpoint automatically if resume="allow" in wandb.init()
```

From [W&B Launch Documentation](https://wandb.ai/site/launch) (accessed 2025-01-31):
- Launch enables automated ML workflow scaling from local to distributed compute
- Queue-based execution supports prioritization and load balancing
- Built-in retry mechanisms handle spot instance preemption

From [Running W&B Launch on GKE](https://cloud.google.com/blog/products/containers-kubernetes/running-weights-and-biases-launch-ml-platform-on-gke/) (accessed 2025-01-31):
- GKE spot instances provide 60-91% cost savings vs on-demand
- Kubernetes node autoscaling adapts to queue depth
- Volcano scheduler enables efficient multi-node job orchestration

---

## Sources

**W&B Documentation:**
- [Tutorial: W&B Launch Basics](https://docs.wandb.ai/platform/launch/walkthrough) - accessed 2025-01-31
- [Log Distributed Training Experiments](https://docs.wandb.ai/models/track/log/distributed-training) - accessed 2025-01-31
- [W&B Launch Platform](https://wandb.ai/site/launch) - accessed 2025-01-31

**Technical Guides:**
- [Running W&B Launch ML Platform on GKE](https://cloud.google.com/blog/products/containers-kubernetes/running-weights-and-biases-launch-ml-platform-on-gke/) - accessed 2025-01-31
- [Multi-GPU Training Using PyTorch Lightning](https://wandb.ai/wandb_fc/articles/reports/Multi-GPU-Training-Using-PyTorch-Lightning--VmlldzozMTk3NTk) - accessed 2025-01-31
- [Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) - accessed 2025-01-31

**Additional References:**
- W&B GitHub Examples: [PyTorch DDP Example](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)
- PyTorch Documentation: [Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- HuggingFace Transformers: [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer)

---

**Integration with ARR-COC-VIS Training:**
- Use Launch for automated VLM fine-tuning experiments
- Queue-based execution enables systematic ablation studies (3 ways of knowing)
- Multi-GPU training accelerates relevance realization learning
- Checkpoint artifacts ensure reproducibility of visual compression experiments
- Cost optimization critical for large-scale visual token budget search
