# Vertex AI Training Job Debugging & Troubleshooting

**Version:** 1.0
**Target Audience:** ML Engineers running production training on Vertex AI
**Prerequisites:** Vertex AI fundamentals, Cloud Logging basics, container experience

## Overview

Debugging Vertex AI training jobs requires understanding multiple layers: Cloud Logging, container lifecycle, GPU/TPU resource management, and distributed training patterns. This guide provides production-ready debugging strategies for Custom Jobs, Hyperparameter Tuning, and Training Pipelines.

**Key debugging tools:**
- Cloud Logging (structured logs, filtering, real-time streaming)
- Interactive shell (SSH-like debugging for live jobs)
- Cloud Profiler (GPU utilization, memory profiling)
- Container local testing (pre-flight validation)

---

## Section 1: Cloud Logging for Training Jobs (~170 lines)

### 1.1 Accessing Custom Job Logs

**Navigate to logs:**
```bash
# Via gcloud CLI
gcloud logging read "resource.type=ml_job" --limit 50 --format json

# Via Console: Logging > Logs Explorer
# Filter: resource.type="ml_job"
```

**Log structure for Custom Jobs:**
```
resource.type: ml_job
resource.labels.job_id: projects/{project}/locations/{region}/customJobs/{job_id}
resource.labels.task_name: workerpool0-0  # Chief worker
jsonPayload: {training script output}
```

From [Vertex AI Pipelines Logging Documentation](https://docs.cloud.google.com/vertex-ai/docs/pipelines/logging) (accessed 2025-01-31):
- Logs appear within 1-2 minutes of job start
- Enable Cloud Logging API: `gcloud services enable logging.googleapis.com`
- Logs Explorer provides filtering, search, and export capabilities

### 1.2 Log Severity Levels

**Standard severity levels:**
```python
import logging
import sys

# Configure structured logging for Vertex AI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Important: stdout for Cloud Logging capture
)

logger = logging.getLogger(__name__)

# Severity levels (Cloud Logging compatible)
logger.debug("Detailed debugging information")
logger.info("Training progress: epoch 5/100")
logger.warning("GPU utilization below 50%")
logger.error("Failed to load checkpoint from GCS")
logger.critical("Out of memory - job will terminate")
```

**Log filtering by severity:**
```
# Logs Explorer queries
severity >= ERROR
severity = WARNING
jsonPayload.epoch >= 50
```

### 1.3 Structured Logging Best Practices

**Emit structured JSON logs:**
```python
import json
import sys

def log_structured(message, severity="INFO", **kwargs):
    """Emit structured logs for Cloud Logging"""
    log_entry = {
        "severity": severity,
        "message": message,
        "timestamp": time.time(),
        **kwargs  # Additional structured fields
    }
    print(json.dumps(log_entry), file=sys.stdout, flush=True)

# Usage in training script
log_structured(
    "Training metrics",
    severity="INFO",
    epoch=10,
    loss=0.234,
    accuracy=0.891,
    gpu_memory_used_gb=24.5,
    training_samples_per_sec=1250
)
```

From [Aggregate Vertex AI Model Training Logs](https://medium.com/google-cloud/aggregate-vertex-ai-model-training-job-logs-into-a-single-bigquery-table-6e074b90b5c2) (accessed 2025-01-31):
- Structured logs enable BigQuery exports for analysis
- Log aggregation across distributed workers
- Custom metrics tracking for training progress

### 1.4 Log Filtering and Querying

**Common log queries:**
```
# Filter by specific job
resource.labels.job_id="projects/my-project/locations/us-central1/customJobs/123456"

# Filter by worker pool (distributed training)
resource.labels.task_name="workerpool0-0"  # Chief
resource.labels.task_name=~"workerpool1-"  # Workers (regex)

# Time range filtering
timestamp >= "2025-01-31T10:00:00Z"
timestamp < "2025-01-31T12:00:00Z"

# Search for specific patterns
jsonPayload.message =~ ".*OOM.*"  # Out of memory errors
jsonPayload.message =~ ".*CUDA.*"  # GPU errors
severity >= ERROR

# Combined filters
resource.type="ml_job" AND severity=ERROR AND timestamp>="2025-01-31T00:00:00Z"
```

**Advanced query examples:**
```
# Find jobs with high error rates
resource.type="ml_job"
severity>=ERROR
jsonPayload.epoch EXISTS

# Track GPU utilization issues
resource.type="ml_job"
jsonPayload.message=~".*GPU utilization.*"
jsonPayload.gpu_util<50

# Identify slow data loading
resource.type="ml_job"
jsonPayload.message=~".*data loading time.*"
jsonPayload.data_load_time_sec>10
```

### 1.5 Real-Time Log Streaming

**Stream logs during training:**
```bash
# Stream logs in real-time (similar to tail -f)
gcloud logging tail "resource.type=ml_job AND resource.labels.job_id=projects/my-project/locations/us-central1/customJobs/123456"

# Stream with filters
gcloud logging tail \
  "resource.type=ml_job" \
  --format="value(jsonPayload.message)" \
  --filter="severity>=WARNING"
```

**Python SDK for log streaming:**
```python
from google.cloud import logging

client = logging.Client()
logger = client.logger("vertex-ai-training")

# Tail logs programmatically
for entry in logger.list_entries(order_by=logging.DESCENDING, max_results=50):
    print(f"{entry.timestamp}: {entry.payload}")
```

### 1.6 Log Exports to BigQuery

**Export logs for analysis:**
```bash
# Create log sink to BigQuery
gcloud logging sinks create vertex-training-logs \
  bigquery.googleapis.com/projects/my-project/datasets/vertex_logs \
  --log-filter='resource.type="ml_job"'

# Query logs in BigQuery
bq query --use_legacy_sql=false '
SELECT
  timestamp,
  resource.labels.job_id,
  jsonPayload.epoch,
  jsonPayload.loss,
  jsonPayload.accuracy
FROM `my-project.vertex_logs.ml_job_*`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 1000
'
```

From [Vertex AI Audit Logging](https://cloud.google.com/vertex-ai/docs/general/audit-logging) (accessed 2025-01-31):
- Log retention: 30 days default (configurable to 3650 days)
- Audit logs track administrative actions
- Data access logs track prediction requests

### 1.7 Log Retention and Quotas

**Retention policies:**
- Default retention: 30 days
- Maximum retention: 3650 days (10 years)
- Configure via: Logging > Log Storage

**Logging quotas:**
- Free tier: 50 GiB/month
- Paid tier: $0.50 per GiB ingested
- Query costs: $0.01 per GiB scanned

---

## Section 2: Common Issues & Solutions (~170 lines)

### 2.1 Container Failures

**Issue: Image pull errors**
```
ERROR: Failed to pull image: gcr.io/my-project/training:latest
Authentication required
```

**Solutions:**
1. **Check Artifact Registry permissions:**
```bash
# Grant service account permissions
gcloud artifacts repositories add-iam-policy-binding my-repo \
  --location=us-central1 \
  --member="serviceAccount:vertex-sa@my-project.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

2. **Verify image exists:**
```bash
gcloud artifacts docker images list us-central1-docker.pkg.dev/my-project/my-repo
```

3. **Test image locally:**
```bash
docker pull us-central1-docker.pkg.dev/my-project/my-repo/training:latest
docker run -it us-central1-docker.pkg.dev/my-project/my-repo/training:latest /bin/bash
```

From [Vertex AI Custom Container Requirements](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers) (accessed 2025-01-31):
- Container must run as non-root user
- Expose health check endpoint
- Handle SIGTERM gracefully for preemption

**Issue: Container entrypoint failures**
```
ERROR: Container exited with status 127
/bin/bash: /app/train.py: No such file or directory
```

**Solutions:**
```dockerfile
# Correct Dockerfile structure
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN chmod +x train.py

# Ensure Python is executable
ENTRYPOINT ["python", "train.py"]
```

### 2.2 Permission Errors (IAM)

**Issue: GCS access denied**
```
PermissionDenied: 403 Access denied to gs://my-bucket/data/
```

**Required IAM roles for training:**
```bash
# Vertex AI service account needs:
roles/aiplatform.user                    # Submit jobs
roles/storage.objectViewer               # Read from GCS
roles/storage.objectCreator              # Write to GCS
roles/artifactregistry.reader            # Pull containers
roles/logging.logWriter                  # Write logs

# Grant roles to service account
gcloud projects add-iam-policy-binding my-project \
  --member="serviceAccount:vertex-sa@my-project.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

From [Vertex AI Custom Job IAM](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) (accessed 2025-01-31):
- Default service account: `{project-number}-compute@developer.gserviceaccount.com`
- Use custom service accounts for least privilege
- Cross-project permissions require explicit grants

**Common permission decision tree:**
```
Permission Error?
├─ Can't pull container → Check artifactregistry.reader
├─ Can't read data → Check storage.objectViewer
├─ Can't write checkpoints → Check storage.objectCreator
├─ Can't submit job → Check aiplatform.user
└─ Can't write logs → Check logging.logWriter
```

### 2.3 Resource Quota Errors

**Issue: Insufficient quota**
```
ERROR: Quota 'NVIDIA_A100_GPUS' exceeded
Requested: 8, Limit: 4, Region: us-central1
```

**Check current quotas:**
```bash
gcloud compute project-info describe --project=my-project

# Check GPU quotas specifically
gcloud compute regions describe us-central1 \
  --format="value(quotas.filter(metric~'.*GPU.*'))"
```

**Request quota increase:**
1. Console: IAM & Admin > Quotas
2. Filter: `Vertex AI API` + `NVIDIA_A100_GPUS`
3. Select quota > EDIT QUOTAS
4. Request increase (justify business need)
5. Approval time: 2-5 business days

From [GCP Quota Management](https://cloud.google.com/compute/quotas) (accessed 2025-01-31):
- Regional quotas (most common)
- Per-VM-family quotas
- Project-level quotas

**Workaround strategies:**
```python
# Use multiple regions
regions = ["us-central1", "us-west1", "europe-west4"]
for region in regions:
    try:
        job = aiplatform.CustomJob(
            display_name="training",
            worker_pool_specs=worker_spec,
            region=region
        )
        job.run()
        break  # Success
    except google.api_core.exceptions.ResourceExhausted:
        continue  # Try next region
```

### 2.4 Out of Memory (OOM) Errors

**Issue: Container OOM**
```
ERROR: Container killed by OOM (Out of Memory)
Exit code: 137
```

**Diagnosis steps:**
```python
# Add memory monitoring to training script
import psutil
import GPUtil

def log_memory_usage():
    # CPU memory
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3

    # GPU memory
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_mem_used = gpu.memoryUsed
        gpu_mem_total = gpu.memoryTotal
        print(f"GPU {gpu.id}: {gpu_mem_used}/{gpu_mem_total} MB")

    print(f"CPU Memory: {cpu_mem_gb:.2f} GB")

# Log every N steps
if step % 100 == 0:
    log_memory_usage()
```

From [Python OOM on Vertex AI](https://stackoverflow.com/questions/78306360/python-process-oom-killed-on-vertex-ai-despite-low-data-size) (accessed 2025-01-31):
- Container OOM manifests as exit code 137
- Monitor with psutil, GPUtil, or nvidia-smi
- Common cause: data loading in main process

**Solutions:**
```python
# 1. Enable gradient checkpointing (reduces memory 50%+)
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Trade compute for memory
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# 2. Reduce batch size dynamically
try:
    output = model(batch)
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        batch_size = batch_size // 2
        print(f"OOM: Reducing batch size to {batch_size}")

# 3. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. Offload to CPU if needed
model.to('cpu')  # Move model to CPU temporarily
torch.cuda.empty_cache()
```

**Select appropriate machine type:**
```python
# High-memory machine types
machine_specs = {
    "n1-highmem-16": "104 GB RAM",  # Good for data-heavy preprocessing
    "n1-highmem-32": "208 GB RAM",
    "n1-highmem-64": "416 GB RAM",
    "a2-megagpu-16g": "1360 GB RAM + 16x A100 80GB",  # Maximum memory
}

worker_pool_spec = {
    "machine_spec": {
        "machine_type": "n1-highmem-32",  # Choose based on needs
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1
    },
    # ...
}
```

### 2.5 Network Connectivity Issues

**Issue: Can't reach GCS**
```
ConnectionError: Failed to connect to storage.googleapis.com
```

**Solutions:**
```bash
# 1. Check VPC firewall rules
gcloud compute firewall-rules list --filter="name~'.*egress.*'"

# 2. Verify Private Google Access (for private VPCs)
gcloud compute networks subnets describe my-subnet \
  --region=us-central1 \
  --format="get(privateIpGoogleAccess)"

# Enable if false
gcloud compute networks subnets update my-subnet \
  --region=us-central1 \
  --enable-private-ip-google-access

# 3. Test connectivity from training container
# Add to Dockerfile for debugging
RUN apt-get update && apt-get install -y curl dnsutils
```

**Network configuration in Custom Job:**
```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="training",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {"image_uri": "gcr.io/my-project/training:latest"},
    }],
    # VPC configuration
    network="projects/my-project/global/networks/my-vpc",
)
```

### 2.6 GPU Initialization Failures

**Issue: CUDA errors**
```
RuntimeError: CUDA error: device-side assert triggered
RuntimeError: CUDA out of memory
```

**Diagnosis:**
```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

# Check GPU memory
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
```

From [Enable Cloud Profiler for Debugging](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-01-31):
- Cloud Profiler tracks GPU utilization
- Memory timeline shows allocation patterns
- Kernel execution analysis identifies bottlenecks

**Solutions:**
```python
# 1. Clear GPU cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# 2. Enable CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA calls

# 3. Check for NaN/Inf values (common cause of device-side asserts)
def check_tensors(tensors, name=""):
    for i, t in enumerate(tensors):
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"NaN/Inf detected in {name}[{i}]")
            raise ValueError(f"Invalid tensor values in {name}")

# 4. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2.7 Complete Troubleshooting Decision Tree

```
Training Job Failed?
│
├─ Exit code 1 (General error)
│  ├─ Check logs: gcloud logging read "resource.type=ml_job"
│  └─ Look for Python tracebacks
│
├─ Exit code 125 (Container error)
│  ├─ Test container locally: docker run ...
│  └─ Check Dockerfile, entrypoint, permissions
│
├─ Exit code 137 (OOM killed)
│  ├─ Monitor memory: psutil, nvidia-smi
│  ├─ Reduce batch size
│  ├─ Enable gradient checkpointing
│  └─ Select larger machine type
│
├─ Exit code 139 (Segmentation fault)
│  ├─ Check CUDA version compatibility
│  └─ Update PyTorch/TensorFlow
│
├─ "Permission denied"
│  └─ Check IAM roles (storage, artifact registry)
│
├─ "Quota exceeded"
│  └─ Request quota increase or use different region
│
└─ "CUDA error"
   ├─ Enable CUDA_LAUNCH_BLOCKING=1
   ├─ Check for NaN/Inf values
   └─ Clear CUDA cache
```

---

## Section 3: Advanced Debugging Techniques (~160 lines)

### 3.1 Interactive Shell for Live Jobs

**Enable interactive shell:**
```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="training-debug",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {"image_uri": "gcr.io/my-project/training:latest"},
    }],
    # Enable interactive shell (SSH-like debugging)
    enable_web_access=True,
)

job.run(sync=False)
print(f"Interactive shell: {job.web_access_uris}")
```

From [Monitor and Debug with Interactive Shell](https://docs.cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell) (accessed 2025-01-31):
- SSH into running training container
- Inspect file system, logs, processes
- Run commands interactively (nvidia-smi, ps, top)
- Available for Custom Jobs (not Pipelines)

**Using interactive shell:**
```bash
# Navigate to Vertex AI > Training > Custom Jobs
# Click job > "Training debugging" tab > "Launch web terminal"

# Inside interactive shell:
nvidia-smi  # Check GPU status
ps aux | grep python  # Find training process
tail -f /var/log/training.log  # Watch logs
du -sh /tmp/*  # Check disk usage
```

**Interactive debugging workflow:**
1. Enable `enable_web_access=True`
2. Submit job
3. Wait for job to start (2-5 minutes)
4. Click "Launch web terminal" in Console
5. Debug live: inspect variables, check files, monitor resources
6. Fix issues and resubmit

**Limitations:**
- Shell access expires after 1 hour of inactivity
- Only available during job execution
- Not available for preemptible VMs after preemption

### 3.2 Cloud Profiler for Performance Analysis

**Enable Cloud Profiler in training script:**
```python
import google.cloud.profiler

try:
    google.cloud.profiler.start(
        service='vertex-ai-training',
        service_version='1.0.0',
        verbose=3,
    )
except Exception as e:
    print(f"Failed to start profiler: {e}")

# Your training code here
for epoch in range(num_epochs):
    train_one_epoch(model, dataloader, optimizer)
```

**Dockerfile additions for profiling:**
```dockerfile
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

# Install profiler dependencies
RUN pip install google-cloud-profiler

# Enable profiler via environment variable
ENV ENABLE_PROFILER=1
```

From [Enable Cloud Profiler for Debugging](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-01-31):
- CPU profiling: identify compute bottlenecks
- GPU profiling: track kernel execution, memory transfers
- Memory profiling: find memory leaks, allocation patterns
- Wall-clock profiling: overall time distribution

**Analyzing profiler results:**
```
Cloud Console > Profiler > Select service: vertex-ai-training

View types:
- Flame graph: hierarchical call stack visualization
- Top table: functions sorted by CPU/memory usage
- Source view: line-by-line profiling
```

**Common profiling insights:**
```
# CPU bottleneck example
- data_loading: 40% of time (TOO HIGH)
  → Solution: Increase num_workers, prefetch_factor

# GPU bottleneck example
- GPU utilization: 35% (TOO LOW)
  → Solution: Increase batch size, reduce CPU preprocessing

# Memory leak example
- torch.cuda.memory_allocated() increasing linearly
  → Solution: Call torch.cuda.empty_cache(), fix dangling references
```

### 3.3 GPU Utilization Monitoring

**Real-time GPU monitoring:**
```python
import subprocess
import time

def monitor_gpu_utilization(interval_sec=10):
    """Log GPU stats periodically"""
    while True:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )

        print(f"[GPU Monitor] {result.stdout.strip()}")
        time.sleep(interval_sec)

# Run in background thread
import threading
monitor_thread = threading.Thread(target=monitor_gpu_utilization, daemon=True)
monitor_thread.start()
```

**GPU utilization targets:**
```
Ideal GPU utilization: 80-95%
- Below 50%: CPU bottleneck (data loading, preprocessing)
- Above 95%: Good utilization
- 100% sustained: Risk of thermal throttling

Memory utilization: 70-90% of total
- Below 50%: Can increase batch size
- Above 95%: Risk of OOM
```

**Optimization strategies:**
```python
# Increase GPU utilization
1. Increase batch size (if memory allows)
2. Reduce CPU preprocessing (move to GPU)
3. Prefetch data: DataLoader(num_workers=8, prefetch_factor=4)
4. Use mixed precision: torch.cuda.amp
5. Optimize model: fuse operations, remove unnecessary layers

# Example: Optimize data loading
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,  # Parallel data loading (match CPU cores)
    prefetch_factor=4,  # Prefetch 4 batches per worker
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reuse worker processes
)
```

### 3.4 Memory Profiling

**PyTorch memory profiling:**
```python
import torch.cuda as cuda

# Track memory allocation
def log_memory_snapshot():
    print(f"Allocated: {cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved:  {cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max allocated: {cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Detailed memory profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Analyze memory usage
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("memory_trace.json")
```

From [GPU Memory Profiling Best Practices](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai) (accessed 2025-01-31):
- Track allocation patterns to avoid OOM
- Identify memory leaks early
- Optimize batch size based on memory profile

**Memory optimization techniques:**
```python
# 1. Gradient accumulation (simulate larger batch size)
accumulation_steps = 4
optimizer.zero_grad()

for i, (input, target) in enumerate(train_loader):
    output = model(input)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Delete intermediate tensors
output = model(input)
loss = criterion(output, target)
loss.backward()
del output, loss  # Free memory immediately
torch.cuda.empty_cache()

# 3. Use model.eval() during validation
model.eval()
with torch.no_grad():  # Disable gradient computation
    for input, target in val_loader:
        output = model(input)
        # No backward pass → saves memory
```

### 3.5 Distributed Training Debugging

**Common distributed training errors:**
```
RuntimeError: NCCL error: unhandled system error
RuntimeError: Timed out initializing process group
RuntimeError: Rank 2 disconnected
```

**Debug distributed setup:**
```python
import torch.distributed as dist

def setup_distributed_debugging():
    # Enable verbose NCCL logging
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

    # Increase timeout (default 30 minutes)
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=60),
    )

    # Verify ranks can communicate
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}/{world_size}] Initialized successfully")

    # Test all-reduce
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print(f"[Rank {rank}] All-reduce result: {tensor.item()}")  # Should be world_size

# Detect hanging operations
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Distributed operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

try:
    dist.all_reduce(tensor)
finally:
    signal.alarm(0)  # Disable alarm
```

From [Debugging Vertex AI Training Jobs](https://cloud.google.com/blog/topics/developers-practitioners/debugging-vertex-ai-training-jobs-interactive-shell) (accessed 2025-01-31):
- NCCL errors often caused by network configuration
- Check firewall rules allow inter-worker communication
- Verify all workers start successfully (check logs per worker)

**Distributed training log analysis:**
```bash
# View logs from all workers
gcloud logging read \
  "resource.type=ml_job AND resource.labels.job_id=projects/my-project/locations/us-central1/customJobs/123456" \
  --format="table(resource.labels.task_name, jsonPayload.message)"

# Expected output:
# workerpool0-0 (chief)    | Rank 0/4 initialized
# workerpool1-0 (worker)   | Rank 1/4 initialized
# workerpool1-1 (worker)   | Rank 2/4 initialized
# workerpool1-2 (worker)   | Rank 3/4 initialized
```

### 3.6 Container Local Testing

**Test container locally before submitting:**
```bash
# 1. Build container
docker build -t gcr.io/my-project/training:latest .

# 2. Run with Vertex AI environment variables
docker run -it --rm \
  --gpus all \
  -e AIP_MODEL_DIR=/tmp/model \
  -e AIP_CHECKPOINT_DIR=/tmp/checkpoint \
  -e AIP_TENSORBOARD_LOG_DIR=/tmp/logs \
  -v /path/to/data:/data:ro \
  gcr.io/my-project/training:latest

# 3. Test with mock GCS paths
docker run -it --rm \
  -e GCS_BUCKET=gs://my-bucket \
  -e DATA_PATH=/data \
  gcr.io/my-project/training:latest

# 4. Interactive debugging
docker run -it --rm \
  --entrypoint /bin/bash \
  gcr.io/my-project/training:latest

# Inside container:
python -c "import torch; print(torch.cuda.is_available())"
```

**Local testing checklist:**
- [ ] Container starts successfully
- [ ] Dependencies installed correctly
- [ ] Training script runs for 1 epoch
- [ ] GPU detected (if using GPUs)
- [ ] Logs written to stdout
- [ ] Checkpoints saved correctly
- [ ] Graceful shutdown on SIGTERM

### 3.7 W&B Integration for Debugging

**Log debugging info to W&B:**
```python
import wandb

wandb.init(project="vertex-ai-training", name="debug-run")

# Log system metrics
wandb.log({
    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    "cpu_memory_gb": psutil.virtual_memory().used / 1024**3,
    "gpu_utilization": GPUtil.getGPUs()[0].load * 100,
    "data_loading_time_sec": data_load_time,
})

# Log gradients for debugging NaN/Inf
wandb.watch(model, log="all", log_freq=100)

# Log error context on crash
try:
    train_model()
except Exception as e:
    wandb.alert(
        title="Training crashed",
        text=f"Error: {str(e)}\nTraceback: {traceback.format_exc()}",
    )
    raise
```

---

## Sources

**Google Cloud Documentation:**
- [Vertex AI Pipelines Logging](https://docs.cloud.google.com/vertex-ai/docs/pipelines/logging) - accessed 2025-01-31
- [Monitor and Debug with Interactive Shell](https://docs.cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell) - accessed 2025-01-31
- [Enable Cloud Profiler for Debugging](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) - accessed 2025-01-31
- [Vertex AI Custom Container Requirements](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers) - accessed 2025-01-31
- [Vertex AI Audit Logging](https://cloud.google.com/vertex-ai/docs/general/audit-logging) - accessed 2025-01-31
- [GCP Quota Management](https://cloud.google.com/compute/quotas) - accessed 2025-01-31
- [Troubleshoot Dataflow OOM](https://docs.cloud.google.com/dataflow/docs/guides/troubleshoot-oom) - accessed 2025-01-31

**Community Resources:**
- [Debugging Vertex AI Training Jobs with Interactive Shell](https://cloud.google.com/blog/topics/developers-practitioners/debugging-vertex-ai-training-jobs-interactive-shell) - Google Cloud Blog, accessed 2025-01-31
- [Aggregate Vertex AI Model Training Logs](https://medium.com/google-cloud/aggregate-vertex-ai-model-training-job-logs-into-a-single-bigquery-table-6e074b90b5c2) - Medium, accessed 2025-01-31
- [GPU Memory Profiling Best Practices](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai) - Ohio Supercomputer Center, accessed 2025-01-31
- [Python OOM on Vertex AI](https://stackoverflow.com/questions/78306360/python-process-oom-killed-on-vertex-ai-despite-low-data-size) - Stack Overflow, accessed 2025-01-31

**Additional References:**
- Stack Overflow discussions on Vertex AI debugging patterns
- GitHub issues for CUDA OOM and memory leak troubleshooting
- Google Developer forums on Vertex AI Custom Job errors
