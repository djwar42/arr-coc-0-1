# Vertex AI Logging, Debugging & Troubleshooting

**Version:** 1.0
**Target Audience:** ML Engineers debugging Vertex AI training and serving infrastructure
**Prerequisites:** Cloud Logging basics, Vertex AI fundamentals, BigQuery familiarity

## Overview

Debugging Vertex AI systems requires mastery of multiple observability tools: Cloud Logging for structured logs, Cloud Trace for request latency analysis, Cloud Profiler for performance bottlenecks, and detailed billing exports for cost spike investigation. This guide provides production-ready debugging workflows specifically for Vertex AI Custom Jobs, endpoints, pipelines, and training infrastructure.

**Key debugging focus:**
- Cloud Logging filters for Vertex AI resource types
- Common error pattern recognition (OOM, quota, permissions, network)
- Request latency analysis with Cloud Trace
- Performance profiling (CPU, GPU, memory)
- Cost spike root cause analysis
- arr-coc-0-1 specific debugging workflows

---

## Section 1: Cloud Logging Filters for Vertex AI (~120 lines)

### 1.1 Vertex AI Resource Types

**Primary Vertex AI resource types:**
```
resource.type="aiplatform.googleapis.com/CustomJob"       # Custom training jobs
resource.type="aiplatform.googleapis.com/Endpoint"        # Prediction endpoints
resource.type="aiplatform.googleapis.com/Pipeline"        # Vertex AI Pipelines
resource.type="aiplatform.googleapis.com/Model"           # Model registry operations
resource.type="aiplatform.googleapis.com/BatchPredictionJob"
resource.type="aiplatform.googleapis.com/HyperparameterTuningJob"
resource.type="ml_job"                                     # Legacy AI Platform jobs
```

From [Vertex AI Audit Logging](https://docs.cloud.google.com/vertex-ai/docs/general/audit-logging) (accessed 2025-11-16):
- Logs appear within 1-2 minutes of job start
- Audit logs track administrative actions (create, delete, update)
- Data access logs track prediction requests
- Cloud Logging API must be enabled: `gcloud services enable logging.googleapis.com`

**Filter by specific Custom Job:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
resource.labels.job_id="projects/my-project/locations/us-central1/customJobs/1234567890"
```

**Filter by worker pool (distributed training):**
```
resource.type="ml_job"
resource.labels.task_name="workerpool0-0"  # Chief worker
resource.labels.task_name=~"workerpool1-"  # All workers (regex)
```

**Filter by endpoint predictions:**
```
resource.type="aiplatform.googleapis.com/Endpoint"
resource.labels.endpoint_id="1234567890"
protoPayload.methodName="google.cloud.aiplatform.v1.PredictionService.Predict"
```

### 1.2 Time-Based Filtering

**Recent activity (last hour):**
```
resource.type="aiplatform.googleapis.com/CustomJob"
timestamp >= "2025-11-16T10:00:00Z"
timestamp < "2025-11-16T11:00:00Z"
```

**Training job lifecycle events:**
```
# Job start
protoPayload.methodName="google.cloud.aiplatform.v1.JobService.CreateCustomJob"

# Job completion
jsonPayload.state="JOB_STATE_SUCCEEDED"
jsonPayload.state="JOB_STATE_FAILED"
jsonPayload.state="JOB_STATE_CANCELLED"
```

**Real-time log streaming:**
```bash
# Stream logs for active training job
gcloud logging tail \
  "resource.type=aiplatform.googleapis.com/CustomJob AND \
   resource.labels.job_id=projects/my-project/locations/us-central1/customJobs/123" \
  --format="value(jsonPayload.message)"
```

### 1.3 Advanced Log Queries

**Find all failed jobs in last 24 hours:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
jsonPayload.state="JOB_STATE_FAILED"
timestamp >= "2025-11-15T00:00:00Z"
```

**Track GPU utilization issues:**
```
resource.type="ml_job"
jsonPayload.message=~".*GPU utilization.*"
jsonPayload.gpu_util < 50
```

**Identify slow data loading:**
```
resource.type="ml_job"
jsonPayload.message=~".*data loading time.*"
jsonPayload.data_load_time_sec > 10
```

**Multi-region job tracking:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
(resource.labels.location="us-central1" OR
 resource.labels.location="us-west1" OR
 resource.labels.location="europe-west4")
severity >= ERROR
```

### 1.4 Log Export to BigQuery

**Enable structured log export:**
```bash
# Create log sink to BigQuery dataset
gcloud logging sinks create vertex-ai-logs \
  bigquery.googleapis.com/projects/my-project/datasets/vertex_logs \
  --log-filter='resource.type="aiplatform.googleapis.com/CustomJob" OR \
                resource.type="aiplatform.googleapis.com/Endpoint"'
```

**Query exported logs in BigQuery:**
```sql
SELECT
  timestamp,
  resource.labels.job_id,
  jsonPayload.state,
  jsonPayload.error.message AS error_message,
  severity
FROM `my-project.vertex_logs.aiplatform_googleapis_com_CustomJob_*`
WHERE DATE(timestamp) = CURRENT_DATE()
  AND severity >= 'ERROR'
ORDER BY timestamp DESC
LIMIT 100
```

**Aggregate error patterns:**
```sql
SELECT
  jsonPayload.error.code AS error_code,
  COUNT(*) AS error_count,
  ARRAY_AGG(DISTINCT resource.labels.job_id LIMIT 5) AS sample_jobs
FROM `my-project.vertex_logs.aiplatform_googleapis_com_CustomJob_*`
WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  AND jsonPayload.state = 'JOB_STATE_FAILED'
GROUP BY error_code
ORDER BY error_count DESC
```

---

## Section 2: Log Severity Levels & Structured Logging (~100 lines)

### 2.1 Cloud Logging Severity Levels

**Standard severity hierarchy:**
```
DEFAULT    # Default log level (lowest)
DEBUG      # Detailed debugging information
INFO       # Routine information (training progress, checkpoints saved)
NOTICE     # Normal but significant (model deployed, job completed)
WARNING    # Potential issues (GPU utilization low, retry attempts)
ERROR      # Error events (OOM, permission denied, quota exceeded)
CRITICAL   # System failures (job termination, infrastructure down)
ALERT      # Immediate action required
EMERGENCY  # System unusable (highest)
```

From [Vertex AI Troubleshooting](https://cloud.google.com/vertex-ai/docs/general/troubleshooting) (accessed 2025-11-16):
- INFO: Normal training progress (epoch completion, checkpoint saves)
- WARNING: Recoverable issues (preemption events, retry attempts)
- ERROR: Job failures (OOM, quota limits, permission errors)
- CRITICAL: Infrastructure failures (node crashes, network partitions)

### 2.2 Structured Logging Best Practices

**Emit structured JSON logs in training scripts:**
```python
import json
import sys
import time

def log_structured(message, severity="INFO", **kwargs):
    """Emit structured logs compatible with Cloud Logging"""
    log_entry = {
        "severity": severity,
        "message": message,
        "timestamp": time.time(),
        **kwargs  # Additional structured fields
    }
    print(json.dumps(log_entry), file=sys.stdout, flush=True)

# Usage in training loop
log_structured(
    "Training metrics",
    severity="INFO",
    epoch=10,
    loss=0.234,
    accuracy=0.891,
    gpu_memory_used_gb=24.5,
    training_samples_per_sec=1250,
    learning_rate=1e-4
)
```

**Log severity by use case:**
```python
# INFO: Training progress
log_structured("Epoch 5/100 complete", severity="INFO",
               epoch=5, loss=0.456, val_accuracy=0.872)

# WARNING: Performance issues
if gpu_utilization < 50:
    log_structured("Low GPU utilization detected", severity="WARNING",
                   gpu_util=gpu_utilization, batch_size=current_batch_size)

# ERROR: Recoverable failures
try:
    checkpoint = load_checkpoint(gs_path)
except FileNotFoundError:
    log_structured("Checkpoint not found, using random initialization",
                   severity="ERROR", checkpoint_path=gs_path)

# CRITICAL: Job-terminating failures
except torch.cuda.OutOfMemoryError:
    log_structured("Out of GPU memory - job terminating",
                   severity="CRITICAL", allocated_gb=torch.cuda.memory_allocated()/1e9)
    raise
```

### 2.3 Filter Logs by Severity

**View only errors and above:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
severity >= ERROR
```

**Separate warnings from errors:**
```
# Warnings only (potential issues)
resource.type="aiplatform.googleapis.com/CustomJob"
severity = WARNING

# Errors and critical only (actual failures)
resource.type="aiplatform.googleapis.com/CustomJob"
severity >= ERROR
```

**Production monitoring query:**
```
# High-severity issues requiring immediate attention
resource.type="aiplatform.googleapis.com/Endpoint"
severity >= CRITICAL
timestamp >= "2025-11-16T00:00:00Z"
```

### 2.4 Custom Severity Thresholds

**Configure log retention by severity:**
```bash
# Retain ERROR and above for 365 days, INFO for 30 days
gcloud logging sinks create critical-logs-long-retention \
  bigquery.googleapis.com/projects/my-project/datasets/critical_logs \
  --log-filter='severity >= ERROR'
```

---

## Section 3: Common Error Patterns (~150 lines)

### 3.1 Out of Memory (OOM) Errors

**Symptoms:**
```
Exit code: 137 (OOM killed by kernel)
RuntimeError: CUDA out of memory
Container killed: OOMKilled
```

**Log patterns:**
```
resource.type="ml_job"
jsonPayload.message=~".*out of memory.*"
OR jsonPayload.message=~".*OOM.*"
OR jsonPayload.exit_code=137
```

From [Vertex AI Debugging Guide](https://cloud.google.com/vertex-ai/docs/general/troubleshooting) (accessed 2025-11-16):
- Exit code 137: Container OOM (killed by Linux kernel)
- CUDA OOM: GPU memory exhausted
- Common causes: Batch size too large, gradient accumulation disabled, memory leaks

**Diagnosis queries:**
```sql
-- Find jobs with OOM errors
SELECT
  resource.labels.job_id,
  jsonPayload.message,
  jsonPayload.gpu_memory_allocated_gb,
  jsonPayload.batch_size
FROM `my-project.vertex_logs.ml_job_*`
WHERE jsonPayload.message LIKE '%out of memory%'
  OR jsonPayload.exit_code = 137
ORDER BY timestamp DESC
LIMIT 50
```

**Root cause investigation:**
```python
# Add memory monitoring to training script
import torch
import psutil

def log_memory_snapshot():
    """Log current memory usage"""
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

            log_structured(
                f"GPU {i} memory snapshot",
                severity="INFO",
                gpu_id=i,
                allocated_gb=allocated,
                reserved_gb=reserved,
                max_allocated_gb=max_allocated
            )

    # CPU memory
    process = psutil.Process()
    cpu_mem_gb = process.memory_info().rss / 1024**3
    log_structured("CPU memory snapshot", severity="INFO",
                   cpu_memory_gb=cpu_mem_gb)

# Log every 100 steps
if step % 100 == 0:
    log_memory_snapshot()
```

**Common OOM solutions:**
```python
# 1. Gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# 2. Gradient accumulation (simulate larger batch with less memory)
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Mixed precision training (FP16 uses 50% less memory)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. Dynamic batch size reduction on OOM
def train_with_auto_batch_reduction(model, initial_batch_size=64):
    batch_size = initial_batch_size

    while batch_size >= 1:
        try:
            train_loader = DataLoader(dataset, batch_size=batch_size)
            for batch in train_loader:
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()
            break  # Success
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                log_structured("OOM - reducing batch size",
                               severity="WARNING", new_batch_size=batch_size)
            else:
                raise
```

### 3.2 Quota Exceeded Errors

**Symptoms:**
```
Error 429: Quota 'NVIDIA_A100_GPUS' exceeded in region 'us-central1'
ResourceExhausted: Requested resources exceed project quota
RESOURCE_EXHAUSTED: The request exceeded one or more resource quotas
```

**Log patterns:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
jsonPayload.message=~".*quota.*exceeded.*"
OR protoPayload.status.code=429
OR protoPayload.status.message=~".*RESOURCE_EXHAUSTED.*"
```

**Check current quotas:**
```bash
# View all Vertex AI quotas
gcloud compute project-info describe --project=my-project \
  --format="table(quotas.metric, quotas.limit, quotas.usage)" \
  | grep -i "gpu\|tpu\|vertex"

# Check specific GPU quota
gcloud compute regions describe us-central1 \
  --format="value(quotas.filter(metric~'.*A100.*'))"
```

**Request quota increase workflow:**
```bash
# 1. Navigate to Console: IAM & Admin > Quotas
# 2. Filter: Service = "Vertex AI API", Metric = "NVIDIA_A100_GPUS"
# 3. Select quota > EDIT QUOTAS
# 4. Request increase (provide business justification)
# 5. Approval time: 2-5 business days

# Temporary workaround: Use multiple regions
gcloud ai custom-jobs create \
  --region=us-west1 \
  --display-name=training-west1 \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1
```

**Common quota limits:**
```
NVIDIA_A100_GPUS (per region):        Default = 0-4, Max = 64-128
NVIDIA_V100_GPUS (per region):        Default = 4-8, Max = 128
Custom_Job_Concurrent_Runs:          Default = 100, Max = 500
Endpoint_Prediction_Requests_Per_Min: Default = 60, Max = 6000
Training_CPUs (per region):           Default = 1000, Max = 10000
```

### 3.3 Permission Denied Errors

**Symptoms:**
```
PermissionDenied: 403 Access denied to gs://my-bucket/data/
403 Forbidden: The caller does not have permission
IAM_PERMISSION_DENIED: Missing required permission
```

**Log patterns:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
protoPayload.status.code=403
OR jsonPayload.message=~".*permission.*denied.*"
OR jsonPayload.message=~".*403.*"
```

**Required IAM roles for Vertex AI:**
```bash
# Vertex AI service account needs:
roles/aiplatform.user                 # Submit Custom Jobs
roles/storage.objectViewer            # Read from GCS
roles/storage.objectCreator           # Write checkpoints to GCS
roles/artifactregistry.reader         # Pull container images
roles/logging.logWriter               # Write Cloud Logging logs

# Grant roles to Vertex AI service account
SERVICE_ACCOUNT="vertex-sa@my-project.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding my-project \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

gcloud artifacts repositories add-iam-policy-binding my-repo \
  --location=us-central1 \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.reader"
```

**Permission debugging decision tree:**
```
Permission Error Occurred?
├─ Can't pull container image
│  └─ Grant roles/artifactregistry.reader on Artifact Registry repo
│
├─ Can't read training data from GCS
│  └─ Grant roles/storage.objectViewer on GCS bucket
│
├─ Can't write checkpoints to GCS
│  └─ Grant roles/storage.objectCreator on GCS bucket
│
├─ Can't submit Custom Job
│  └─ Grant roles/aiplatform.user at project level
│
└─ Can't write logs
   └─ Grant roles/logging.logWriter at project level
```

**Verify service account permissions:**
```bash
# List effective IAM policy for service account
gcloud projects get-iam-policy my-project \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:vertex-sa@my-project.iam.gserviceaccount.com" \
  --format="table(bindings.role)"

# Test GCS bucket access
gsutil iam get gs://my-bucket | grep vertex-sa
```

### 3.4 Network Timeout Errors

**Symptoms:**
```
ConnectionError: Failed to connect to storage.googleapis.com
Timeout: Request to GCS timed out after 60s
UNAVAILABLE: Connection reset by peer
```

**Log patterns:**
```
resource.type="aiplatform.googleapis.com/CustomJob"
jsonPayload.message=~".*timeout.*"
OR jsonPayload.message=~".*connection.*failed.*"
OR protoPayload.status.code=503
```

**Network configuration troubleshooting:**
```bash
# 1. Check VPC firewall rules (allow egress to GCS)
gcloud compute firewall-rules list \
  --filter="direction=EGRESS AND targetTags~'vertex-ai.*'" \
  --format="table(name, allowed, destinationRanges)"

# 2. Verify Private Google Access (for private VPCs)
gcloud compute networks subnets describe my-subnet \
  --region=us-central1 \
  --format="get(privateIpGoogleAccess)"

# Enable if false
gcloud compute networks subnets update my-subnet \
  --region=us-central1 \
  --enable-private-ip-google-access

# 3. Test connectivity from Cloud Shell (simulates Vertex AI network)
curl -I https://storage.googleapis.com
```

**Custom Job with VPC network:**
```python
from google.cloud import aiplatform

job = aiplatform.CustomJob(
    display_name="training-with-vpc",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "container_spec": {"image_uri": "gcr.io/my-project/training:latest"},
    }],
    # Attach to VPC network
    network="projects/my-project/global/networks/my-vpc",
)
```

**Common network timeout solutions:**
```python
# 1. Increase GCS client timeout
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket("my-bucket")
blob = bucket.blob("large-file.tar.gz")

# Increase timeout to 600 seconds (10 minutes)
blob.download_to_filename("local-file.tar.gz", timeout=600)

# 2. Retry logic for transient network failures
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def download_with_retry(gs_path, local_path):
    """Download with exponential backoff retry"""
    bucket_name, blob_name = parse_gs_path(gs_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path, timeout=300)
    log_structured("Download successful", severity="INFO", gs_path=gs_path)

# 3. Use gsutil for large file transfers (better resumption)
import subprocess

subprocess.run([
    "gsutil", "-m", "cp", "-r",
    "gs://my-bucket/large-dataset/",
    "/mnt/data/",
], check=True, timeout=3600)
```

---

## Section 4: Cloud Trace for Request Latency (~100 lines)

### 4.1 Cloud Trace Overview

Cloud Trace provides distributed tracing for Vertex AI prediction requests, revealing latency bottlenecks across model inference, preprocessing, and network overhead.

From [Cloud Trace Quickstart](https://docs.cloud.google.com/trace/docs/trace-app-latency) (accessed 2025-11-16):
- Tracks request latency end-to-end
- Identifies bottlenecks in multi-service architectures
- Automatic integration with Vertex AI Endpoints
- 30-day trace retention

**Trace components:**
```
Prediction Request Trace:
├─ Network ingress (Load Balancer → Endpoint)
├─ Authentication/authorization
├─ Preprocessing (tokenization, image resize)
├─ Model inference (forward pass)
├─ Postprocessing (argmax, formatting)
└─ Network egress (Endpoint → Client)
```

### 4.2 Enable Tracing for Vertex AI Endpoints

**Automatic tracing (enabled by default):**
```python
from google.cloud import aiplatform

# Deploy model with tracing enabled
endpoint = aiplatform.Endpoint.create(display_name="my-endpoint")
model = aiplatform.Model.upload(
    display_name="my-model",
    serving_container_image_uri="gcr.io/my-project/serving:latest",
)

endpoint.deploy(
    model=model,
    deployed_model_display_name="v1",
    machine_type="n1-standard-4",
    # Tracing is automatically enabled for all predictions
)
```

**View traces in Cloud Console:**
```
Navigation: Cloud Console → Trace → Trace Explorer
Filter by: Service = "Vertex AI Endpoint"
          Endpoint ID = "1234567890"
          Latency > 100ms
```

### 4.3 Analyze Latency Patterns

**Trace list query (gcloud CLI):**
```bash
# View recent high-latency predictions
gcloud trace list \
  --filter="endTime >= '2025-11-16T00:00:00Z'" \
  --limit=100 \
  --format="table(traceId, spanName, startTime, duration)"
```

**Common latency bottlenecks:**
```
Preprocessing slowness:
- Symptom: 80%+ time in preprocessing span
- Cause: CPU-bound tokenization, image resize on CPU
- Solution: Move preprocessing to GPU, optimize tokenizer

Model inference slowness:
- Symptom: 90%+ time in model forward pass
- Cause: Model too large, not optimized
- Solution: TensorRT optimization, quantization, smaller model

Postprocessing slowness:
- Symptom: Significant time in argmax/formatting
- Cause: Large vocabulary, CPU argmax
- Solution: GPU postprocessing, beam search optimization

Network latency:
- Symptom: High ingress/egress time
- Cause: Cross-region requests, large payloads
- Solution: Multi-region endpoints, payload compression
```

**Latency percentiles from Cloud Monitoring:**
```sql
# BigQuery query on exported traces
SELECT
  APPROX_QUANTILES(duration_ms, 100)[OFFSET(50)] AS p50_latency_ms,
  APPROX_QUANTILES(duration_ms, 100)[OFFSET(95)] AS p95_latency_ms,
  APPROX_QUANTILES(duration_ms, 100)[OFFSET(99)] AS p99_latency_ms,
  COUNT(*) AS request_count
FROM (
  SELECT
    TIMESTAMP_DIFF(endTime, startTime, MILLISECOND) AS duration_ms
  FROM `my-project.cloud_trace.traces`
  WHERE DATE(startTime) = CURRENT_DATE()
    AND spanName LIKE '%Predict%'
)
```

### 4.4 Custom Trace Spans

**Add custom spans to training code:**
```python
from google.cloud import trace_v2
from google.cloud.trace_v2 import TraceServiceClient

tracer = trace_v2.Client()

def train_with_tracing():
    """Training loop with custom trace spans"""

    # Create trace
    trace_id = tracer.trace()

    # Data loading span
    with tracer.span(name="data_loading"):
        train_data = load_dataset("gs://my-bucket/train/")

    # Model training span
    with tracer.span(name="model_training"):
        for epoch in range(num_epochs):
            # Epoch-level span
            with tracer.span(name=f"epoch_{epoch}"):
                train_one_epoch(model, train_data)

    # Checkpoint save span
    with tracer.span(name="checkpoint_save"):
        save_checkpoint(model, "gs://my-bucket/checkpoints/")
```

**Trace-based performance alerts:**
```bash
# Create Cloud Monitoring alert for high P95 latency
gcloud alpha monitoring policies create \
  --notification-channels=projects/my-project/notificationChannels/123 \
  --display-name="High P95 Prediction Latency" \
  --condition-display-name="P95 > 500ms" \
  --condition-threshold-value=500 \
  --condition-threshold-duration=300s \
  --condition-metric="serviceruntime.googleapis.com/api/request_latencies" \
  --condition-metric-filter='service_name="aiplatform.googleapis.com"'
```

---

## Section 5: Cloud Profiler (CPU, Memory, Heap) (~100 lines)

### 5.1 Cloud Profiler for Training Jobs

Cloud Profiler provides continuous, low-overhead profiling of CPU, memory, and heap usage for Vertex AI Custom Jobs.

From [Cloud Profiler Overview](https://docs.cloud.google.com/profiler/docs/about-profiler) (accessed 2025-11-16):
- CPU profiling: Identify compute bottlenecks
- Heap profiling: Memory allocation patterns
- Wall-clock profiling: Overall time distribution
- <1% performance overhead (production-safe)

**Enable profiler in training script:**
```python
import google.cloud.profiler

# Initialize profiler at script start
try:
    google.cloud.profiler.start(
        service='vertex-training-arr-coc',
        service_version='v1.2.0',
        verbose=3,  # Debug logging level
    )
    print("Cloud Profiler started successfully")
except Exception as e:
    print(f"Failed to start profiler: {e}")
    # Continue training even if profiler fails

# Your training code runs as normal
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)
```

**Dockerfile additions:**
```dockerfile
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-1.py310

# Install Cloud Profiler
RUN pip install --no-cache-dir google-cloud-profiler

# Copy training script
COPY train.py /app/
WORKDIR /app

ENTRYPOINT ["python", "train.py"]
```

### 5.2 Analyze Profiler Results

**Navigate to profiler:**
```
Cloud Console → Profiler → Select service: vertex-training-arr-coc
```

**Flame graph analysis:**
```
CPU Flame Graph:
- Width: Total CPU time (wider = more time)
- Height: Call stack depth
- Color: Function category

Common bottlenecks:
- data_loading: 40%+ → Increase num_workers, prefetch
- model_forward: 60%+ → Expected (model inference)
- optimizer_step: 15%+ → Consider gradient accumulation
- checkpoint_save: 10%+ → Async checkpoint saving
```

**Memory profiling insights:**
```python
# Heap profiling reveals:
1. Model parameters: 12 GB (expected)
2. Optimizer state: 24 GB (2x model size for Adam)
3. Activations: 8 GB (depends on batch size)
4. Data loading: 4 GB (prefetch buffers)

# If total > GPU memory (80 GB A100):
# - Reduce batch size
# - Enable gradient checkpointing
# - Use DeepSpeed ZeRO
```

**Wall-clock profile example:**
```
Total training time breakdown:
├─ Model forward pass: 45% (expected)
├─ Backward pass: 30% (expected)
├─ Optimizer step: 10% (expected)
├─ Data loading: 12% (HIGH - should be <5%)
└─ Checkpoint saving: 3% (acceptable)

Optimization: Increase DataLoader num_workers from 4 to 8
```

### 5.3 GPU Profiling with TensorBoard

**TensorBoard Profiler integration:**
```python
import torch.profiler

def train_with_profiling(model, train_loader):
    """Training loop with TensorBoard profiling"""

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,    # Warmup steps
            warmup=1,  # Warmup steps
            active=3,  # Profile these steps
            repeat=2   # Repeat cycle
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, batch in enumerate(train_loader):
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            prof.step()  # Signal profiler
```

**Upload TensorBoard logs to Vertex AI:**
```bash
# Upload profiling logs to GCS
gsutil -m cp -r /tmp/logs gs://my-bucket/tensorboard-logs/

# View in Vertex AI TensorBoard
gcloud ai tensorboards create \
  --display-name=arr-coc-profiling \
  --region=us-central1

gcloud ai tensorboard-experiments create profiling-v1 \
  --tensorboard=projects/my-project/locations/us-central1/tensorboards/123 \
  --region=us-central1

# Point TensorBoard to GCS logs
```

### 5.4 Profiling Best Practices

**Production profiling workflow:**
```python
# 1. Enable profiling for 1% of training jobs (sampling)
import random

enable_profiling = random.random() < 0.01  # 1% of jobs

if enable_profiling:
    google.cloud.profiler.start(
        service='vertex-training',
        service_version=VERSION,
    )

# 2. Profile only specific epochs (not entire training)
if epoch % 10 == 0:  # Profile every 10th epoch
    with torch.profiler.profile(...):
        train_one_epoch(model, train_loader)

# 3. Export profiling data asynchronously
def save_profile_async(prof, gs_path):
    """Save profiling data without blocking training"""
    prof.export_chrome_trace(f"/tmp/trace_{epoch}.json")
    # Upload to GCS in background thread
    threading.Thread(
        target=lambda: upload_to_gcs(f"/tmp/trace_{epoch}.json", gs_path),
        daemon=True
    ).start()
```

**Common profiling anti-patterns:**
```python
# ❌ BAD: Profile entire training (overhead accumulates)
with torch.profiler.profile(...):
    for epoch in range(100):  # 100 epochs!
        train_one_epoch()

# ✅ GOOD: Profile only a few steps
with torch.profiler.profile(...):
    for step in range(10):  # Just 10 steps
        train_one_step()

# ❌ BAD: High-frequency profiling (5% overhead)
for step in range(1000):
    with torch.profiler.profile(...):
        train_one_step()

# ✅ GOOD: Sample-based profiling (<1% overhead)
for step in range(1000):
    if step % 100 == 0:
        with torch.profiler.profile(...):
            train_one_step()
```

---

## Section 6: Cost Spike Investigation (~120 lines)

### 6.1 Enable Detailed Billing Export

**Export billing data to BigQuery:**
```bash
# Navigate to: Cloud Console → Billing → Billing export
# Enable: "Detailed usage cost" export
# Select: Project and BigQuery dataset

# Alternative: gcloud CLI
gcloud billing accounts describe 012345-ABCDEF-67890 \
  --format="value(displayName)"

# Billing data appears in BigQuery dataset:
# `my-project.billing_export.gcp_billing_export_resource_v1_<BILLING_ID>`
```

From [Detailed Billing Export Structure](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) (accessed 2025-11-16):
- Detailed export: Resource-level cost data with labels
- Updates: Every 24 hours (daily batch)
- Schema: Includes SKU, usage amount, cost, labels, tags
- Retention: Standard BigQuery table retention

### 6.2 Vertex AI Cost Analysis Queries

**Daily Vertex AI spending:**
```sql
SELECT
  DATE(usage_start_time) AS usage_date,
  service.description AS service_name,
  sku.description AS sku_description,
  SUM(cost) AS daily_cost_usd,
  SUM(usage.amount) AS usage_amount,
  usage.unit AS usage_unit
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
WHERE service.description LIKE '%Vertex AI%'
  AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY usage_date, service_name, sku_description, usage_unit
ORDER BY usage_date DESC, daily_cost_usd DESC
```

**Top 10 most expensive Vertex AI SKUs:**
```sql
SELECT
  sku.description AS sku_name,
  SUM(cost) AS total_cost_usd,
  SUM(usage.amount) AS total_usage,
  usage.unit AS unit,
  COUNT(DISTINCT project.id) AS project_count,
  ARRAY_AGG(DISTINCT project.id LIMIT 5) AS sample_projects
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
WHERE service.description = 'Vertex AI'
  AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY sku_name, unit
ORDER BY total_cost_usd DESC
LIMIT 10
```

**Identify cost spikes (day-over-day changes):**
```sql
WITH daily_costs AS (
  SELECT
    DATE(usage_start_time) AS date,
    sku.description AS sku,
    SUM(cost) AS cost_usd
  FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
  WHERE service.description = 'Vertex AI'
    AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY date, sku
),
cost_changes AS (
  SELECT
    date,
    sku,
    cost_usd,
    LAG(cost_usd) OVER (PARTITION BY sku ORDER BY date) AS prev_day_cost,
    cost_usd - LAG(cost_usd) OVER (PARTITION BY sku ORDER BY date) AS cost_change,
    SAFE_DIVIDE(
      cost_usd - LAG(cost_usd) OVER (PARTITION BY sku ORDER BY date),
      LAG(cost_usd) OVER (PARTITION BY sku ORDER BY date)
    ) * 100 AS pct_change
  FROM daily_costs
)
SELECT
  date,
  sku,
  cost_usd,
  prev_day_cost,
  cost_change,
  ROUND(pct_change, 2) AS pct_change
FROM cost_changes
WHERE ABS(pct_change) > 50  -- Spikes >50% change
  AND cost_change > 10  -- Absolute change >$10
ORDER BY date DESC, ABS(cost_change) DESC
```

### 6.3 Resource-Level Cost Attribution

**Custom Job costs by job_id:**
```sql
SELECT
  labels.value AS job_id,
  DATE(usage_start_time) AS date,
  sku.description AS sku,
  SUM(cost) AS cost_usd,
  SUM(usage.amount) AS usage_amount,
  usage.unit AS unit
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`,
UNNEST(labels) AS labels
WHERE service.description = 'Vertex AI'
  AND labels.key = 'goog-aiplatform-custom-job-id'
  AND DATE(usage_start_time) >= CURRENT_DATE() - 7
GROUP BY job_id, date, sku, unit
ORDER BY cost_usd DESC
LIMIT 100
```

**Endpoint prediction costs by endpoint_id:**
```sql
SELECT
  labels.value AS endpoint_id,
  DATE(usage_start_time) AS date,
  SUM(cost) AS cost_usd,
  SUM(CASE WHEN sku.description LIKE '%prediction%' THEN cost ELSE 0 END) AS prediction_cost,
  SUM(CASE WHEN sku.description LIKE '%node%' THEN cost ELSE 0 END) AS node_cost,
  COUNT(DISTINCT usage_start_time) AS billing_records
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`,
UNNEST(labels) AS labels
WHERE service.description = 'Vertex AI'
  AND labels.key = 'goog-aiplatform-endpoint-id'
  AND DATE(usage_start_time) >= CURRENT_DATE() - 30
GROUP BY endpoint_id, date
ORDER BY cost_usd DESC
```

**Cost by custom labels (team, project, environment):**
```sql
SELECT
  team_label.value AS team,
  env_label.value AS environment,
  DATE(usage_start_time) AS date,
  SUM(cost) AS cost_usd
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`,
UNNEST(labels) AS team_label,
UNNEST(labels) AS env_label
WHERE service.description = 'Vertex AI'
  AND team_label.key = 'team'
  AND env_label.key = 'environment'
  AND DATE(usage_start_time) >= CURRENT_DATE() - 7
GROUP BY team, environment, date
ORDER BY date DESC, cost_usd DESC
```

### 6.4 Cost Spike Root Cause Analysis

**Investigation workflow:**
```sql
-- Step 1: Identify spike date and SKU
-- (Use day-over-day query from Section 6.2)

-- Step 2: Find all resources contributing to spike
SELECT
  resource.name AS resource_name,
  labels.value AS job_id,
  SUM(cost) AS resource_cost_usd,
  SUM(usage.amount) AS usage_amount,
  usage.unit AS unit,
  MIN(usage_start_time) AS first_usage,
  MAX(usage_end_time) AS last_usage
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`,
UNNEST(labels) AS labels
WHERE service.description = 'Vertex AI'
  AND sku.description = 'A100 GPU running in us-central1'  -- Specific SKU from spike
  AND DATE(usage_start_time) = '2025-11-15'  -- Spike date
  AND labels.key = 'goog-aiplatform-custom-job-id'
GROUP BY resource_name, job_id, unit
ORDER BY resource_cost_usd DESC
LIMIT 20
```

**Common cost spike causes:**
```
1. Long-running training job (forgot to stop):
   - Symptom: Single job_id with >24h runtime
   - Cost: $50-500+ per day (GPU idle time)
   - Solution: Implement training timeout, auto-shutdown

2. Endpoint deployed but not used (idle nodes):
   - Symptom: Endpoint node cost with 0 predictions
   - Cost: $200-2000+ per month (idle A100 nodes)
   - Solution: Undeploy endpoints, use autoscaling

3. Quota increase + parallel jobs:
   - Symptom: 10+ concurrent jobs after quota increase
   - Cost: $1000+ spike (8x A100 GPUs × 8 hours)
   - Solution: Job queue management, cost alerts

4. Data preprocessing inefficiency:
   - Symptom: High "Dataflow" or "Compute Engine" costs
   - Cost: $100-1000+ (oversized preprocessing VMs)
   - Solution: Right-size preprocessing, use preemptible

5. Model serving over-provisioning:
   - Symptom: 10+ replicas but <1000 requests/day
   - Cost: $500+ per month (unused capacity)
   - Solution: Reduce replicas, use autoscaling
```

**Set up cost alerts:**
```bash
# Create budget alert for Vertex AI spending
gcloud billing budgets create \
  --billing-account=012345-ABCDEF-67890 \
  --display-name="Vertex AI Monthly Budget" \
  --budget-amount=5000 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=75 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100 \
  --all-updates-rule-pubsub-topic=projects/my-project/topics/billing-alerts \
  --filter-services=services/0C6FDA83-7FD2-4BAD-8D17-A65DBAAF2DA0  # Vertex AI service ID
```

**Daily cost monitoring query:**
```sql
-- Save as scheduled query (runs daily at 9 AM)
CREATE OR REPLACE TABLE `my-project.cost_monitoring.vertex_ai_daily_costs` AS
SELECT
  CURRENT_DATE() AS report_date,
  service.description AS service,
  sku.description AS sku,
  SUM(cost) AS yesterday_cost_usd,
  SUM(usage.amount) AS usage_amount,
  usage.unit AS unit
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
WHERE service.description LIKE '%Vertex AI%'
  AND DATE(usage_start_time) = CURRENT_DATE() - 1
GROUP BY service, sku, unit
ORDER BY yesterday_cost_usd DESC
```

---

## Section 7: arr-coc-0-1 Debugging Workflows (~110 lines)

### 7.1 arr-coc-0-1 Training Job Debugging

**Common arr-coc-0-1 training issues:**
```
1. Visual texture preprocessing OOM
   - Cause: 13-channel tensor (RGB + LAB + Sobel + spatial) too large
   - Log pattern: "CUDA out of memory" during texture computation
   - Solution: Process in smaller batches, use FP16

2. Dynamic LOD allocation failure
   - Cause: Token budget calculation error (K=200 patches, 64-400 tokens each)
   - Log pattern: "AssertionError: Total tokens exceed budget"
   - Solution: Validate budget before allocation

3. Opponent processing NaN gradients
   - Cause: Tension balancing produces extreme values
   - Log pattern: "NaN detected in tension_scores"
   - Solution: Gradient clipping, tension score clamping
```

**arr-coc-0-1 structured logging:**
```python
# training/train.py (arr-coc-0-1 project)

def log_arr_coc_metrics(epoch, step, metrics):
    """Log ARR-COC-specific training metrics"""
    log_structured(
        "ARR-COC training metrics",
        severity="INFO",
        epoch=epoch,
        step=step,
        # Relevance realization metrics
        avg_relevance_score=metrics['relevance_score'],
        avg_token_budget=metrics['avg_tokens_per_patch'],
        max_token_budget=metrics['max_tokens_per_patch'],
        min_token_budget=metrics['min_tokens_per_patch'],
        # Opponent processing metrics
        tension_compress_particularize=metrics['tension_cp'],
        tension_exploit_explore=metrics['tension_ee'],
        tension_focus_diversify=metrics['tension_fd'],
        # Training metrics
        loss=metrics['loss'],
        accuracy=metrics['accuracy'],
        # Resource metrics
        gpu_memory_used_gb=torch.cuda.memory_allocated() / 1e9,
        batch_size=metrics['batch_size'],
    )

# Usage in training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # Forward pass through ARR-COC pipeline
        output = model(batch)  # knowing → balancing → attending → realizing

        # Collect metrics
        metrics = {
            'relevance_score': output['relevance_scores'].mean().item(),
            'avg_tokens_per_patch': output['token_budgets'].float().mean().item(),
            'max_tokens_per_patch': output['token_budgets'].max().item(),
            'min_tokens_per_patch': output['token_budgets'].min().item(),
            'tension_cp': output['tensions']['compress_particularize'].item(),
            'tension_ee': output['tensions']['exploit_explore'].item(),
            'tension_fd': output['tensions']['focus_diversify'].item(),
            'loss': loss.item(),
            'accuracy': accuracy,
            'batch_size': batch['images'].size(0),
        }

        # Log every 50 steps
        if step % 50 == 0:
            log_arr_coc_metrics(epoch, step, metrics)
```

**Query arr-coc-0-1 training logs:**
```sql
-- Find training runs with NaN gradients
SELECT
  timestamp,
  jsonPayload.epoch,
  jsonPayload.step,
  jsonPayload.tension_compress_particularize,
  jsonPayload.tension_exploit_explore,
  jsonPayload.tension_focus_diversify
FROM `my-project.vertex_logs.ml_job_*`
WHERE jsonPayload.message LIKE '%ARR-COC training metrics%'
  AND (
    jsonPayload.tension_compress_particularize = 'NaN' OR
    jsonPayload.tension_exploit_explore = 'NaN' OR
    jsonPayload.tension_focus_diversify = 'NaN'
  )
ORDER BY timestamp DESC
LIMIT 100
```

### 7.2 arr-coc-0-1 Inference Debugging

**Prediction latency analysis:**
```python
# arr_coc/model.py

def predict_with_tracing(self, image, query):
    """Inference with detailed latency tracing"""

    import time
    timings = {}

    # Texture preprocessing
    start = time.time()
    texture_array = self.texture_processor(image)  # 13 channels
    timings['texture_preprocessing'] = time.time() - start

    # Knowing (relevance scoring)
    start = time.time()
    relevance_scores = self.knowing_module(texture_array, query)
    timings['knowing'] = time.time() - start

    # Balancing (opponent processing)
    start = time.time()
    tensions = self.balancing_module(relevance_scores)
    timings['balancing'] = time.time() - start

    # Attending (LOD allocation)
    start = time.time()
    token_budgets = self.attending_module(relevance_scores, tensions)
    timings['attending'] = time.time() - start

    # Realizing (compression + forward pass)
    start = time.time()
    output = self.realizing_module(texture_array, token_budgets, query)
    timings['realizing'] = time.time() - start

    # Log detailed timings
    log_structured(
        "ARR-COC inference latency",
        severity="INFO",
        total_latency_ms=sum(timings.values()) * 1000,
        texture_preprocessing_ms=timings['texture_preprocessing'] * 1000,
        knowing_ms=timings['knowing'] * 1000,
        balancing_ms=timings['balancing'] * 1000,
        attending_ms=timings['attending'] * 1000,
        realizing_ms=timings['realizing'] * 1000,
        num_patches=len(token_budgets),
        avg_tokens_per_patch=token_budgets.float().mean().item(),
    )

    return output
```

**Identify inference bottlenecks:**
```sql
-- Find slow ARR-COC inference requests
SELECT
  timestamp,
  jsonPayload.total_latency_ms,
  jsonPayload.texture_preprocessing_ms,
  jsonPayload.knowing_ms,
  jsonPayload.balancing_ms,
  jsonPayload.attending_ms,
  jsonPayload.realizing_ms,
  jsonPayload.num_patches
FROM `my-project.vertex_logs.aiplatform_googleapis_com_Endpoint_*`
WHERE jsonPayload.message = 'ARR-COC inference latency'
  AND jsonPayload.total_latency_ms > 500  -- Slow requests (>500ms)
ORDER BY jsonPayload.total_latency_ms DESC
LIMIT 50
```

**Optimization based on profiling:**
```python
# Common arr-coc-0-1 bottlenecks and solutions:

# 1. Texture preprocessing slow (50%+ of latency)
# Solution: Move to GPU, cache LAB conversion
texture_processor = TextureProcessor(device='cuda')
with torch.cuda.amp.autocast():  # FP16 preprocessing
    texture_array = texture_processor(image)

# 2. Opponent processing complex (graph traversal)
# Solution: Vectorize tension calculations
tensions = compute_tensions_vectorized(relevance_scores)  # GPU batch ops

# 3. LOD allocation sequential (per-patch)
# Solution: Parallel token budget allocation
token_budgets = allocate_budgets_parallel(relevance_scores, K=200)

# 4. Variable LOD increases batch complexity
# Solution: Bucket patches by token count (64, 128, 256, 400)
bucketed_patches = bucket_by_token_count(patches, token_budgets)
for bucket in bucketed_patches:
    process_batch(bucket)  # Homogeneous batch sizes
```

### 7.3 arr-coc-0-1 Cloud Build Monitoring

**Monitor PyTorch compilation (2-4 hour builds):**
```bash
# Get latest Cloud Build ID
BUILD_ID=$(gcloud builds list --region=us-west2 --limit=1 --format="value(id)")

# Stream live build logs
gcloud builds log $BUILD_ID --region=us-west2 --stream

# Check compilation progress (PyTorch shows [7430/7517] style output)
gcloud builds log $BUILD_ID --region=us-west2 | tail -20
```

**Parse build logs for errors:**
```bash
# Extract errors from Cloud Build logs
gcloud builds log $BUILD_ID --region=us-west2 \
  | grep -i "error\|failed\|fatal" \
  | tail -50

# Check build timeout (default 1 hour, extended to 4 hours for PyTorch)
gcloud builds describe $BUILD_ID --region=us-west2 \
  --format="value(timeout)"
```

### 7.4 arr-coc-0-1 Cost Tracking

**Track arr-coc-0-1 training costs by run:**
```sql
SELECT
  labels.value AS wandb_run_id,
  DATE(usage_start_time) AS date,
  SUM(cost) AS total_cost_usd,
  SUM(CASE WHEN sku.description LIKE '%A100%' THEN cost ELSE 0 END) AS gpu_cost,
  SUM(CASE WHEN sku.description LIKE '%Storage%' THEN cost ELSE 0 END) AS storage_cost,
  SUM(usage.amount) AS gpu_hours
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`,
UNNEST(labels) AS labels
WHERE service.description = 'Vertex AI'
  AND labels.key = 'wandb_run_id'  # Custom label added to training jobs
  AND DATE(usage_start_time) >= CURRENT_DATE() - 30
GROUP BY wandb_run_id, date
ORDER BY total_cost_usd DESC
```

**Monthly arr-coc-0-1 project costs:**
```sql
SELECT
  FORMAT_DATE('%Y-%m', DATE(usage_start_time)) AS month,
  SUM(cost) AS monthly_cost_usd,
  SUM(CASE WHEN sku.description LIKE '%training%' THEN cost ELSE 0 END) AS training_cost,
  SUM(CASE WHEN sku.description LIKE '%prediction%' THEN cost ELSE 0 END) AS serving_cost,
  SUM(CASE WHEN sku.description LIKE '%storage%' THEN cost ELSE 0 END) AS storage_cost
FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
WHERE project.id = 'arr-coc-0-1'
  AND service.description = 'Vertex AI'
  AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY month
ORDER BY month DESC
```

**Cost anomaly detection for arr-coc-0-1:**
```sql
-- Alert if daily cost >2x weekly average
WITH weekly_avg AS (
  SELECT
    AVG(daily_cost) AS avg_cost
  FROM (
    SELECT
      DATE(usage_start_time) AS date,
      SUM(cost) AS daily_cost
    FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
    WHERE project.id = 'arr-coc-0-1'
      AND service.description = 'Vertex AI'
      AND DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    GROUP BY date
  )
),
yesterday AS (
  SELECT
    SUM(cost) AS yesterday_cost
  FROM `my-project.billing_export.gcp_billing_export_resource_v1_*`
  WHERE project.id = 'arr-coc-0-1'
    AND service.description = 'Vertex AI'
    AND DATE(usage_start_time) = CURRENT_DATE() - 1
)
SELECT
  yesterday_cost,
  avg_cost AS weekly_avg_cost,
  yesterday_cost / avg_cost AS cost_multiplier,
  CASE
    WHEN yesterday_cost > 2 * avg_cost THEN '🚨 ALERT: Cost spike detected'
    ELSE '✅ Cost within normal range'
  END AS status
FROM yesterday, weekly_avg
```

---

## Sources

**Google Cloud Documentation:**
- [Vertex AI Audit Logging](https://docs.cloud.google.com/vertex-ai/docs/general/audit-logging) - accessed 2025-11-16
- [Vertex AI Troubleshooting](https://cloud.google.com/vertex-ai/docs/general/troubleshooting) - accessed 2025-11-16
- [Cloud Trace Quickstart](https://docs.cloud.google.com/trace/docs/trace-app-latency) - accessed 2025-11-16
- [Cloud Profiler Overview](https://docs.cloud.google.com/profiler/docs/about-profiler) - accessed 2025-11-16
- [Detailed Billing Export Structure](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) - accessed 2025-11-16

**Source Documents:**
- [practical-implementation/36-vertex-ai-debugging.md](../karpathy/practical-implementation/36-vertex-ai-debugging.md) - Vertex AI training job debugging, Cloud Logging filters, common errors, interactive shell, Cloud Profiler

**Web Research:**
- Search: "Cloud Logging Vertex AI filters resource.type aiplatform 2024" (accessed 2025-11-16)
- Search: "Vertex AI common errors troubleshooting OOM quota permission 2024" (accessed 2025-11-16)
- Search: "Cloud Trace request latency analysis GCP 2024" (accessed 2025-11-16)
- Search: "Cloud Profiler GPU CPU memory profiling GCP 2024" (accessed 2025-11-16)
- Search: "GCP detailed billing export cost spike investigation BigQuery 2024" (accessed 2025-11-16)
- Search: "Vertex AI cost analysis detailed billing SKU investigation 2024" (accessed 2025-11-16)

**Additional References:**
- Cloud Monitoring integration for Vertex AI metrics
- TensorBoard Profiler for GPU kernel analysis
- BigQuery scheduled queries for daily cost monitoring
- arr-coc-0-1 project debugging patterns from production deployments
