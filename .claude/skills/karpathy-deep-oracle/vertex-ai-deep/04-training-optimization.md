# Vertex AI Training Optimization

## Overview

Training optimization on Vertex AI focuses on maximizing GPU utilization, minimizing training time, and reducing costs while maintaining model quality. This guide covers GPU utilization patterns, profiling tools, multi-GPU strategies, I/O optimization, and memory management techniques specific to Vertex AI custom training jobs.

From [Optimizing Custom Job Runs on Vertex AI](https://medium.com/@haely.shah/optimizing-custom-job-runs-on-vertex-ai-a-comprehensive-guide-628c496e5eba) (accessed 2025-02-03):
> Running machine learning (ML) jobs efficiently at scale is a key challenge, especially when leveraging powerful platforms like Google Cloud's Vertex AI. Optimizing custom job runs on Vertex AI requires a balance between fine-tuning hyperparameters, managing Docker containers, handling data efficiently, and leveraging parallelization strategies.

## GPU Utilization Optimization

### Understanding GPU Utilization

GPU utilization measures how effectively your training code uses available GPU compute resources. Target: **>80% GPU utilization** for efficient training.

**Key GPU Metrics:**
- **SM (Streaming Multiprocessor) Utilization**: Percentage of time GPU cores are active
- **Memory Bandwidth**: Rate of data transfer between GPU memory and compute units
- **Tensor Core Utilization**: Usage of specialized hardware for mixed-precision operations
- **GPU Memory Usage**: Fraction of available VRAM occupied by model and data

### Common GPU Utilization Bottlenecks

**1. Data Loading Bottleneck**
- **Symptom**: GPU utilization drops during data loading, spikes during computation
- **Cause**: CPU can't feed data fast enough to keep GPU busy
- **Solution**: Increase DataLoader workers, use prefetching, cache data in memory

**2. Small Batch Sizes**
- **Symptom**: Low GPU utilization (<50%) even during training
- **Cause**: Insufficient parallelism to saturate GPU cores
- **Solution**: Increase batch size (use gradient accumulation if memory-limited)

**3. CPU-Heavy Operations**
- **Symptom**: Training loop alternates between GPU compute and CPU wait
- **Cause**: Data augmentation, preprocessing on CPU during training
- **Solution**: Move augmentation to GPU (kornia, NVIDIA DALI), preprocess offline

**4. I/O Bottlenecks**
- **Symptom**: GPU starves waiting for data from GCS
- **Cause**: Network bandwidth, GCS read latency, small file sizes
- **Solution**: Use TFRecord/sharded formats, parallel data loading, local SSD caching

### Optimizing Batch Size for GPU Utilization

From [Optimizing Custom Job Runs on Vertex AI](https://medium.com/@haely.shah/optimizing-custom-job-runs-on-vertex-ai-a-comprehensive-guide-628c496e5eba) (accessed 2025-02-03):

**Recommended Batch Sizes:**
- **Start**: Powers of 2 (64, 128, 256)
- **Large models/datasets**: 512-2048 (if GPU memory allows)
- **Memory-limited**: Use gradient accumulation

**Batch Size Trade-offs:**

| Batch Size | GPU Utilization | Training Speed | Generalization | Memory |
|------------|----------------|----------------|----------------|---------|
| Small (32-64) | Low (~40-60%) | Slow | Better | Low |
| Medium (128-256) | High (~80-90%) | Fast | Good | Medium |
| Large (512-2048) | Very High (~95%) | Very Fast | May need tuning | High |

**Gradient Accumulation Pattern:**
```python
# Simulate large batch size without memory overhead
effective_batch_size = 512
physical_batch_size = 64
accumulation_steps = effective_batch_size // physical_batch_size  # 8

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### GPU Memory Optimization

**Techniques to Reduce Memory Usage:**

**1. Mixed Precision Training (FP16/BF16)**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16 ops
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2x memory reduction (FP16 vs FP32)
- 2-3x faster training on Tensor Core GPUs (V100, A100)
- Minimal accuracy impact (with loss scaling)

**2. Gradient Checkpointing**
```python
import torch.utils.checkpoint as checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Trade compute for memory
        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        return x
```

**Trade-off**: 20-30% slower training, 40-50% memory reduction

**3. Activation Checkpointing**
- Store only selected layer activations during forward pass
- Recompute others during backward pass
- Useful for very deep models (ResNet-152, Transformers)

### Monitoring GPU Utilization in Vertex AI

**Using nvidia-smi in Training Code:**
```python
import subprocess

def log_gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    gpu_util, mem_used, mem_total = result.stdout.strip().split(',')
    print(f"GPU Util: {gpu_util}% | Memory: {mem_used}/{mem_total} MB")

# Log every 100 steps
if step % 100 == 0:
    log_gpu_utilization()
```

**Cloud Profiler Integration:**
- Vertex AI TensorBoard Profiler captures GPU utilization automatically
- View trace timeline, kernel stats, memory usage over time
- Identify bottlenecks in training loop

## Profiling Tools

### TensorBoard Profiler

From [Enable Cloud Profiler for debugging model training performance](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-02-03):

**Setup in Training Code:**
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()  # Signal end of step
```

**Vertex AI Integration:**
```python
from google.cloud import aiplatform

# Create TensorBoard instance
tensorboard = aiplatform.Tensorboard.create(
    display_name='training-profiler',
    project='my-project',
    location='us-central1'
)

# Associate with custom training job
job = aiplatform.CustomTrainingJob(
    display_name='optimized-training',
    container_uri='gcr.io/my-project/trainer:latest',
    tensorboard=tensorboard.resource_name
)
```

### Profiler Analysis: 5 Key Views

From [5 ways to optimize training performance with TensorBoard Profiler](https://cloud.google.com/blog/topics/developers-practitioners/how-optimize-training-performance-tensorflow-profiler-vertex-ai) (accessed 2025-02-03):

**1. Overview Page**
- **Step Time Breakdown**: See where time is spent (input, compute, output)
- **Device Utilization**: GPU vs CPU time
- **Bottleneck Detection**: Highlights main performance limiters

**2. Trace Viewer**
- **Timeline Visualization**: See operations over time
- **Kernel Execution**: Individual GPU kernel durations
- **Data Transfer**: Host-to-device, device-to-host copies
- **Gaps**: Identify idle GPU time

**3. GPU Kernel Stats**
- **Top Kernels by Time**: Which operations dominate
- **Occupancy**: How well kernels utilize GPU cores
- **Memory Bandwidth**: Data throughput per kernel

**4. Memory Profile**
- **Peak Memory Usage**: Maximum VRAM consumed
- **Allocation Timeline**: When memory is allocated/freed
- **Memory Fragmentation**: Inefficient memory layout

**5. TensorFlow Op Profile**
- **Operation Time**: Time spent in each TF op type
- **Device Placement**: CPU vs GPU execution
- **Host Overheads**: Python, data loading, augmentation

### Common Profiling Insights

**Symptom: High Step Time, Low GPU Utilization**
- **Diagnosis**: Data loading bottleneck
- **Fix**: Increase `num_workers` in DataLoader, use prefetching
- **Profiler Evidence**: Large gaps in trace viewer between steps

**Symptom: GPU Utilization Spikes, Then Drops**
- **Diagnosis**: Small batch size, insufficient parallelism
- **Fix**: Increase batch size or use gradient accumulation
- **Profiler Evidence**: Short kernel execution times, low occupancy

**Symptom: High GPU Utilization, Slow Training**
- **Diagnosis**: Inefficient operations (e.g., reshape, transpose on GPU)
- **Fix**: Optimize operation order, fuse operations
- **Profiler Evidence**: Many small kernels instead of few large ones

**Symptom: Out of Memory During Training**
- **Diagnosis**: Peak memory exceeds GPU VRAM
- **Fix**: Reduce batch size, enable gradient checkpointing, use FP16
- **Profiler Evidence**: Memory profile shows spike before OOM

### PyTorch Profiler Integration

**Detailed Profiling:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for step in range(10):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# Export Chrome trace
prof.export_chrome_trace("trace.json")

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Output Analysis:**
```
---------------------------------  ------------  ------------
Name                               CPU time      CUDA time
---------------------------------  ------------  ------------
aten::conv2d                       15.234 ms     142.567 ms
aten::addmm                        8.123 ms      89.234 ms
aten::relu                         2.456 ms      12.345 ms
cudaMemcpyAsync                    5.678 ms      0.000 ms
```

## Multi-GPU Training Strategies

### PyTorch DistributedDataParallel (DDP)

From [PyTorch DistributedDataParallel Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (accessed 2025-02-03):

**DDP vs DataParallel:**

| Feature | DataParallel | DistributedDataParallel |
|---------|--------------|-------------------------|
| Process Model | Single process | Multi-process |
| Communication | Python GIL limited | NCCL backend |
| Scalability | Poor (>4 GPUs) | Excellent |
| Speed | Slow | 2-3x faster |
| Recommended | ❌ No | ✅ Yes |

**DDP Training Pattern:**
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    # Vertex AI sets these automatically
    dist.init_process_group(backend='nccl')

    # Set device for this process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank

def train():
    local_rank = setup_distributed()

    # Create model and move to GPU
    model = YourModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Distributed sampler ensures no data overlap
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch

        for batch in train_loader:
            batch = batch.to(local_rank)
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    dist.destroy_process_group()
```

### NCCL Optimization

**NCCL (NVIDIA Collective Communications Library)** handles gradient synchronization across GPUs.

**Environment Variables for NCCL Tuning:**
```yaml
# In Vertex AI job spec
env:
  - name: NCCL_DEBUG
    value: INFO  # Verbose logging for debugging
  - name: NCCL_IB_DISABLE
    value: "0"  # Enable InfiniBand (if available)
  - name: NCCL_NET_GDR_LEVEL
    value: "5"  # GPUDirect RDMA level
  - name: NCCL_P2P_LEVEL
    value: "NVL"  # NVLink peer-to-peer
```

**Communication Patterns:**

**All-Reduce (Default in DDP):**
- Each GPU computes gradients
- NCCL averages gradients across all GPUs
- Each GPU gets averaged result
- **Bandwidth**: Scales with number of GPUs

**Ring All-Reduce:**
- GPUs arranged in ring topology
- Data passes through ring, aggregating
- Bandwidth-optimal for many GPUs
- **NCCL uses this automatically for >8 GPUs**

**Gradient Accumulation with DDP:**
```python
# Simulate larger batch size with limited memory
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    # Don't sync gradients until accumulation_steps
    with model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext():
        loss = model(batch) / accumulation_steps
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit**: Reduces communication overhead by syncing less frequently

### Multi-Node Multi-GPU Training

**Vertex AI Custom Job Spec:**
```yaml
workerPoolSpecs:
- machineSpec:
    machineType: n1-highmem-16
    acceleratorType: NVIDIA_TESLA_V100
    acceleratorCount: 4
  replicaCount: 4  # 4 nodes × 4 GPUs = 16 total GPUs
  containerSpec:
    imageUri: gcr.io/my-project/trainer:latest
    env:
    - name: NCCL_DEBUG
      value: INFO
```

**Training Considerations:**
- **Global Batch Size**: `batch_size_per_gpu × num_gpus × num_nodes`
- **Learning Rate Scaling**: Linear scaling (LR × num_gpus) for large batches
- **Warmup**: Gradual LR increase for first few epochs stabilizes training
- **Gradient Clipping**: Important for multi-GPU stability

**Multi-Node Synchronization:**
```python
import torch.distributed as dist

# Only rank 0 saves checkpoints
if dist.get_rank() == 0:
    torch.save(model.state_dict(), 'checkpoint.pth')

# All ranks wait here
dist.barrier()

# Broadcast saved state to all ranks
if dist.get_rank() != 0:
    model.load_state_dict(torch.load('checkpoint.pth'))
```

### Mixed Precision + Multi-GPU

**Combining AMP with DDP:**
```python
from torch.cuda.amp import autocast, GradScaler

model = DDP(model, device_ids=[local_rank])
scaler = GradScaler()

for batch in train_loader:
    with autocast():  # FP16 forward pass
        loss = model(batch)

    scaler.scale(loss).backward()

    # Gradient clipping in FP32
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

**Performance Gains:**
- **V100/A100 GPUs**: 2-3x faster training
- **Memory Savings**: 2x (enables larger models/batches)
- **Accuracy**: Minimal loss with proper loss scaling

## I/O Optimization

### GCS Data Loading Strategies

**Problem**: Reading many small files from GCS is slow (high latency per request)

**Solution**: Use sharded formats (TFRecord, Parquet, HDF5)

**Sharding Pattern:**
```python
# Instead of 100,000 individual image files
# gs://bucket/images/img_00001.jpg
# gs://bucket/images/img_00002.jpg
# ...

# Use 100 sharded files
# gs://bucket/shards/shard_000.tfrecord (1000 images)
# gs://bucket/shards/shard_001.tfrecord (1000 images)
# ...
```

**Benefits:**
- Fewer GCS requests (100 vs 100,000)
- Better sequential read performance
- Parallel shard loading across workers

### DataLoader Optimization

From [Optimizing Custom Job Runs on Vertex AI](https://medium.com/@haely.shah/optimizing-custom-job-runs-on-vertex-ai-a-comprehensive-guide-628c496e5eba) (accessed 2025-02-03):

**Optimal DataLoader Configuration:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,  # 4 CPU workers per GPU
    pin_memory=True,  # Faster host-to-device transfer
    prefetch_factor=2,  # Prefetch 2 batches per worker
    persistent_workers=True  # Reuse workers (PyTorch 1.7+)
)
```

**num_workers Guidelines:**
- **Rule of Thumb**: 2-4 workers per GPU
- **Too Few**: Data loading bottleneck
- **Too Many**: CPU contention, memory overhead
- **Test Range**: Try 0, 2, 4, 8 and measure step time

**Prefetching Pattern:**
```python
# Manual prefetching (if DataLoader doesn't support)
class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True
        for next_batch in self.loader:
            with torch.cuda.stream(self.stream):
                next_batch = next_batch.cuda(non_blocking=True)
            if not first:
                yield batch
            else:
                first = False
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
        yield batch

train_loader = PrefetchLoader(DataLoader(...))
```

### Local SSD Caching

**Vertex AI Custom Job with Local SSD:**
```yaml
workerPoolSpecs:
- machineSpec:
    machineType: n1-highmem-16
    acceleratorType: NVIDIA_TESLA_V100
    acceleratorCount: 1
  diskSpec:
    bootDiskType: pd-ssd
    bootDiskSizeGb: 200  # Cache dataset here
```

**Caching Strategy:**
```python
import os
import shutil
from google.cloud import storage

def cache_dataset_to_local_ssd():
    """Download GCS data to local SSD once at training start"""
    local_cache = '/mnt/disks/ssd/data'
    gcs_path = 'gs://my-bucket/training-data'

    if not os.path.exists(local_cache):
        os.makedirs(local_cache)
        print("Downloading dataset to local SSD...")
        # Use gsutil for fast parallel download
        os.system(f'gsutil -m cp -r {gcs_path}/* {local_cache}/')

    return local_cache

# Use at training start
local_data_path = cache_dataset_to_local_ssd()
dataset = ImageDataset(local_data_path)  # Load from SSD
```

**Benefits:**
- **First Epoch**: Slow (download from GCS)
- **Subsequent Epochs**: Fast (read from local SSD)
- **Cost**: Minimal (local SSD included in machine type)

### Data Pipeline Profiling

**Measuring Data Loading Time:**
```python
import time

def profile_data_loading():
    data_time = 0
    compute_time = 0

    for batch in train_loader:
        t0 = time.time()
        batch = batch.to(device)
        t1 = time.time()

        loss = model(batch)
        loss.backward()
        optimizer.step()
        t2 = time.time()

        data_time += (t1 - t0)
        compute_time += (t2 - t1)

    print(f"Data Time: {data_time:.2f}s ({data_time/(data_time+compute_time)*100:.1f}%)")
    print(f"Compute Time: {compute_time:.2f}s ({compute_time/(data_time+compute_time)*100:.1f}%)")

# Target: Data time <20% of total time
```

**Interpretation:**
- **Data Time >50%**: I/O bottleneck (increase workers, use SSD cache)
- **Data Time 20-50%**: Acceptable (some overlap with compute)
- **Data Time <20%**: Well optimized (GPU is bottleneck)

## Memory Optimization Techniques

### Gradient Checkpointing Deep Dive

**Memory vs Compute Trade-off:**
```python
# Without checkpointing
# Memory: O(n) for n layers (store all activations)
# Compute: 1x forward + 1x backward

# With checkpointing
# Memory: O(sqrt(n)) (store checkpoints only)
# Compute: 1x forward + 2x forward (recompute) + 1x backward
```

**Selective Checkpointing:**
```python
import torch.utils.checkpoint as cp

class OptimizedTransformer(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(24)])

    def forward(self, x):
        # Checkpoint every 4th layer
        for i, layer in enumerate(self.layers):
            if i % 4 == 0:
                x = cp.checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

**When to Use:**
- **Deep Models**: >50 layers
- **Memory-Bound**: GPU utilization low due to small batch size
- **Acceptable Slowdown**: 20-30% longer training time

### Memory-Efficient Attention

**Standard Attention Memory: O(n²)**
```python
# Standard attention (memory-intensive)
Q = query_proj(x)  # (batch, seq_len, dim)
K = key_proj(x)
V = value_proj(x)

attention_scores = Q @ K.T  # (batch, seq_len, seq_len) - O(n²) memory!
attention_probs = softmax(attention_scores)
output = attention_probs @ V
```

**Flash Attention: O(n) Memory**
```python
# Flash Attention (memory-efficient)
from flash_attn import flash_attn_func

output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False
)
# Same output, 4x less memory, 2x faster
```

**Benefits:**
- 75% memory reduction for attention
- 2x faster on A100 GPUs
- Enables longer sequences (4k → 16k tokens)

### Offloading Strategies

**CPU Offloading for Large Models:**
```python
from deepspeed import zero

# ZeRO Stage 3: Offload optimizer states to CPU
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config={
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        }
    }
)
```

**When to Use:**
- Models >10B parameters
- Limited GPU memory
- Acceptable: 10-20% slowdown

## Cost-Performance Trade-offs

### GPU Type Selection

**Vertex AI GPU Options:**

| GPU Type | Memory | Cost/Hour | Best Use Case |
|----------|--------|-----------|---------------|
| NVIDIA T4 | 16 GB | $0.35 | Development, small models |
| NVIDIA V100 | 16 GB | $2.48 | Production training |
| NVIDIA A100 | 40 GB | $3.67 | Large models, fast training |
| NVIDIA A100 | 80 GB | $4.89 | Huge models (>20B params) |

**Performance Comparison (ResNet-50, ImageNet):**
- **T4**: 150 images/sec, 48h training time
- **V100**: 400 images/sec, 18h training time
- **A100-40GB**: 800 images/sec, 9h training time
- **A100-80GB**: 850 images/sec, 8.5h training time (memory not bottleneck)

**Cost Optimization:**
- **Development**: Use T4 for debugging, hyperparameter search
- **Production**: Use V100/A100 for final training runs
- **Large Models**: Use A100-80GB only if model doesn't fit in 40GB

### Spot/Preemptible Instances

**Cost Savings: 60-91%**

**Vertex AI Spot Configuration:**
```yaml
scheduling:
  timeout: 86400s  # 24 hours
  restartJobOnWorkerRestart: true
```

**Checkpointing for Preemption:**
```python
import os
import torch
from google.cloud import storage

def save_checkpoint(model, optimizer, epoch, step):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    # Save locally first
    local_path = f'/tmp/checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, local_path)

    # Upload to GCS (survives preemption)
    client = storage.Client()
    bucket = client.bucket('my-training-checkpoints')
    blob = bucket.blob(f'model/checkpoint_epoch_{epoch}.pth')
    blob.upload_from_filename(local_path)

def resume_from_checkpoint():
    # Check GCS for latest checkpoint
    client = storage.Client()
    bucket = client.bucket('my-training-checkpoints')
    blobs = list(bucket.list_blobs(prefix='model/checkpoint'))

    if blobs:
        latest = sorted(blobs, key=lambda x: x.time_created)[-1]
        local_path = f'/tmp/{latest.name}'
        latest.download_to_filename(local_path)
        return torch.load(local_path)
    return None

# Training loop
checkpoint = resume_from_checkpoint()
start_epoch = 0 if checkpoint is None else checkpoint['epoch']

for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(train_loader):
        # Training
        loss = model(batch)
        loss.backward()
        optimizer.step()

        # Save checkpoint every 1000 steps
        if step % 1000 == 0:
            save_checkpoint(model, optimizer, epoch, step)
```

**Best Practices:**
- Save checkpoints every 15-30 minutes
- Test resume logic before long training runs
- Monitor preemption rate (should be <10%)

### Training Time vs Cost Analysis

**Example: ResNet-50 on ImageNet (90 epochs)**

| Configuration | Time | Cost | Cost/Epoch |
|---------------|------|------|------------|
| 1x V100 | 48h | $119 | $1.32 |
| 4x V100 (DDP) | 14h | $139 | $1.54 |
| 8x V100 (DDP) | 8h | $159 | $1.77 |
| 1x A100 | 24h | $88 | $0.98 |
| 4x A100 (DDP) | 7h | $103 | $1.14 |

**Insights:**
- **Single GPU**: Cheapest per-run, slowest time
- **Multi-GPU**: Faster training, higher total cost, higher cost per epoch
- **A100 vs V100**: A100 is faster AND cheaper for this workload
- **Scaling Efficiency**: 4 GPUs = 3.4x speedup (not 4x, due to communication)

**Cost Optimization Strategy:**
1. **Development**: 1x T4 ($8/day)
2. **Hyperparameter Tuning**: 4x V100 spot ($60/day, 70% savings)
3. **Final Training**: 8x A100 spot ($120/day, 70% savings)

## Sources

**Google Cloud Documentation:**
- [Enable Cloud Profiler for debugging model training performance](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) - Vertex AI TensorBoard Profiler guide (accessed 2025-02-03)
- [5 ways to optimize training performance with TensorBoard Profiler](https://cloud.google.com/blog/topics/developers-practitioners/how-optimize-training-performance-tensorflow-profiler-vertex-ai) - Google Cloud Blog (accessed 2025-02-03)
- [Configure compute resources for custom training](https://cloud.google.com/vertex-ai/docs/training/configure-compute) - Worker pool specifications (accessed 2025-02-03)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing) - GPU costs and SKU groups (accessed 2025-02-03)

**PyTorch Documentation:**
- [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) - PyTorch DDP reference (accessed 2025-02-03)
- [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) - PyTorch profiling tools

**Community Resources:**
- [Optimizing Custom Job Runs on Vertex AI](https://medium.com/@haely.shah/optimizing-custom-job-runs-on-vertex-ai-a-comprehensive-guide-628c496e5eba) - Comprehensive optimization guide by Haely Shah (accessed 2025-02-03)
- [Some PyTorch multi-GPU training tips](https://cerfacs.fr/coop/pytorch-multi-gpu) - Cerfacs multi-GPU guide (accessed 2025-02-03)
- [Multi-GPU Training with PyTorch (DDP)](https://medium.com/@bingqian/multi-gpu-training-with-pytorch-ddp-9eeefe5e2b13) - DDP training tutorial (accessed 2025-02-03)

**Performance Research:**
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704) - NCCL optimization research
- [Understanding the Impact of GPU Memory on Training Large Language Models](https://hydrahost.com/post/understanding-impact-gpu-memory-training-large-language-models/) - Memory optimization strategies (accessed 2025-02-03)
