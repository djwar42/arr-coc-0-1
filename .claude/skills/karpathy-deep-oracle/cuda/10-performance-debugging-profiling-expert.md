# CUDA Performance Debugging & Profiling (Expert-Level)

**Expert-level guide to profiling PyTorch CUDA kernels, diagnosing performance bottlenecks, and optimization strategies**

Performance debugging transforms slow CUDA kernels into production-ready code. This guide covers systematic profiling workflows, bottleneck diagnosis (compute vs memory vs overhead), and expert optimization techniques using NSight Systems, NSight Compute, and torch.profiler.

---

## Overview: The Three Performance Bottlenecks

From [Meta's AI Performance Profiling Infrastructure](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/) (PyTorch Blog, accessed 2025-11-13):

> "Slow CUDA kernels can become major performance bottlenecks. These include slow data loading, small and/or slow GPU kernels, distributed training issues such as load imbalance and excessive communication."

**The Three Bottleneck Categories:**

1. **Compute Bound** - Low SM (Streaming Multiprocessor) utilization, underutilized Tensor Cores
2. **Memory Bound** - HBM bandwidth saturation, uncoalesced memory access, cache misses
3. **Overhead Bound** - Kernel launch overhead, CPU-GPU synchronization, small kernels

From [CUDA 3: Your Checklist for Optimizing CUDA Kernels](https://medium.com/@rimikadhara/cuda-3-your-checklist-for-optimizing-cuda-kernels-68ef2a42332d) (Medium, accessed 2025-11-13):

> "Performance hinges on three things: compute (how efficient are your threads?), memory bandwidth (how fast can you move data?), and overhead (how much time is wasted not computing?)."

**Profiling Workflow:**

```
Step 1: Identify bottleneck category (compute/memory/overhead)
   ↓
Step 2: Profile with appropriate tool
   ↓
Step 3: Analyze metrics (SM utilization, memory bandwidth, kernel time)
   ↓
Step 4: Apply targeted optimizations
   ↓
Step 5: Re-profile and validate improvement
```

---

## 1. Profiling Tools & Workflow

### NSight Systems (System-Level Profiling)

From [Navigating NVIDIA Nsight Systems for Efficient Profiling](https://henryhmko.github.io/posts/profiling/profiling.html) (GitHub Pages, accessed 2025-11-13):

**Why NSight Systems:**
- Non-intrusive (no code changes required)
- Shows CPU + GPU timeline correlation
- Framework agnostic (PyTorch, JAX, TensorFlow)
- Granular trace information (CUDA API calls, kernel launches, memory transfers)

**Basic Profiling Command:**

```bash
nsys profile --trace=cuda,nvtx,osrt -o output_profile python train.py
```

**Flags explained:**
- `--trace=cuda`: CUDA API calls and kernel information
- `--trace=nvtx`: Custom annotations (NVIDIA Tools Extension)
- `--trace=osrt`: OS runtime (threading, synchronization, hardware I/O)
- `-o output_profile`: Custom output filename (generates .nsys-rep file)

**Opening Results:**

```bash
# Open in NSight Systems GUI
nsight-sys output_profile.nsys-rep
```

**Timeline View Breakdown:**

```
┌─ CUDA HW          : GPU utilization rate (SM active %)
├─ Python/Process   : CPU utilization rate
├─ OS Runtime Libs  : Thread activities (sync, semaphores)
├─ CUDA API         : cudaMalloc, cudaMemcpy, kernel launches
└─ NVTX Annotations : Custom code markers (forward, backward, optimizer)
```

From the guide:

> "Click the `CUDA HW` dropdown and see the `Kernels` row. It contains information about grid sizes, block sizes, registers per thread, and such which is useful for calculating the occupancy of SMs."

### Adding NVTX Annotations

**Example: Annotating Training Loop**

```python
import nvtx

with nvtx.annotate("training"):
    for epoch in range(num_epochs):
        with nvtx.annotate("epoch"):
            for batch_idx, (data, targets) in enumerate(train_loader):
                with nvtx.annotate("forward"):
                    outputs = model(data)

                with nvtx.annotate("calculate loss"):
                    loss = criterion(outputs, targets)

                with nvtx.annotate("backward"):
                    loss.backward()

                with nvtx.annotate("optimizer step"):
                    optimizer.step()
```

**Benefits:**
- Maps profiler timeline to code sections
- Identifies slow stages (data loading, forward, backward, optimizer)
- Isolates bottlenecks to specific operations

### NSight Compute (Kernel-Level Profiling)

**When to use:** After NSight Systems identifies slow kernels, use NSight Compute for deep kernel analysis.

**Profiling Command:**

```bash
ncu --set full -o kernel_profile python train.py
```

**Metrics Collected:**
- SM efficiency (warp occupancy, active warps)
- Memory throughput (L1/L2 cache hit rates, DRAM bandwidth utilization)
- Compute throughput (FP32/FP16/INT8 operations per second)
- Tensor Core utilization (% of peak TFLOPs achieved)

**Opening Results:**

```bash
ncu-ui kernel_profile.ncu-rep
```

### torch.profiler (PyTorch-Specific)

From [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (PyTorch Docs, accessed 2025-11-13):

**Basic Usage:**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # Profile first 10 batches
            break

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Advanced: TensorBoard Integration**

```python
from torch.profiler import profile, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()  # Signal profiler to move to next step

# View in TensorBoard
# tensorboard --logdir=./log/profiler
```

**Key Metrics:**
- `cuda_time_total`: Total GPU time spent
- `cpu_time_total`: Total CPU time spent
- `cuda_memory_usage`: Peak GPU memory allocated
- `self_cuda_time_total`: GPU time excluding sub-operations

---

## 2. GPU Utilization Debugging

### Case Study: Meta Production Model

From [Meta's Performance Debugging Blog](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/) (accessed 2025-11-13):

**Problem Identified:**

```
Timeline Analysis:
- Pattern: "GPU-idle, GPU-active, GPU-idle, GPU-active..."
- GPU idle for >50% of training time
- SM utilization: 9.1% (target: >80%)
- Tensor Core utilization: 0% (should be >50% for mixed precision)
- GPU memory utilization: 47.13% (half unused)
```

**Root Cause:**

```python
# CPU profiling during GPU idle period
# Bottleneck: sharded_iterrows() function
# Issue: Dataset too large for CPU memory
# Solution: Read sub-datasets every 10 epochs → GPU sits idle during I/O
```

**Step-by-Step Diagnosis:**

**Step 1: Check CPU vs GPU Timeline**

```
NSight Systems View:
┌─ CPU Usage  ████░░░░████░░░░  (spiky pattern)
└─ GPU Usage  ░░░░████░░░░████  (inverse pattern - BAD!)

Goal: GPU usage should be continuous (████████████)
```

**Step 2: Profile CPU During GPU Idle**

```python
# Identify CPU function consuming time while GPU is idle
# Use torch.profiler or Python cProfile

import cProfile

with cProfile.Profile() as pr:
    # Run training iteration
    pr.print_stats(sort='cumtime')
```

**Step 3: Collect GPU Performance Metrics**

```bash
# NVIDIA DCGM metrics
dcgmi dmon -e 1001,1002,1003,1004  # SM active %, memory bandwidth, Tensor Core %

# Expected values (A100/H100):
# - SM utilization: >80% (good), <50% (poor)
# - Memory bandwidth: >70% of peak (good)
# - Tensor Core utilization: >50% (if using mixed precision)
```

### Low SM Utilization Diagnosis

**Common Causes:**

1. **Small Batch Sizes**
   - Symptom: Grid size << number of SMs
   - Fix: Increase batch size (use gradient accumulation if OOM)

2. **Kernel Launch Overhead**
   - Symptom: Many small kernels (<100μs each)
   - Fix: Kernel fusion (fuse multiple operations into one kernel)

3. **Thread Divergence**
   - Symptom: Warps executing different code paths
   - Fix: Minimize if/else branches in kernel code

4. **Register Spilling**
   - Symptom: High register usage per thread → low occupancy
   - Fix: Reduce local variables, use shared memory

**Occupancy Calculation:**

```python
# Check kernel occupancy
import torch

# Get device properties
device = torch.device('cuda:0')
props = torch.cuda.get_device_properties(device)

# Example kernel: 256 threads/block, 64 registers/thread
threads_per_block = 256
registers_per_thread = 64

# Calculate occupancy
max_blocks_per_sm = props.max_threads_per_multiprocessor // threads_per_block
max_registers_per_sm = props.regs_per_multiprocessor
max_blocks_by_registers = max_registers_per_sm // (threads_per_block * registers_per_thread)

occupancy = min(max_blocks_per_sm, max_blocks_by_registers) / max_blocks_per_sm
print(f"Theoretical occupancy: {occupancy * 100:.1f}%")

# Goal: >75% occupancy
```

### Zero Tensor Core Utilization

**Diagnosis Workflow:**

```python
# Check if mixed precision is enabled
import torch

# Verify AMP is active
print(torch.backends.cuda.matmul.allow_tf32)  # Should be True for A100/H100

# Check dtype in forward pass
with torch.cuda.amp.autocast():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    z = torch.matmul(x, y)
    print(f"Result dtype: {z.dtype}")  # Should be float16 or bfloat16
```

**Common Causes:**

1. **Using FP32 Throughout**
   - Fix: Enable `torch.cuda.amp.autocast()`

2. **Matrix Dimensions Not Aligned**
   - Tensor Cores require dimensions aligned to 8 (FP16) or 16 (INT8)
   - Fix: Pad tensors to multiples of 8/16

3. **Small Matrix Sizes**
   - Tensor Cores inefficient for tiny matrices (<128×128)
   - Fix: Batch small operations together

**Example Fix:**

```python
# Before: FP32 (no Tensor Cores)
model = MyModel().cuda()
loss = model(data)

# After: Mixed precision (Tensor Cores active)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(data)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 3. Memory Bandwidth Analysis

### HBM Bandwidth Saturation

**Theoretical vs Achieved Bandwidth:**

```
A100 80GB:
- Peak HBM bandwidth: 2.0 TB/s
- Typical achieved: 1.4-1.6 TB/s (70-80%)
- Memory-bound if: >80% bandwidth utilized AND low SM utilization

H100 80GB:
- Peak HBM bandwidth: 3.35 TB/s
- Typical achieved: 2.4-2.8 TB/s (70-85%)
```

**Measuring Bandwidth Utilization:**

```bash
# NSight Compute
ncu --metrics dram__bytes.sum.per_second python script.py

# Expected output:
# dram__bytes.sum.per_second = 1.4 TB/s (A100)
# If >1.8 TB/s (>90% peak) → memory-bound bottleneck
```

**Optimization: Kernel Fusion**

```python
# Before: Two memory-bound kernels
x = torch.relu(input)        # Kernel 1: load input, store x
y = torch.dropout(x, p=0.1)  # Kernel 2: load x, store y

# After: Fused kernel (custom CUDA extension)
# Single kernel: load input, compute relu + dropout, store y
# 2× reduction in memory traffic
```

### Memory Coalescing

From [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) (NVIDIA Docs, accessed 2025-11-13):

**Coalesced vs Uncoalesced Access:**

```cpp
// Uncoalesced (BAD): Each warp accesses scattered memory
__global__ void uncoalesced_kernel(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx * stride];  // Stride access → many cache line loads
}

// Coalesced (GOOD): Warp threads access contiguous memory
__global__ void coalesced_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // Sequential access → single cache line load
}
```

**Performance Impact:**

```
Uncoalesced access (stride=32):
- 32 cache line loads per warp (128 bytes each)
- Effective bandwidth: 60 GB/s (3% of A100 peak)

Coalesced access:
- 1 cache line load per warp (if all threads in same 128-byte line)
- Effective bandwidth: 1,600 GB/s (80% of A100 peak)

Speedup: 26× faster
```

**PyTorch Memory Layout:**

```python
# Check tensor memory layout
x = torch.randn(1024, 1024, device='cuda')
print(x.is_contiguous())  # Should be True

# Non-contiguous after transpose
y = x.T
print(y.is_contiguous())  # False

# Fix: Make contiguous
y_contig = y.contiguous()
print(y_contig.is_contiguous())  # True

# Before contiguous(): memory access jumps by stride
# After contiguous(): sequential memory access (coalesced)
```

### Cache Optimization

**L1/L2 Cache Hit Rates:**

```bash
# NSight Compute metrics
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l2_cache_hit_rate python script.py

# Goal:
# - L1 cache hit rate: >80% (good), <50% (poor)
# - L2 cache hit rate: >70% (good)
```

**Shared Memory Optimization:**

```cpp
// Use shared memory to increase cache hits
__global__ void optimized_kernel(float* input, float* output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory (coalesced global mem access)
    tile[threadIdx.y][threadIdx.x] = input[row * N + col];
    __syncthreads();

    // Compute using shared memory (zero global mem access)
    float result = 0.0f;
    for (int k = 0; k < TILE_SIZE; k++) {
        result += tile[threadIdx.y][k] * tile[k][threadIdx.x];
    }

    output[row * N + col] = result;
}
```

**Benefits:**
- Shared memory bandwidth: ~19 TB/s (10× faster than HBM)
- Reduces global memory traffic
- Improves cache locality

---

## 4. Expert Optimization Techniques

### Kernel Fusion

**Pattern 1: Elementwise Operation Fusion**

```python
# Before: 3 separate kernels
x = torch.relu(input)
y = x * scale
z = y + bias

# After: Fused kernel (torch.jit.script or custom CUDA)
@torch.jit.script
def fused_op(input: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return torch.relu(input) * scale + bias

z = fused_op(input, scale, bias)
```

**Benefits:**
- 3 kernel launches → 1 kernel launch
- 3× reduction in memory traffic (input loaded once, not 3 times)
- Speedup: ~2-3× for memory-bound operations

**Pattern 2: Reduction Fusion**

```python
# Before: Separate mean and variance computation
mean = x.mean(dim=-1, keepdim=True)
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

# After: Fused (torch.var computes both in single pass)
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, unbiased=False, keepdim=True)
```

### Block Size Tuning

**Finding Optimal Block Size:**

```python
import torch
import time

def benchmark_block_size(kernel_func, block_sizes, input_tensor):
    results = {}

    for block_size in block_sizes:
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):  # Warm-up + timing
            kernel_func(input_tensor, threads_per_block=block_size)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        results[block_size] = elapsed / 100  # Average time

    return results

# Test common block sizes
block_sizes = [64, 128, 256, 512, 1024]
timings = benchmark_block_size(my_kernel, block_sizes, input_data)

optimal_size = min(timings, key=timings.get)
print(f"Optimal block size: {optimal_size}")
```

**Heuristics:**
- Most GPUs: 128-256 threads/block works well
- Memory-bound kernels: Larger blocks (512-1024) for better coalescing
- Compute-bound kernels: Smaller blocks (128-256) for higher occupancy

### Warp Efficiency

**Avoiding Thread Divergence:**

```cpp
// BAD: Warp divergence (if branches)
__global__ void divergent_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0) {
        data[idx] = compute_even(data[idx]);  // Half of warp executes this
    } else {
        data[idx] = compute_odd(data[idx]);   // Other half executes this
    }
    // Result: Warp executes BOTH branches sequentially (2× slower)
}

// GOOD: No divergence (all threads execute same code)
__global__ void efficient_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = data[idx];
    int is_even = (idx % 2 == 0);

    // Branchless computation (both paths computed, one selected)
    float result_even = compute_even(val);
    float result_odd = compute_odd(val);
    data[idx] = is_even ? result_even : result_odd;
}
```

**Measuring Warp Divergence:**

```bash
# NSight Compute
ncu --metrics smsp__average_warps_issue_stalled_short_scoreboard.pct python script.py

# High stall % → likely divergence issue
```

### Asynchronous Operations

**Overlapping Compute and Memory Transfers:**

```python
import torch

# Create CUDA streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Overlap H2D transfer with computation
with torch.cuda.stream(stream1):
    data_gpu = data_cpu.to('cuda', non_blocking=True)  # Async H2D

with torch.cuda.stream(stream2):
    output = model(prev_data)  # Compute on previous batch

# Synchronize before next iteration
torch.cuda.synchronize()
```

**Benefits:**
- Hide PCIe transfer latency (~10-20% speedup)
- Better GPU utilization (compute while transferring)

---

## 5. Common Performance Anti-Patterns

### Anti-Pattern 1: Small Batch Sizes

```python
# BAD: batch_size=8 (underutilizes GPU)
dataloader = DataLoader(dataset, batch_size=8)

# GOOD: batch_size=128 (better GPU utilization)
dataloader = DataLoader(dataset, batch_size=128)

# If OOM: Use gradient accumulation
for i, (data, target) in enumerate(dataloader):
    loss = model(data) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Anti-Pattern 2: CPU-GPU Synchronization

```python
# BAD: .item() forces CPU-GPU sync every iteration
for i, (data, target) in enumerate(dataloader):
    loss = model(data)
    loss.backward()

    # This blocks GPU until result is copied to CPU!
    loss_val = loss.item()
    print(f"Loss: {loss_val}")

# GOOD: Accumulate losses, sync once at end
losses = []
for i, (data, target) in enumerate(dataloader):
    loss = model(data)
    loss.backward()
    losses.append(loss.detach())  # Store on GPU

# Sync once after training loop
avg_loss = torch.stack(losses).mean().item()
print(f"Average loss: {avg_loss}")
```

### Anti-Pattern 3: Inefficient DataLoader

```python
# BAD: Single-threaded data loading (CPU bottleneck)
dataloader = DataLoader(dataset, batch_size=64, num_workers=0)

# GOOD: Multi-threaded + pinned memory
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,          # Use multiple CPU cores
    pin_memory=True,        # Faster H2D transfers
    persistent_workers=True # Reuse worker processes
)
```

From Meta's case study:

> "Tuning `num_worker_threads` by trying a few possible values within the number of CPU cores on each host" eliminated GPU idle time during data loading.

---

## 6. Production Profiling Workflow (Meta's Approach)

From [Meta's Performance Debugging Blog](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/):

**MAIProf Design Principles:**

1. **No source code changes required** - Profile by sampling execution
2. **Holistic view** - System-wide analysis (CPU + GPU)
3. **Multiple tools for different skill levels** - From automatic trace comprehension to low-level analysis
4. **Distributed GPU profiling** - Multi-host, multi-GPU combined view
5. **Highly scalable** - Service-based architecture

**4-Step Optimization Process:**

```
Step 1: Inspect CPU/GPU utilization timeline
   → Identify "GPU-idle, GPU-active" patterns

Step 2: Collect Python call trace during GPU idle
   → Find CPU bottleneck (e.g., data loading)

Step 3: Collect GPU performance metrics
   → Measure SM utilization, Tensor Core usage, memory utilization

Step 4: Collect GPU trace (Kineto)
   → Identify slow kernels, phase breakdown (data/forward/backward/optimizer)
```

**Optimizations Applied (Production Model):**

```
1. Tune num_worker_threads      → Eliminated GPU idle time
2. Double batch sizes            → Improved GPU memory utilization (47% → 85%)
3. Enable mixed precision (AMP)  → Enabled Tensor Cores (0% → 62% utilization)
4. Use multitensor optimizer     → Reduced small kernel overhead
```

**Results:**
- SM utilization: 9.1% → 74% (8× improvement)
- Tensor Core utilization: 0% → 62%
- GPU memory utilization: 47% → 85%
- Overall speedup: ~4-5× end-to-end training time

---

## 7. Profiling Gotchas & Best Practices

### torch.compile Profiling

From [Navigating NVIDIA Nsight Systems](https://henryhmko.github.io/posts/profiling/profiling.html):

**Issue: First iteration extremely slow with torch.compile**

```
Eager mode:
  train time 0: 0.218s
  train time 1: 0.049s
  train time 2: 0.047s

torch.compile mode:
  train time 0: 227.8s  ← Compilation overhead!
  train time 1: 9.1s    ← Still compiling
  train time 2: 0.027s  ← Now optimized (2× faster than eager)
```

**Best Practice: Warm-up before profiling**

```python
model = torch.compile(model)

# Warm-up (compile kernels)
for _ in range(5):
    loss = model(dummy_data)
    loss.backward()

# Now profile
with torch.profiler.profile() as prof:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
```

### Profiling Overhead

**NSight Systems overhead: 5-10%**
- Safe for production profiling
- Minimal impact on timing accuracy

**NSight Compute overhead: 50-100%**
- DO NOT use in production
- Use only for isolated kernel analysis

**torch.profiler overhead: 10-20%**
- Profile limited iterations (10-100)
- Disable profiling for accuracy measurements

### Profiling Distributed Training

```python
# Only profile rank 0 to avoid overwhelming storage
import torch.distributed as dist

if dist.get_rank() == 0:
    with torch.profiler.profile() as prof:
        train_loop()
else:
    train_loop()
```

---

## Sources

**Web Research (accessed 2025-11-13):**

- [Performance Debugging of Production PyTorch Models at Meta](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/) - Meta AI, PyTorch Blog
  - MAIProf infrastructure design
  - Case study: 4-5× speedup on production model
  - SM utilization diagnosis (9.1% → 74%)

- [Navigating NVIDIA Nsight Systems for Efficient Profiling](https://henryhmko.github.io/posts/profiling/profiling.html) - Henry Ko, GitHub Pages
  - NSight Systems practical tutorial
  - NVTX annotation examples
  - torch.compile profiling gotchas

- [CUDA 3: Your Checklist for Optimizing CUDA Kernels](https://medium.com/@rimikadhara/cuda-3-your-checklist-for-optimizing-cuda-kernels-68ef2a42332d) - Rimika Dhara, Medium
  - Three bottleneck categories (compute/memory/overhead)
  - Profiling workflow and systematic optimization

**Additional References:**

- [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - PyTorch Official Docs
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - NVIDIA Documentation (memory coalescing, bandwidth optimization)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) - NVIDIA Documentation

---

**Related Files:**

- `cuda/00-streams-concurrency-async.md` - Async operations for hiding latency
- `cuda/01-memory-management-unified.md` - Memory allocation and profiling
- `cuda/05-tensor-core-programming-wmma-mma.md` - Tensor Core optimization
- `cuda/06-pytorch-jit-torch-compile.md` - torch.compile and kernel fusion
- `cuda/07-mixed-precision-training-internals.md` - Mixed precision profiling
- `karpathy/llm-gpu-integration/00-flashattention-internals.md` - Memory-bound kernel optimization example
