# GPU Profiling: Nsight Systems, Nsight Compute, and TensorBoard Profiler

**Deep dive into GPU profiling tools for identifying performance bottlenecks, analyzing kernel execution, and optimizing training workflows**

---

## Overview

GPU profiling is the foundation of performance optimization for deep learning. When training runs slowly, profiling reveals *why* - whether it's inefficient kernels, CPU-GPU synchronization overhead, memory bandwidth saturation, or data loading bottlenecks. Modern profiling tools provide timeline visualizations, kernel-level metrics, and memory access patterns that turn "my training is slow" into actionable optimization targets.

From [PyTorch Profiler documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (accessed 2025-11-14):
> "PyTorch profiler is enabled through the context manager and accepts a number of parameters... activities can profile CPU operators, TorchScript functions, and on-device CUDA kernels."

From [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) (accessed 2025-11-14):
> "NVIDIA Nsight Systems is a system-wide performance analysis tool designed to visualize an application's algorithms, identify the largest opportunities to optimize, and tune to scale efficiently across any quantity or size of CPUs and GPUs."

**Why GPU Profiling Matters:**
- **Identify bottlenecks** - Find the 5% of code consuming 95% of runtime
- **Optimize kernel usage** - Detect underutilized Tensor Cores or low occupancy
- **Debug memory issues** - Trace allocations, find OOM causes, eliminate leaks
- **Improve GPU utilization** - Move from 40% to 95% utilization through profiling insights
- **Validate optimizations** - Measure actual speedup from code changes

**The Profiling Stack:**
```
Timeline Analysis (what happens when)
    ↓
Nsight Systems - System-wide view, CPU-GPU interaction
    ↓
Kernel Analysis (why kernels are slow)
    ↓
Nsight Compute - Per-kernel metrics, roofline analysis
    ↓
Framework Integration (easy profiling in Python)
    ↓
PyTorch Profiler + TensorBoard - Training loop analysis
```

---

## Section 1: Nsight Systems - Timeline View and System Analysis

### What is Nsight Systems?

Nsight Systems provides a **chronological timeline view** of everything happening on your system - CPU threads, GPU kernels, memory transfers, API calls, and OS interactions. It answers "what is the GPU doing right now?" and "why is the GPU idle?"

**Key Features:**
- System-wide profiling with minimal overhead (<5%)
- Timeline visualization (CPU activity, GPU kernels, memory transfers)
- Multi-GPU and multi-process support
- Network metrics for distributed training
- Python backtrace sampling for deep learning frameworks

From [NVIDIA Nsight Systems documentation](https://developer.nvidia.com/nsight-systems) (accessed 2025-11-14):
> "Nsight Systems visualizes system workload metrics on a unified timeline, allowing developers to investigate correlations, dependencies, activity, bottlenecks, and resource allocation."

### Installing and Running Nsight Systems

**Installation:**
```bash
# Download from NVIDIA Developer website
# https://developer.nvidia.com/nsight-systems

# Or via package manager (Linux)
sudo apt-get install nsight-systems-cli

# Verify installation
nsys --version
```

**Basic Profiling Workflow:**
```bash
# Profile a Python training script
nsys profile -o my_profile \
    --trace cuda,cudnn,cublas,nvtx,osrt \
    --cuda-memory-usage true \
    python train.py

# Profile with GPU metrics sampling (adds overhead but provides SM utilization)
nsys profile -o my_profile \
    --trace cuda,cudnn,cublas,nvtx \
    --cuda-memory-usage true \
    --gpu-metrics-device 0 \
    --sample cpu \
    python train.py

# Open the GUI to view results
nsys-ui my_profile.nsys-rep
```

**Timeline View Components:**
```
[CPU Thread 1]  ────┬───┬───────┬──────  (Python execution)
                    │   │       │
[CUDA API]      ────┴───┴───────┴──────  (cudaLaunchKernel, cudaMemcpy)
                        │       │
[GPU Kernels]   ────────●───────●──────  (Actual kernel execution)
                        │
[Memory Copy]   ────────▬───────────────  (H2D, D2H transfers)
```

### Identifying Common Bottlenecks

**1. CPU-GPU Synchronization Overhead:**

Nsight Systems reveals when the GPU is idle waiting for CPU commands:
```
[CPU Thread]    ──X───────X───────X────  (Long gaps = slow Python)
[GPU Kernels]   ─────●────────●─────●──  (GPU waiting for work)
                      ↑ Idle gap
```

**Solution:** Reduce Python overhead, use asynchronous operations, batch API calls

**2. Kernel Launch Overhead:**

Many small kernels create launch overhead:
```
[CUDA API]      ─┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬─  (Frequent launches)
[GPU Kernels]   ─●●●●●●●●●●●●●●●●●●●─  (Each kernel too short)
```

**Solution:** Use kernel fusion, torch.compile, or CUDA graphs

**3. Memory Transfer Bottlenecks:**

Data transfer overlapping (or lack thereof):
```
# Bad: Sequential transfer and compute
[Memory H2D]    ▬▬▬▬▬─────────────────
[GPU Kernel]    ─────●●●●●─────────────
[Memory D2H]    ──────────▬▬▬▬▬───────

# Good: Overlapped with CUDA streams
[Memory H2D]    ▬▬▬▬▬▬▬▬▬▬▬▬▬────────
[GPU Kernel]    ───●●●●●●●●●●●●●──────
                  ↑ Overlapped!
```

### NVTX Annotations for Custom Ranges

Use NVTX (NVIDIA Tools Extension) to label code regions:

```python
import torch

# Python API (requires torch.cuda.nvtx)
with torch.cuda.nvtx.range("data_loading"):
    batch = next(dataloader)

with torch.cuda.nvtx.range("forward_pass"):
    output = model(batch)

with torch.cuda.nvtx.range("backward_pass"):
    loss.backward()
```

These annotations appear in the Nsight Systems timeline:
```
[NVTX Ranges]   ──[data_loading]──[forward_pass]──[backward_pass]──
```

### Multi-GPU Profiling

Nsight Systems supports multi-GPU and multi-node analysis:

```bash
# Profile distributed training
nsys profile -o multi_gpu \
    --trace cuda,nvtx,mpi \
    python -m torch.distributed.launch \
        --nproc_per_node=8 train.py
```

Timeline shows all GPUs simultaneously:
```
[GPU 0 Kernels]  ●●●●────●●●●────●●●●
[GPU 1 Kernels]  ─●●●●────●●●●────●●●
[NCCL AllReduce] ─────▬▬▬▬────▬▬▬▬────  (Cross-GPU communication)
```

---

## Section 2: Nsight Compute - Detailed Kernel Metrics and Roofline Analysis

### What is Nsight Compute?

While Nsight Systems shows *when* kernels run, Nsight Compute explains *why* they're slow. It provides per-kernel performance metrics, roofline analysis, warp execution statistics, and memory access patterns.

**Key Features:**
- Detailed kernel profiling (occupancy, memory throughput, instruction throughput)
- Roofline model (compute-bound vs memory-bound analysis)
- Warp execution statistics (divergence, stalls)
- Source code correlation (SASS assembly mapping)
- Baseline comparison (compare kernel versions)

From [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (accessed 2025-11-14):
> "Nsight Compute inserts libraries to collect performance metrics from the GPU, and uses pre-defined rules to identify optimization opportunities."

### Installing and Running Nsight Compute

**Installation:**
```bash
# Download from NVIDIA Developer website
# https://developer.nvidia.com/nsight-compute

# Or included with CUDA Toolkit
which ncu  # /usr/local/cuda/bin/ncu

# Verify installation
ncu --version
```

**Basic Kernel Profiling:**
```bash
# Profile specific kernels (regex match)
ncu --kernel-name "gemm" python train.py

# Full profiling (all metrics, slow but comprehensive)
ncu -o kernel_profile --set full python train.py

# Lightweight profiling (faster, fewer metrics)
ncu -o kernel_profile --set default python train.py

# Profile first 5 invocations of each kernel
ncu --launch-count 5 python train.py
```

**Opening Results:**
```bash
# Launch GUI
ncu-ui kernel_profile.ncu-rep
```

### Understanding Kernel Metrics

**1. Occupancy:**

Percentage of maximum active warps achieved:
```
Theoretical Occupancy: 75%  (24 warps out of 32 max per SM)
Achieved Occupancy:    45%  (actual average during execution)

Analysis:
- Register usage: 128 registers/thread (limits to 8 warps/SM)
- Shared memory:  32KB (no limit, 48KB available)
→ Optimization: Reduce register usage to increase occupancy
```

**2. Memory Throughput:**

```
DRAM Bandwidth Utilization: 40%
L2 Cache Hit Rate:          85%
L1 Cache Hit Rate:          60%

Analysis:
- Memory-bound kernel
- Good L2 caching, room for improvement in L1
→ Optimization: Improve memory access patterns (coalescing)
```

**3. Compute Throughput:**

```
SM Utilization:           65%
Tensor Core Utilization:  20%  ← Low!
FP32 Pipeline Utilization: 45%

Analysis:
- Tensor Cores underutilized despite FP16 data
→ Optimization: Ensure matmul sizes are multiples of 8/16 for Tensor Cores
```

### Roofline Analysis

Roofline model visualizes kernel performance limits:

```
        ↑ Performance (TFLOPS)
        │
    100 │     ●────────────────  (Compute Bound Ceiling)
        │    ╱ ●
     50 │   ╱   ●  (Your kernel)
        │  ╱     ●
     10 │ ╱       ●
        │╱─────────●──────────  (Memory Bound Ceiling)
        └────────────────────→
           Arithmetic Intensity (FLOPS/Byte)

Analysis:
- Kernel is below both ceilings → not hitting hardware limits
- Below memory bound line → memory access inefficient
→ Optimization: Improve memory coalescing first
```

### Warp Execution Analysis

Nsight Compute shows why warps stall:

```
Warp Stall Breakdown:
- Memory Throttle:     40%  (DRAM accesses slow)
- Execution Stall:     25%  (waiting for ALU)
- Synchronization:     15%  (barrier waits)
- Not Selected:        10%  (scheduler chose other warps)
- Instruction Fetch:    5%
- Other:                5%

Analysis:
- High memory throttle → memory-bound kernel
→ Optimization: Reduce DRAM accesses via caching/reuse
```

### Source Code Correlation

Nsight Compute maps metrics to actual code:

```python
# Python code
def matmul_kernel(A, B, C):
    C = torch.matmul(A, B)  # ← Mapped to CUDA kernel

# Nsight Compute shows SASS assembly:
# Line 42: LDG.E [R2], [R4+0x1000]  ← 60% of DRAM accesses
# Line 43: HMMA.16816 R0, R2, R6    ← Tensor Core instruction
# Line 44: STG.E [R8], R0           ← 30% of DRAM accesses
```

### Baseline Comparison

Compare kernel performance before/after optimization:

```bash
# Profile baseline
ncu -o baseline --set full python train_v1.py

# Profile optimized version
ncu -o optimized --set full python train_v2.py

# Compare in GUI
ncu-ui --import baseline.ncu-rep optimized.ncu-rep
```

Comparison shows:
```
Metric                  Baseline    Optimized    Change
---------------------------------------------------------
Kernel Duration         2.5ms       1.2ms        -52%  ✓
DRAM Throughput         150 GB/s    280 GB/s     +87%  ✓
Occupancy               45%         68%          +51%  ✓
Tensor Core Util        20%         85%          +325% ✓✓
```

---

## Section 3: PyTorch Profiler - Framework-Level Profiling

### What is PyTorch Profiler?

PyTorch Profiler provides easy, Python-integrated profiling without leaving your training script. It captures operator-level timing, memory usage, and generates Chrome traces or TensorBoard visualizations.

From [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (accessed 2025-11-14):
> "PyTorch profiler is enabled through the context manager and accepts activities (CPU, CUDA, XPU), record_shapes, and profile_memory parameters."

**Key Features:**
- Context manager API (easy to add to existing code)
- Operator-level timing (torch.matmul, conv2d, etc.)
- Memory profiling (allocations per operator)
- Stack trace analysis (map performance to Python source)
- TensorBoard integration (visualize in browser)
- CUDA event recording (precise GPU timing)

### Basic Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

model = MyModel().cuda()
inputs = torch.randn(32, 3, 224, 224).cuda()

# Profile forward pass
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("forward_pass"):
        output = model(inputs)

# Print table
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))
```

**Output:**
```
---------------------------------  ------------  ------------  ------------
                             Name    CUDA time   CPU time      # of Calls
---------------------------------  ------------  ------------  ------------
                  forward_pass       45.2ms        52.3ms             1
                     aten::conv2d   25.8ms        28.1ms            20
                aten::convolution   25.7ms        27.9ms            20
               aten::_convolution   25.6ms        27.8ms            20
         aten::cudnn_convolution   25.4ms        27.5ms            20
                 aten::batch_norm    8.2ms         9.1ms            20
     aten::_batch_norm_impl_index    8.1ms         8.9ms            20
          aten::native_batch_norm    7.9ms         8.6ms            20
                       aten::relu    2.1ms         2.3ms            21
                     aten::matmul    1.8ms         2.0ms             1
---------------------------------  ------------  ------------  ------------
CUDA time total: 45.2ms
CPU time total:  52.3ms
```

### Profiling Memory Usage

```python
with profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(inputs)

print(prof.key_averages().table(
    sort_by="self_cpu_memory_usage",
    row_limit=10
))
```

**Output:**
```
---------------------------------  ------------  ------------
                             Name       CPU Mem  Self CPU Mem
---------------------------------  ------------  ------------
                       aten::empty      94.79 MB      94.79 MB
     aten::max_pool2d_with_indices  11.48 MB      11.48 MB
                       aten::addmm      19.53 KB      19.53 KB
               aten::empty_strided       572 B         572 B
                     aten::resize_       240 B         240 B
                         aten::abs       480 B         240 B
---------------------------------  ------------  ------------
```

### Recording Custom Ranges

Use `record_function` to label code sections:

```python
from torch.profiler import record_function

# Training loop
for epoch in range(num_epochs):
    with record_function(f"epoch_{epoch}"):

        # Data loading
        with record_function("data_loading"):
            for batch in dataloader:
                inputs, targets = batch

        # Forward
        with record_function("forward"):
            outputs = model(inputs)

        # Loss
        with record_function("loss_computation"):
            loss = criterion(outputs, targets)

        # Backward
        with record_function("backward"):
            loss.backward()

        # Optimizer
        with record_function("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()
```

These labels appear in profiler output and TensorBoard visualizations.

### Export to Chrome Trace

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

# Export trace
prof.export_chrome_trace("trace.json")
```

Open in Chrome at `chrome://tracing`:
- Timeline view of all operators
- Zoom in to microsecond resolution
- See CPU-GPU parallelism
- Identify synchronization points

### Profiling Training Loops

For long training runs, use `schedule` to control when profiling happens:

```python
from torch.profiler import profile, schedule

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10
    ))
    prof.export_chrome_trace(f"trace_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=5,      # Skip first 5 steps
        warmup=1,    # Warmup for 1 step (discard results)
        active=3,    # Profile for 3 steps
        repeat=2     # Repeat cycle 2 times
    ),
    on_trace_ready=trace_handler
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()  # Signal profiler to advance
```

**Schedule visualization:**
```
Steps:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        ─  ─  ─  ─  ─  W  ●  ●  ●  ─  ─  W  ●  ●  ●
        └──wait──┘  ↑  └─active─┘  └──wait──┘
                   warmup
```

---

## Section 4: TensorBoard Profiler - Visual Analysis and Input Pipeline Debugging

### What is TensorBoard Profiler?

TensorBoard Profiler provides browser-based visualization of PyTorch Profiler data, with specialized views for op profiles, kernel stats, memory timelines, and input pipeline analysis.

From [TensorBoard Profiler on Vertex AI](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-11-14):
> "TensorBoard Profiler provides detailed performance metrics and API debugging via a user interface... offering insights into CPU-GPU interactions, memory usage, and input pipeline efficiency."

**Key Features:**
- **Overview Page:** High-level performance summary with recommendations
- **Operator View:** Time breakdown by operator type
- **Kernel View:** GPU kernel execution statistics
- **Trace Viewer:** Chrome-style timeline (same as chrome://tracing)
- **Memory Viewer:** Allocation timeline and peak usage
- **Input Pipeline Analyzer:** Data loading bottleneck detection

### Setting Up TensorBoard Profiler

**Installation:**
```bash
pip install torch-tb-profiler tensorboard
```

**Profiling for TensorBoard:**
```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Profile with TensorBoard handler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./tensorboard_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

**Launch TensorBoard:**
```bash
tensorboard --logdir=./tensorboard_logs
# Open browser to http://localhost:6006
```

### Overview Page

The overview provides performance summary and recommendations:

```
Performance Summary:
─────────────────────
GPU Utilization:        67.2%  (Target: >80%)
Step Time:              42.5ms
  - Kernel Time:        28.6ms  (67.3%)
  - Data Loading:       8.2ms   (19.3%)
  - Host Overhead:      5.7ms   (13.4%)

Recommendations:
1. Low GPU utilization detected
   → Increase batch size or reduce CPU preprocessing
2. Data loading taking 19.3% of step time
   → Increase num_workers in DataLoader
3. Kernel time is 67% of step, but GPU util only 67%
   → Check for small kernel launches (use kernel fusion)
```

### Operator View

Breaks down time by operator type:

```
Operator                Self Time    Total Time    Calls    % of Total
───────────────────────────────────────────────────────────────────────
aten::conv2d            2.5ms        25.8ms        20       60.7%
  └ aten::convolution   0.2ms        23.3ms        20       54.8%
    └ cudnn_convolution 23.1ms       23.1ms        20       54.3%

aten::batch_norm        0.8ms        8.2ms         20       19.3%
  └ native_batch_norm   7.4ms        7.4ms         20       17.4%

aten::relu              2.1ms        2.1ms         21        4.9%
aten::matmul            1.8ms        1.8ms          1        4.2%
───────────────────────────────────────────────────────────────────────
Total:                  42.5ms
```

**Input Shapes View:**

See operator time broken down by input shapes:

```
aten::conv2d:
  [32, 64, 56, 56], [64, 64, 3, 3]  →  8.2ms   (5 calls)
  [32, 128, 28, 28], [128, 128, 3, 3] →  6.8ms   (4 calls)
  [32, 256, 14, 14], [256, 256, 3, 3] →  5.5ms   (3 calls)
  [32, 512, 7, 7], [512, 512, 3, 3]   →  5.3ms   (2 calls)
```

### Kernel View

Shows GPU kernel statistics:

```
Kernel Name                          Time    Calls   Avg Time   Occupancy
─────────────────────────────────────────────────────────────────────────
cudnn_infer_v8_conv_fwd              15.2ms  20      0.76ms     75.2%
batch_norm_collect_statistics        4.1ms   20      0.21ms     68.4%
batch_norm_transform_input           3.3ms   20      0.17ms     72.1%
elementwise_kernel_relu              2.1ms   21      0.10ms     45.8%  ← Low!
gemm_16x16x16_tn                     1.8ms   1       1.80ms     82.3%
─────────────────────────────────────────────────────────────────────────

Analysis:
- relu kernel has low occupancy (45.8%)
- Could be fused with batch_norm for better performance
```

### Trace Viewer

Timeline visualization in TensorBoard (same as Chrome trace):

```
[CPU Thread]     ──┬───────┬───────┬───────┬──
                   │       │       │       │
[CUDA Kernel]    ──●───────●───────●───────●──
                   ↑
                   Click to see kernel details:
                   - Duration: 0.76ms
                   - Name: cudnn_infer_v8_conv_fwd
                   - Input shapes: [32, 64, 56, 56]
```

### Memory Viewer

Tracks memory allocations over time:

```
        ↑ Memory (GB)
    4.0 │     ╱─────────╲
        │    ╱           ╲
    3.0 │   ╱             ╲
        │  ╱               ╲
    2.0 │ ╱                 ╲
        │╱                   ╲
    1.0 │                     ╲
        └──────────────────────→ Time
           ↑                  ↑
         Forward           Backward
         activations       gradients

Peak Memory: 3.8GB at step 42
Recommendation: Reduce batch size or use gradient checkpointing
```

### Input Pipeline Analyzer

Identifies data loading bottlenecks:

```
Input Pipeline Analysis:
────────────────────────
Step Time Breakdown:
  - Device Compute:     28.6ms  (67.3%)  ← GPU working
  - Host Compute:        5.7ms  (13.4%)  ← CPU overhead
  - Input:               8.2ms  (19.3%)  ← Data loading
    └ DataLoader:        8.2ms
      └ Preprocessing:   6.5ms  ← Bottleneck!
      └ Disk I/O:        1.7ms

Recommendations:
1. DataLoader preprocessing taking 6.5ms
   → Increase num_workers from 4 to 8
2. GPU is idle 19.3% of the time waiting for data
   → Consider using DALI (GPU-accelerated data loading)
3. prefetch_factor=2 might be too low
   → Try prefetch_factor=4 for better overlap
```

### Best Practices for TensorBoard Profiler

**1. Profile Multiple Steps:**
```python
# Don't profile just 1 step - variance is high
schedule(wait=5, warmup=1, active=10, repeat=3)  # 30 steps total
```

**2. Use with_stack for Python Traces:**
```python
with profile(
    ...,
    with_stack=True  # Enables Python call stack recording
) as prof:
    train_step()
```

Shows Python source in TensorBoard:
```
Stack Trace:
  train.py:42  train_step()
  model.py:128  forward()
  conv.py:56  conv2d()
  → aten::conv2d (25.8ms)
```

**3. Profile Both Training and Validation:**
```python
# Training loop
with profile(...) as train_prof:
    for batch in train_loader:
        train_step(batch)
        train_prof.step()

# Validation loop
with profile(...) as val_prof:
    for batch in val_loader:
        val_step(batch)
        val_prof.step()
```

Compare training vs validation performance in TensorBoard.

---

## Section 5: Identifying Bottlenecks - Kernel Time, Memory Bandwidth, CPU Wait

### The Bottleneck Hierarchy

Performance issues fall into distinct categories requiring different solutions:

```
1. CPU-bound
   → GPU waiting for CPU to launch kernels
   → Solution: Reduce Python overhead, use async ops

2. Memory-bound
   → Kernel limited by DRAM bandwidth
   → Solution: Improve memory access patterns, use caching

3. Compute-bound
   → Kernel limited by ALU/Tensor Core throughput
   → Solution: Increase arithmetic intensity, use better algorithms

4. Launch-bound
   → Too many small kernels (launch overhead dominates)
   → Solution: Kernel fusion, torch.compile, CUDA graphs

5. I/O-bound
   → Waiting for data from disk/network
   → Solution: Faster storage, prefetching, caching
```

### Detecting CPU-Bound Workloads

**Nsight Systems signs:**
```
[CPU Thread]    ──X──────X──────X──────  (Long gaps)
[CUDA Kernel]   ─────●──────●──────●───  (GPU idle waiting)
                     ↑ Idle gap = CPU overhead
```

**PyTorch Profiler signs:**
```
CPU time total:   85.2ms
CUDA time total:  42.5ms
                  ↑ GPU only working 50% of the time!
```

**Solutions:**
```python
# 1. Reduce Python overhead with torch.compile
model = torch.compile(model, mode="reduce-overhead")

# 2. Use non-blocking data transfers
inputs = inputs.to(device, non_blocking=True)

# 3. Batch operations to reduce kernel launches
# Bad: 100 small operations
for i in range(100):
    x = x + 1

# Good: Single fused operation
x = x + 100

# 4. Use async operations
with torch.cuda.stream(stream):
    # Async work
    output = model(inputs)
# Don't synchronize until needed
```

### Detecting Memory-Bound Kernels

**Nsight Compute signs:**
```
Memory Throughput:       280 GB/s  (out of 900 GB/s theoretical)
DRAM Bandwidth Util:     31%       ← Low!
Compute Throughput:      15 TFLOPS (out of 312 TFLOPS)
Compute Util:            4.8%      ← Very low!

Roofline Position:  Below memory-bound line
→ Kernel is memory-bound but not achieving peak bandwidth
```

**Warp stalls:**
```
Warp Stall Reasons:
- Memory Throttle:  65%  ← Waiting for DRAM
- Execution Stall:  10%
- Synchronization:   5%
```

**Solutions:**
```python
# 1. Improve memory coalescing
# Bad: Strided access
for i in range(N):
    x = input[i * stride]  # Non-coalesced

# Good: Contiguous access
for i in range(N):
    x = input[i]  # Coalesced

# 2. Use shared memory for reuse
# Tile matrix multiply to reuse data in shared memory

# 3. Increase arithmetic intensity
# Fuse operations to reduce memory traffic
# Bad: x = a + b; y = x * c  (3 memory ops)
# Good: y = (a + b) * c     (2 memory ops, fused)

# 4. Use Tensor Cores (higher compute intensity)
# Convert to FP16/BF16 for Tensor Core usage
model = model.half()  # or .bfloat16()
```

### Detecting Compute-Bound Kernels

**Nsight Compute signs:**
```
Compute Throughput:      280 TFLOPS (out of 312 TFLOPS)
Compute Util:            89.7%      ← High!
DRAM Bandwidth Util:     25%        ← Low (not the bottleneck)

Tensor Core Util:        92.3%      ← Tensor Cores saturated

Roofline Position:  At compute-bound ceiling
→ Kernel hitting hardware limits
```

**Solutions:**
```python
# 1. Use lower precision (if accuracy allows)
# FP32 → FP16: 2× faster on Tensor Cores
# FP16 → FP8: Another 2× faster on H100

# 2. Optimize algorithm (reduce FLOPs)
# Use Flash Attention instead of standard attention
# Use efficient transformer variants

# 3. Ensure Tensor Core usage
# Matmul dimensions should be multiples of 8/16
# Bad:  [32, 127] @ [127, 64]  (127 not multiple of 8)
# Good: [32, 128] @ [128, 64]  (128 = 16*8)

# 4. Check for the right dtype
with torch.cuda.amp.autocast():
    output = model(inputs)  # Uses FP16/BF16 for Tensor Cores
```

### Detecting Launch-Bound Workloads

**Nsight Systems signs:**
```
[CUDA API]      ─┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬─  (Many cudaLaunchKernel calls)
[CUDA Kernel]   ─●●●●●●●●●●●●●●●●─  (Each kernel < 50μs)
                  ↑ Launch overhead > kernel time
```

**PyTorch Profiler signs:**
```
Number of operator calls: 15,000 per step
Average kernel duration:  35μs
Launch overhead:          ~5μs per kernel
→ 25% of time is launch overhead!
```

**Solutions:**
```python
# 1. Use torch.compile (automatic kernel fusion)
model = torch.compile(model, mode="max-autotune")

# 2. Use CUDA graphs (eliminate launch overhead)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

# Replay graph (single launch for entire graph)
g.replay()

# 3. Manual kernel fusion with custom CUDA/Triton kernels
# Fuse elementwise ops: relu(batch_norm(conv(x)))
# Into single kernel instead of 3 separate launches
```

### Detecting I/O-Bound Workloads

**Nsight Systems signs:**
```
[CPU Thread]    ──────────X────────────X──  (Long gaps in I/O)
[GPU Kernel]    ──●──────────●──────────●─  (GPU idle)
                    ↑ Waiting for data
```

**TensorBoard Input Pipeline Analyzer:**
```
Step Time: 125ms
  - Device Compute: 45ms  (36%)
  - Input:          80ms  (64%)  ← I/O bottleneck!
    └ DataLoader:   80ms
      └ Disk Read:  70ms
      └ Decode:     10ms
```

**Solutions:**
```python
# 1. Increase DataLoader workers
DataLoader(dataset, batch_size=32, num_workers=8)  # More parallel I/O

# 2. Use persistent workers (avoid respawning processes)
DataLoader(..., persistent_workers=True)

# 3. Use faster storage (NVMe SSD instead of HDD)
# 4. Cache dataset in RAM or Local SSD
# 5. Prefetch next batch while GPU works
DataLoader(..., prefetch_factor=4)

# 6. Use DALI for GPU-accelerated data loading
import nvidia.dali as dali
# Data decoding/augmentation on GPU
```

---

## Section 6: CUDA Event Recording - Timing GPU Operations

### What are CUDA Events?

CUDA events provide precise timing of GPU operations without CPU-GPU synchronization overhead. They're placed in CUDA streams and record timestamps when reached during GPU execution.

**Why Use CUDA Events:**
- **Accurate GPU timing** - No CPU overhead, pure device time
- **Asynchronous** - Don't block CPU while GPU works
- **Stream-aware** - Measure operations across multiple streams
- **Minimal overhead** - Event recording is ~1-2μs

### Basic CUDA Event Usage

```python
import torch

# Create events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Record events
start.record()
output = model(inputs)  # GPU work
end.record()

# Wait for GPU to finish
torch.cuda.synchronize()

# Get elapsed time (in milliseconds)
elapsed = start.elapsed_time(end)
print(f"Kernel time: {elapsed:.3f} ms")
```

### Timing Specific Operations

```python
# Time forward pass only
start_fwd = torch.cuda.Event(enable_timing=True)
end_fwd = torch.cuda.Event(enable_timing=True)

start_fwd.record()
output = model(inputs)
end_fwd.record()

# Time backward pass
start_bwd = torch.cuda.Event(enable_timing=True)
end_bwd = torch.cuda.Event(enable_timing=True)

start_bwd.record()
loss = criterion(output, targets)
loss.backward()
end_bwd.record()

torch.cuda.synchronize()

print(f"Forward:  {start_fwd.elapsed_time(end_fwd):.3f} ms")
print(f"Backward: {start_bwd.elapsed_time(end_bwd):.3f} ms")
```

### Multi-Stream Timing

CUDA events can measure work across multiple streams:

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record(stream=stream1)

# Work on stream 1
with torch.cuda.stream(stream1):
    x1 = model1(input1)

# Work on stream 2
with torch.cuda.stream(stream2):
    x2 = model2(input2)

# Record end event on stream1 (will wait for both streams)
end.record(stream=stream1)
torch.cuda.synchronize()

print(f"Total time: {start.elapsed_time(end):.3f} ms")
```

### Benchmarking with CUDA Events

```python
def benchmark_kernel(func, inputs, num_iterations=100, warmup=10):
    """Benchmark GPU operation with CUDA events"""

    # Warmup
    for _ in range(warmup):
        _ = func(inputs)

    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        _ = func(inputs)
    end.record()

    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    avg_time = elapsed / num_iterations

    return avg_time

# Usage
inputs = torch.randn(32, 3, 224, 224).cuda()
avg_time = benchmark_kernel(model, inputs)
print(f"Average kernel time: {avg_time:.3f} ms")
```

### CUDA Event Gotchas

**1. Synchronization is required to read timing:**
```python
start.record()
output = model(inputs)
end.record()

# Wrong: Reading before synchronization
time = start.elapsed_time(end)  # May return 0 or incorrect value!

# Correct: Synchronize first
torch.cuda.synchronize()
time = start.elapsed_time(end)  # Correct
```

**2. Events are stream-specific:**
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record(stream=stream1)
with torch.cuda.stream(stream2):
    output = model(inputs)
end.record(stream=stream2)  # Different stream!

# elapsed_time will only measure stream2 work
```

**3. Default stream synchronization:**
```python
# Default stream synchronizes with all other streams
start.record()  # On default stream
with torch.cuda.stream(my_stream):
    output = model(inputs)
end.record()  # Back to default stream

# This works because default stream waits for my_stream
```

### Comparing with PyTorch Profiler

CUDA events vs PyTorch Profiler:

| Feature | CUDA Events | PyTorch Profiler |
|---------|-------------|------------------|
| Precision | Microsecond | Microsecond |
| Overhead | Minimal (~2μs) | Moderate (~5-10%) |
| Ease of use | Manual | Automatic |
| Detailed breakdown | No | Yes (per-operator) |
| Visualization | No | Yes (TensorBoard) |
| Production use | ✓ (low overhead) | × (too slow) |

**Best practice:** Use CUDA events for production monitoring, PyTorch Profiler for development/debugging.

---

## Section 7: GCloud Integration - Profiling on Compute Engine and Vertex AI

### Profiling on GCP Compute Engine

**Setting up GPU instance:**
```bash
# Create GPU instance
gcloud compute instances create profiling-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE

# SSH into instance
gcloud compute ssh profiling-vm --zone=us-central1-a

# Verify GPU
nvidia-smi
```

**Install profiling tools:**
```bash
# Nsight Systems
wget https://developer.download.nvidia.com/devtools/nsight-systems/...
sudo dpkg -i nsight-systems-*.deb

# Nsight Compute
wget https://developer.download.nvidia.com/devtools/nsight-compute/...
sudo dpkg -i nsight-compute-*.deb

# PyTorch Profiler (included with PyTorch)
pip install torch-tb-profiler tensorboard
```

**Remote profiling workflow:**
```bash
# On GCP VM: Profile training
nsys profile -o training_profile \
    --trace cuda,cudnn,cublas,nvtx \
    --cuda-memory-usage true \
    python train.py

# Download profile to local machine
gcloud compute scp profiling-vm:~/training_profile.nsys-rep \
    ./local_profile.nsys-rep --zone=us-central1-a

# Open in local Nsight Systems GUI
nsys-ui local_profile.nsys-rep
```

### Profiling on Vertex AI

Vertex AI provides managed training with built-in TensorBoard Profiler support.

**Enable TensorBoard Profiler in training code:**

```python
# training_script.py
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Vertex AI TensorBoard integration
from google.cloud import aiplatform
from torch.profiler import tensorboard_trace_handler

# Initialize Vertex AI
aiplatform.init(
    project='my-project',
    location='us-central1'
)

# Create TensorBoard instance (done once)
tensorboard = aiplatform.Tensorboard.create(
    display_name='profiling-tensorboard',
    project='my-project',
    location='us-central1'
)

# In training script: Enable profiling
def train():
    model = MyModel().cuda()

    # Profile with TensorBoard handler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=5, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(
            "./tensorboard_logs"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, batch in enumerate(dataloader):
            train_step(batch)
            prof.step()

if __name__ == '__main__':
    train()
```

**Submit Vertex AI training job:**

```python
# submit_job.py
from google.cloud import aiplatform

aiplatform.init(
    project='my-project',
    location='us-central1',
    staging_bucket='gs://my-bucket'
)

job = aiplatform.CustomTrainingJob(
    display_name='profiling-job',
    script_path='training_script.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['torch-tb-profiler'],
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1
)

# Enable TensorBoard
job.run(
    tensorboard=tensorboard.resource_name,
    service_account='my-service-account@project.iam.gserviceaccount.com'
)
```

**Access TensorBoard in Vertex AI:**

```bash
# Via gcloud CLI
gcloud ai tensorboards describe TENSORBOARD_ID \
    --region=us-central1

# Or via Console:
# https://console.cloud.google.com/vertex-ai/experiments/tensorboard
```

**TensorBoard Profiler on Vertex AI provides:**
- **Automatic profiling** - No manual setup needed
- **Cloud storage** - Profiles stored in GCS
- **Team access** - Shareable TensorBoard instances
- **Cost tracking** - Integrated with GCP billing

### Best Practices for GCP Profiling

**1. Use preemptible GPUs for profiling experiments:**
```bash
gcloud compute instances create profiling-vm \
    --preemptible \  # 70% cheaper
    --accelerator=type=nvidia-tesla-v100,count=1 \
    ...
```

**2. Profile on same hardware as production:**
```python
# If production uses A100, profile on A100
# Kernel performance varies significantly by architecture
```

**3. Store profiles in Cloud Storage:**
```bash
# Upload profiles to GCS for team access
gsutil cp *.nsys-rep gs://my-bucket/profiles/
gsutil cp *.ncu-rep gs://my-bucket/profiles/
```

**4. Use Vertex AI Experiments for tracking:**
```python
from google.cloud import aiplatform

aiplatform.init(
    experiment='profiling-experiments',
    experiment_description='GPU profiling runs'
)

# Log profiling results
aiplatform.log_metrics({
    'gpu_utilization': 87.2,
    'step_time_ms': 42.5,
    'kernel_time_ms': 28.6
})
```

**5. Schedule profiling runs:**
```bash
# Use Cloud Scheduler for periodic profiling
gcloud scheduler jobs create http profiling-job \
    --schedule="0 2 * * *" \  # Daily at 2 AM
    --uri="https://us-central1-aiplatform.googleapis.com/v1/projects/..."
```

From [Vertex AI TensorBoard Profiler Best Practices](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) (accessed 2025-11-14):
> "Enable Cloud Profiler for debugging model training performance... The Profiler dashboard provides an overview page, operator views, kernel views, and a trace view similar to Chrome's tracing."

---

## Section 8: arr-coc-0-1 Profiling Workflow - End-to-End Analysis

### arr-coc-0-1 Architecture Profiling

The arr-coc-0-1 project (ARR-COC-VIS MVP from Platonic Dialogue 46) has specific profiling needs due to its Vervaekean architecture with dynamic token allocation.

**Key components to profile:**
1. **Texture array generation** - 13-channel preprocessing
2. **Knowing module** - Three relevance scorers (Propositional, Perspectival, Participatory)
3. **Balancing module** - Opponent processing across tensions
4. **Attending module** - Token budget allocation (64-400 tokens per patch)
5. **Visual encoder** - Qwen3-VL integration with variable LOD

**Performance targets:**
- Texture generation: <50ms per image
- Relevance scoring: <20ms per batch
- Token allocation: <10ms per batch
- Total forward pass: <200ms on A100

### Initial Profiling Setup

**Project structure:**
```
arr-coc-0-1/
├── arr_coc/
│   ├── knowing.py      # Relevance scorers
│   ├── balancing.py    # Opponent processing
│   ├── attending.py    # Token allocation
│   └── model.py        # Full model
├── profiling/
│   ├── profile_knowing.py
│   ├── profile_full_model.py
│   └── analyze_results.py
└── tensorboard_logs/
```

**Profiling script:**

```python
# profiling/profile_full_model.py
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from arr_coc.model import ARRCOCModel
from arr_coc.texture import TextureArrayGenerator

def profile_arr_coc():
    device = "cuda"
    model = ARRCOCModel().to(device)
    texture_gen = TextureArrayGenerator()

    # Sample inputs
    images = torch.randn(8, 3, 224, 224).to(device)
    queries = torch.randn(8, 768).to(device)

    # Profile with detailed annotations
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "../tensorboard_logs/arr_coc_profile"
        )
    ) as prof:

        # Texture generation
        with record_function("texture_generation"):
            texture_arrays = texture_gen(images)

        # Knowing: Relevance scoring
        with record_function("knowing_module"):
            with record_function("propositional_scorer"):
                prop_scores = model.knowing.propositional(texture_arrays)

            with record_function("perspectival_scorer"):
                persp_scores = model.knowing.perspectival(texture_arrays)

            with record_function("participatory_scorer"):
                partic_scores = model.knowing.participatory(
                    queries, texture_arrays
                )

        # Balancing: Opponent processing
        with record_function("balancing_module"):
            balanced_scores = model.balancing(
                prop_scores, persp_scores, partic_scores
            )

        # Attending: Token allocation
        with record_function("attending_module"):
            token_budgets = model.attending(balanced_scores)

        # Full forward pass
        with record_function("full_forward"):
            output = model(images, queries)

        prof.step()

    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30
    ))

if __name__ == "__main__":
    profile_arr_coc()
```

### Expected Profiling Results

**First pass profiling (baseline, unoptimized):**

```
---------------------------------  ------------  ------------  ------------
                             Name    CUDA time    CPU time      Calls
---------------------------------  ------------  ------------  ------------
                  full_forward       285.3ms       298.4ms           1
              texture_generation     68.2ms        72.1ms            1
                knowing_module       127.4ms       135.2ms           1
   propositional_scorer (entropy)    45.3ms        48.1ms            1
     perspectival_scorer (sobel)     38.6ms        41.2ms            1
  participatory_scorer (attn)        43.5ms        45.9ms            1
              balancing_module       22.1ms        23.8ms            1
              attending_module       18.3ms        19.5ms            1
                     aten::conv2d    52.4ms        55.1ms           26
                aten::convolution    52.2ms        54.9ms           26
               aten::_convolution    51.9ms        54.6ms           26
         aten::cudnn_convolution    51.7ms        54.3ms           26
                 aten::batch_norm   15.8ms        16.9ms           26
---------------------------------  ------------  ------------  ------------
Total time: 285.3ms
```

**Analysis:**
1. Texture generation (68ms) - Too slow! Target: <50ms
2. Knowing module (127ms) - Dominant component, needs optimization
3. Participatory scorer (43.5ms) - Cross-attention bottleneck

### Optimization 1: Fuse Texture Channels

**Problem:** Texture generation runs 13 separate convolutions

**Profiling reveals:**
```
[Nsight Systems Timeline]
Texture Generation:
  conv2d (RGB→RGB)     5.2ms  ──●──
  conv2d (LAB→L)       4.8ms  ─────●──
  conv2d (LAB→A)       4.9ms  ────────●──
  conv2d (LAB→B)       4.7ms  ───────────●──
  conv2d (Sobel X)     6.1ms  ──────────────●──
  conv2d (Sobel Y)     6.2ms  ─────────────────●──
  ... (7 more)

Total: 13 sequential kernels = 68ms
```

**Solution:** Batch all convolutions

```python
# Before: 13 separate convolutions
class TextureArrayGenerator:
    def forward(self, images):
        rgb = self.rgb_conv(images)
        lab_l = self.lab_l_conv(rgb_to_lab(images))
        lab_a = self.lab_a_conv(rgb_to_lab(images))
        # ... 10 more separate calls

# After: Single batched convolution
class TextureArrayGeneratorOptimized:
    def forward(self, images):
        # Stack all 13 filters into single conv
        all_filters = torch.cat([
            self.rgb_filters,  # 3 filters
            self.lab_filters,  # 3 filters
            self.sobel_filters,  # 2 filters
            # ... rest
        ], dim=0)  # Shape: [13, 3, K, K]

        # Single convolution produces all 13 channels
        texture_array = F.conv2d(images, all_filters)
        return texture_array  # [B, 13, H, W]
```

**After optimization:**
```
Texture Generation:
  conv2d (batched 13ch)  12.4ms  ──●──  ← Single kernel!

Total: 12.4ms (5.5× faster)
```

### Optimization 2: torch.compile the Knowing Module

**Problem:** Three separate scorer calls with Python overhead

**Apply torch.compile:**
```python
# arr_coc/knowing.py
import torch
import torch.nn as nn

class KnowingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.propositional = ProposionalScorer()
        self.perspectival = PerspectivalScorer()
        self.participatory = ParticipatoryScorer()

    @torch.compile(mode="max-autotune")
    def forward(self, texture_arrays, queries):
        prop = self.propositional(texture_arrays)
        persp = self.perspectival(texture_arrays)
        partic = self.participatory(queries, texture_arrays)
        return prop, persp, partic
```

**Profiling after torch.compile:**
```
Before torch.compile:
  knowing_module         127.4ms
    propositional         45.3ms
    perspectival          38.6ms
    participatory         43.5ms

After torch.compile:
  knowing_module          68.2ms  (1.87× faster)
    compiled_forward      68.2ms  ← Single fused kernel
```

**Kernel fusion detected by Nsight Systems:**
```
Before:
[entropy_kernel]     ──●──
[sobel_x_kernel]     ─────●──
[sobel_y_kernel]     ────────●──
[cross_attn_qk]      ───────────●──
[cross_attn_v]       ──────────────●──

After torch.compile:
[fused_knowing]      ──●●●──  ← Fused kernels
```

### Optimization 3: Profile Memory Usage

**Identify memory allocations:**
```python
with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    output = model(images, queries)

print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=15
))
```

**Results:**
```
---------------------------------  ------------  ------------
                             Name    CUDA Mem   Self CUDA Mem
---------------------------------  ------------  ------------
                       aten::empty   3.2 GB        3.2 GB  ← Activations!
              texture_generation    856 MB          0 MB
                knowing_module      1.8 GB          0 MB
   propositional_scorer             612 MB        612 MB
     perspectival_scorer            584 MB        584 MB
  participatory_scorer              605 MB        605 MB
---------------------------------  ------------  ------------
Peak memory: 3.8 GB
```

**Solution: Gradient checkpointing for memory reduction**

```python
# arr_coc/model.py
import torch.utils.checkpoint as checkpoint

class ARRCOCModel(nn.Module):
    def forward(self, images, queries):
        # Checkpoint knowing module (recompute in backward)
        prop, persp, partic = checkpoint.checkpoint(
            self.knowing,
            images,
            queries,
            use_reentrant=False
        )

        balanced = self.balancing(prop, persp, partic)
        tokens = self.attending(balanced)
        output = self.realize(images, tokens)
        return output
```

**Memory after checkpointing:**
```
Peak memory: 2.1 GB (45% reduction)
Forward time: +15ms (gradient checkpointing overhead)
→ Tradeoff: Slower but fits larger batches
```

### Final Profiling Results

**After all optimizations:**

```
Component               Before    After     Speedup
─────────────────────────────────────────────────────
Texture Generation      68.2ms    12.4ms    5.5×
Knowing Module         127.4ms    68.2ms    1.87×
Balancing Module        22.1ms    18.3ms    1.21×
Attending Module        18.3ms    15.1ms    1.21×
Total Forward Pass     285.3ms   142.7ms    2.0×
Peak Memory             3.8 GB    2.1 GB    45% reduction
─────────────────────────────────────────────────────

GPU Utilization:        67% → 89%
Throughput:             28 img/s → 56 img/s (2× faster)
```

**Production deployment checklist:**
- ✓ Use torch.compile(mode="max-autotune") for knowing module
- ✓ Fuse texture convolutions into single batched kernel
- ✓ Enable gradient checkpointing for memory efficiency
- ✓ Profile on A100 (target hardware) before deploying
- ✓ Monitor with CUDA events in production for regression detection

### Continuous Profiling in Production

**Add lightweight profiling to training script:**

```python
# arr_coc/training.py
import torch

class ProductionProfiler:
    def __init__(self):
        self.step_times = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start_step(self):
        self.start_event.record()

    def end_step(self):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)
        self.step_times.append(elapsed)

    def report(self):
        import numpy as np
        times = np.array(self.step_times)
        return {
            'mean': times.mean(),
            'std': times.std(),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }

# In training loop
profiler = ProductionProfiler()
for step, batch in enumerate(dataloader):
    profiler.start_step()

    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    profiler.end_step()

    # Log every 100 steps
    if step % 100 == 0:
        stats = profiler.report()
        wandb.log({
            'step_time_mean': stats['mean'],
            'step_time_p99': stats['p99']
        })
```

**Benefits:**
- Detect performance regressions early
- Track improvements from optimization
- Alert on anomalies (sudden slowdowns)
- Minimal overhead (<0.5%)

---

## Sources

### Source Documents

**karpathy-deep-oracle existing knowledge:**
- [cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md) - torch.compile and JIT compilation
- [karpathy/practical-implementation/08-gpu-memory-debugging-vlm-2025-01-30.md](../karpathy/practical-implementation/08-gpu-memory-debugging-vlm-2025-01-30.md) - GPU memory debugging patterns

### Web Research

**NVIDIA Nsight Tools:**
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - Official product page (accessed 2025-11-14)
  - System-wide profiling, timeline visualization, multi-GPU support
  - Python backtrace sampling for deep learning

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - NVIDIA Documentation (accessed 2025-11-14)
  - Kernel-level metrics, roofline analysis, warp execution stats
  - Detailed profiling workflow and metric interpretation

**PyTorch Profiler:**
- [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Official PyTorch tutorial (accessed 2025-11-14)
  - Complete profiler API documentation
  - Memory profiling, stack traces, scheduling for long jobs

- [PyTorch Profiler with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) - PyTorch documentation (accessed 2025-11-14)
  - TensorBoard integration and visualization
  - Note: TensorBoard integration deprecated in favor of Perfetto/Chrome trace

**Google Cloud Platform:**
- [Vertex AI TensorBoard Profiler](https://cloud.google.com/vertex-ai/docs/training/tensorboard-profiler) - GCP Documentation (accessed 2025-11-14)
  - Cloud Profiler setup for Vertex AI training jobs
  - TensorBoard dashboard access and best practices
  - Multi-node profiling on GCP infrastructure

**Additional Resources:**
- Medium articles on PyTorch profiling workflows
- NVIDIA Developer Blog posts on Nsight Systems usage
- Stack Overflow discussions on profiler interpretation
- GitHub issues for PyTorch Profiler troubleshooting

### Code Examples

All code examples in this document are original implementations based on:
- PyTorch Profiler official API documentation
- NVIDIA Nsight Systems command-line reference
- arr-coc-0-1 project architecture (ARR-COC-VIS MVP)
- Standard GPU profiling best practices
