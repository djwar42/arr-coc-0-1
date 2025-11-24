# PyTorch Profiling Ecosystem: Tools, Architecture, and Production Strategies

**Complete guide to PyTorch profiling tools: torch.profiler + Kineto, NSight Systems/Compute, CUPTI dependencies, profiling without CUPTI, and production deployment patterns**

---

## Overview

PyTorch provides a rich profiling ecosystem with multiple tools operating at different abstraction levels. Understanding which tool to use when, and how they interoperate, is critical for production ML optimization.

**Key insight**: PyTorch profiler (torch.profiler) is NOT the only option. NSight Systems and NSight Compute can profile PyTorch workloads without CUPTI, making them essential alternatives when torch.profiler fails.

---

## Section 1: PyTorch Profiler & Kineto Architecture (~120 lines)

### What is torch.profiler?

From [PyTorch Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html) (accessed 2025-11-13):

> "PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference. Profiler's context manager API can be used to better understand what model operators are the most expensive, examine their input shapes and stack traces, study device kernel activity and visualize the execution trace."

**Architecture**:
```
torch.profiler (Python API)
    ↓
Kineto library (C++ backend)
    ↓
CUPTI (NVIDIA CUDA Profiling Tools Interface)
    ↓
GPU hardware counters
```

### Kineto: The Profiling Backend

From [PyTorch Kineto GitHub](https://github.com/pytorch/kineto) (accessed 2025-11-13):

**Kineto** is PyTorch's CPU+GPU profiling library that provides:
- **Low-overhead GPU timeline tracing** (central focus)
- **CPU profiling integration** (PyTorch operations)
- **Distributed training support** (multi-node traces)
- **CUPTI integration** for GPU kernel events

**Key Kineto features**:
```python
# Kineto provides:
# 1. GPU kernel launch/execution events
# 2. CUDA memory allocation/free events
# 3. CPU-side PyTorch operator execution
# 4. CUDA stream synchronization events
# 5. Python stack traces (optional)
```

### CUPTI Dependency: Why It's Required

From [PyTorch Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) (accessed 2025-11-13):

> "GPU kernels execute asynchronously, and GPU-side support is needed to create the trace. NVIDIA provides this visibility via the CUPTI library. Kineto is the subsystem within Profiler that interfaces with CUPTI."

**What Kineto needs from CUPTI**:
1. **GPU kernel launch events** (when kernel starts)
2. **GPU kernel execution events** (kernel duration on GPU)
3. **CUDA runtime API calls** (cudaMalloc, cudaMemcpy, etc.)
4. **GPU hardware counters** (optional, for advanced metrics)

**What breaks without CUPTI**:
- ❌ No GPU kernel timelines in trace
- ❌ No GPU memory events
- ❌ No GPU utilization metrics
- ✅ CPU-side profiling still works (PyTorch operators)
- ✅ Python stack traces still work

### Basic torch.profiler Usage

From [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (accessed 2025-11-13):

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Basic profiling with GPU
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(inputs)
    loss.backward()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")
```

**Advanced profiling with schedule**:
```python
from torch.profiler import profile, schedule, tensorboard_trace_handler

# Profile specific iterations
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True
) as prof:
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prof.step()  # Signal end of iteration
```

### Kineto Overhead Characteristics

From [PyTorch Kineto profiling overhead research](https://github.com/pytorch/kineto) (accessed 2025-11-13):

**Overhead benchmarks**:
- **CPU profiling alone**: ~2-5% overhead
- **GPU profiling (with CUPTI)**: ~5-10% overhead
- **Memory profiling**: Additional ~3-5% overhead
- **Python stack traces**: Additional ~10-15% overhead

**Production recommendation**: Profile for short bursts (3-5 iterations) rather than continuous profiling.

---

## Section 2: NSight Tools (CUPTI-Free Profiling) (~120 lines)

### NSight Systems: System-Level Timeline Profiling

**Key distinction**: NSight Systems does NOT require CUPTI for basic GPU profiling.

From [NVIDIA NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/) (accessed 2025-11-13):

**What NSight Systems profiles**:
1. **CUDA API calls** (cudaLaunchKernel, cudaMemcpy) - via CUDA driver intercept
2. **GPU kernel execution** (start/end times) - via GPU driver, NOT CUPTI
3. **CPU threads and processes** - OS-level tracing
4. **NVTX ranges** (if you annotate code with torch.cuda.nvtx.range)
5. **Memory transfers** (H2D, D2H, D2D)

**Usage for PyTorch**:
```bash
# Profile PyTorch training script
nsys profile --trace=cuda,nvtx,osrt python train.py

# Generate timeline GUI (trace.nsys-rep)
nsys-ui trace.nsys-rep
```

**Advantages over torch.profiler**:
- ✅ Works without CUPTI (uses GPU driver hooks)
- ✅ Lower overhead (~2-3% vs torch.profiler's 5-10%)
- ✅ System-wide view (CPU + GPU + network + disk I/O)
- ✅ Multi-process profiling (DDP training)
- ❌ No PyTorch operator names (just CUDA kernel names)
- ❌ No automatic iteration detection

**PyTorch + NSight Systems best practice**:
```python
# Add NVTX annotations for visibility
import torch.cuda.nvtx as nvtx

for epoch in range(num_epochs):
    nvtx.range_push(f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(dataloader):
        nvtx.range_push(f"Batch {batch_idx}")

        # Forward pass
        nvtx.range_push("Forward")
        output = model(data)
        loss = criterion(output, target)
        nvtx.range_pop()

        # Backward pass
        nvtx.range_push("Backward")
        loss.backward()
        nvtx.range_pop()

        optimizer.step()
        nvtx.range_pop()  # End batch
    nvtx.range_pop()  # End epoch
```

### NSight Compute: Kernel-Level Analysis

From [NVIDIA NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (accessed 2025-11-13):

**What NSight Compute does**:
- **Single kernel deep-dive** (not timeline profiling)
- **GPU hardware counters** (SM occupancy, memory bandwidth, warp efficiency)
- **Roofline analysis** (compute-bound vs memory-bound)
- **Source-level metrics** (per-line SASS/PTX analysis)

**CRITICAL: NSight Compute is standalone** - does NOT require CUPTI installation on target system.

**Usage**:
```bash
# Profile specific kernel
ncu --set full --target-processes all python train.py

# Profile specific kernel by name
ncu --kernel-name "volta_scudnn_winograd.*" python train.py

# Export metrics to CSV
ncu --csv --metrics sm__throughput.avg,dram__throughput.avg python train.py
```

**Use cases**:
- ✅ Optimize custom CUDA kernels
- ✅ Understand why kernel is slow (occupancy? memory bandwidth?)
- ✅ Validate kernel optimization (before/after comparison)
- ❌ NOT for end-to-end training profiling (too slow - kernel replay)
- ❌ NOT for multi-kernel timeline analysis

### torch.profiler vs NSight Systems vs NSight Compute

From [NVIDIA Developer Forums discussion](https://forums.developer.nvidia.com/t/nsight-compute-vs-nsight-systems-vs-pytorch-profiler/283271) (accessed 2025-11-13):

| Feature | torch.profiler | NSight Systems | NSight Compute |
|---------|----------------|----------------|----------------|
| **CUPTI required** | ✅ Yes | ❌ No | ❌ No |
| **PyTorch operator names** | ✅ Yes | ❌ No | ❌ No |
| **GPU kernel timeline** | ✅ Yes | ✅ Yes | ❌ No (single kernel) |
| **GPU hardware counters** | ❌ Limited | ❌ No | ✅ Yes (full) |
| **Overhead** | 5-10% | 2-3% | Very high (kernel replay) |
| **Python stack traces** | ✅ Yes | ❌ No | ❌ No |
| **TensorBoard integration** | ✅ Yes | ❌ No | ❌ No |
| **Multi-node support** | ✅ Yes | ✅ Yes | ❌ No |
| **Use case** | PyTorch training | System bottlenecks | Kernel optimization |

**When to use which tool**:
1. **torch.profiler**: First choice for PyTorch-specific optimization (if CUPTI available)
2. **NSight Systems**: System-wide bottlenecks, multi-GPU/multi-node, or when CUPTI missing
3. **NSight Compute**: Deep-dive into specific slow kernel after identifying it

---

## Section 3: Profiling Without CUPTI (~80 lines)

### Why CUPTI Might Be Missing

From CUPTI investigation experience (arr-coc-0-1 project, 2025-11-13):

**Common scenarios where CUPTI is missing**:
1. **Docker runtime images** - nvidia/cuda:*-runtime-* base images exclude CUPTI
2. **Cloud environments** - Some cloud providers don't include CUPTI in GPU VMs
3. **Production deployments** - Security-hardened containers often strip dev tools
4. **Minimal installs** - cuda-libraries-* packages don't include CUPTI (need cuda-libraries-dev-*)

**CUPTI location**:
```bash
# CUPTI lives in development packages
/usr/local/cuda/extras/CUPTI/lib64/libcupti.so  # Typical location
/usr/local/cuda-12.4/extras/CUPTI/              # Version-specific

# Check if CUPTI is available
ldconfig -p | grep cupti
# If empty → CUPTI not installed
```

### Fallback Strategy 1: NSight Systems (Recommended)

**Best alternative when torch.profiler fails due to missing CUPTI**:

```bash
# Install NSight Systems (doesn't need CUPTI)
# Download from: https://developer.nvidia.com/nsight-systems

# Profile PyTorch training
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=training_trace \
    --force-overwrite true \
    python train.py

# View trace
nsys-ui training_trace.nsys-rep
```

**What you get**:
- ✅ GPU kernel timelines (via driver hooks, not CUPTI)
- ✅ CUDA API calls (cudaMalloc, cudaMemcpy, etc.)
- ✅ CPU thread activity
- ❌ No PyTorch operator names (workaround: add NVTX annotations)
- ❌ No automatic TensorBoard export

### Fallback Strategy 2: CPU-Only Profiling

**torch.profiler with CUPTI disabled**:

```python
from torch.profiler import profile, ProfilerActivity

# Profile CPU only (no CUPTI needed)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    model(inputs)
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

**What you get**:
- ✅ PyTorch operator CPU time
- ✅ Python stack traces
- ✅ Memory allocations (CPU side)
- ❌ No GPU kernel timings
- ❌ No GPU memory events

**Use case**: Identify CPU bottlenecks (data loading, preprocessing, Python overhead)

### Fallback Strategy 3: Manual CUDA Event Timing

**Low-overhead manual profiling**:

```python
import torch

# Create CUDA events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Measure forward pass
start.record()
outputs = model(inputs)
end.record()
torch.cuda.synchronize()
forward_time = start.elapsed_time(end)

# Measure backward pass
start.record()
loss.backward()
end.record()
torch.cuda.synchronize()
backward_time = start.elapsed_time(end)

print(f"Forward: {forward_time:.2f}ms, Backward: {backward_time:.2f}ms")
```

**Advantages**:
- ✅ Zero dependencies (works everywhere)
- ✅ Minimal overhead (~0.1%)
- ✅ Production-safe (always-on monitoring)
- ❌ Manual instrumentation required
- ❌ No kernel-level detail

### Fallback Strategy 4: Simple Timing Wrapper

```python
import time
import torch

class SimpleProfiler:
    def __init__(self):
        self.timings = {}

    def measure(self, name):
        return self.Timer(self, name)

    class Timer:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def __exit__(self, *args):
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start) * 1000
            self.profiler.timings[self.name] = elapsed

# Usage
profiler = SimpleProfiler()

with profiler.measure("forward"):
    outputs = model(inputs)

with profiler.measure("backward"):
    loss.backward()

with profiler.measure("optimizer"):
    optimizer.step()

print(profiler.timings)
# {'forward': 45.23, 'backward': 78.91, 'optimizer': 12.34}
```

---

## Section 4: Production Profiling Strategies (~80 lines)

### Meta's Automated Profiling Approach

From [PyTorch Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) (accessed 2025-11-13):

**Dynolog + PyTorch Profiler architecture**:

```
1. PyTorch app registers with Dynolog daemon (KINETO_USE_DAEMON=True)
2. Engineer triggers profiling via Dynolog CLI (dyno gputrace)
3. Dynolog sends profiling config to PyTorch app over network
4. PyTorch profiler collects 3-5 iterations
5. Trace saved to shared storage (NFS/S3)
6. HTA (Holistic Trace Analysis) analyzes traces automatically
```

**Key innovation**: Zero code instrumentation - profiling triggered externally.

**Implementation**:
```bash
# 1. Start PyTorch training with Dynolog integration
export KINETO_USE_DAEMON=True
python train.py

# 2. In another terminal, trigger profiling
dyno gputrace --pid <pytorch_pid> --duration 30

# 3. Trace automatically saved to configured location
```

### Production Profiling Best Practices

**1. Profile in Short Bursts**

```python
# DON'T: Profile entire training run (too much overhead)
with profile(activities=[...]) as prof:
    for epoch in range(100):  # ❌ 100 epochs profiled
        train_epoch()

# DO: Profile a few iterations periodically
for epoch in range(100):
    if epoch % 10 == 0:  # Profile every 10th epoch
        with profile(activities=[...]) as prof:
            for i in range(5):  # Profile 5 iterations only
                train_batch()
            prof.export_chrome_trace(f"trace_epoch_{epoch}.json")
    else:
        train_epoch()  # No profiling overhead
```

**2. Use Profiler Schedule for Minimal Overhead**

```python
from torch.profiler import schedule

# Profile first 3 iterations of every 10
prof = profile(
    schedule=schedule(
        wait=5,      # Skip first 5 iterations (warmup)
        warmup=2,    # Warmup for 2 iterations (stabilize)
        active=3,    # Profile 3 iterations
        repeat=1     # Repeat pattern 1 time
    ),
    on_trace_ready=lambda prof: prof.export_chrome_trace("trace.json")
)

with prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()  # Signal iteration end
```

**3. Docker Image Considerations**

From CUPTI investigation (arr-coc-0-1, 2025-11-13):

**Multi-stage build for optional profiling**:

```dockerfile
# Stage 1: Runtime (no CUPTI, minimal size)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime
RUN pip install torch torchvision
# Image size: ~1.5GB

# Stage 2: Development (with CUPTI, for profiling)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS devel
RUN pip install torch torchvision
# CUPTI included in devel image
# Image size: ~4.5GB

# Production: Use runtime
# Development/Profiling: Use devel
```

**Alternative: Add CUPTI to runtime image**:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install only CUPTI (without full devel)
RUN apt-get update && apt-get install -y \
    cuda-cupti-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Set LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

**4. TensorBoard Integration**

```python
from torch.profiler import profile, tensorboard_trace_handler

# Automatically export to TensorBoard format
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler('./runs/profiler')
) as prof:
    train_step()

# View in TensorBoard
# tensorboard --logdir=./runs
```

### Holistic Trace Analysis (HTA) for Batch Analysis

From [PyTorch HTA blog](https://pytorch.org/blog/trace-analysis-for-masses/) (accessed 2025-11-13):

**Automated bottleneck detection for multiple traces**:

```python
from hta.trace_analysis import TraceAnalysis

# Load traces from distributed training (8 ranks)
analyzer = TraceAnalysis(trace_dir="./traces")

# Temporal breakdown (compute vs communication vs idle)
analyzer.get_temporal_breakdown()

# Idle time analysis (why are GPUs idle?)
analyzer.get_idle_time_breakdown()

# Kernel duration distribution across ranks
analyzer.get_kernel_duration_distribution()

# Communication-computation overlap
analyzer.get_comm_comp_overlap()
```

**HTA features**:
- ✅ Multi-rank trace analysis (distributed training)
- ✅ Automated bottleneck detection
- ✅ GPU utilization metrics
- ✅ Communication efficiency analysis
- ✅ Jupyter notebook integration

### Minimal-Overhead Production Monitoring

**Always-on lightweight monitoring** (no CUPTI needed):

```python
import torch
import time

class ProductionMonitor:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_times = []

    def log_step(self, step):
        if step % self.log_interval == 0:
            torch.cuda.synchronize()
            avg_time = sum(self.step_times[-100:]) / len(self.step_times[-100:])
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"Step {step}: {avg_time:.2f}ms/step, GPU mem: {gpu_mem:.2f}GB")
            self.step_times = []  # Reset

    def measure_step(self):
        return self.StepTimer(self)

    class StepTimer:
        def __init__(self, monitor):
            self.monitor = monitor

        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def __exit__(self, *args):
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start) * 1000
            self.monitor.step_times.append(elapsed)

# Usage
monitor = ProductionMonitor(log_interval=100)

for step, batch in enumerate(dataloader):
    with monitor.measure_step():
        train_step(batch)
    monitor.log_step(step)
```

**Overhead**: <0.5% (just torch.cuda.synchronize() + time.perf_counter())

---

## Sources

**PyTorch Official Documentation:**
- [PyTorch Profiler Documentation](https://docs.pytorch.org/docs/stable/profiler.html) (accessed 2025-11-13)
- [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (accessed 2025-11-13)
- [PyTorch Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) (Sept 5, 2023, accessed 2025-11-13)
- [PyTorch Introducing Profiler blog](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) (March 25, 2021, accessed 2025-11-13)

**PyTorch Kineto:**
- [PyTorch Kineto GitHub Repository](https://github.com/pytorch/kineto) (accessed 2025-11-13)

**NVIDIA Profiling Tools:**
- [NVIDIA NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/) (accessed 2025-11-13)
- [NVIDIA NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) (accessed 2025-11-13)
- [NVIDIA Developer Forums: NSight Compute vs NSight Systems vs PyTorch Profiler](https://forums.developer.nvidia.com/t/nsight-compute-vs-nsight-systems-vs-pytorch-profiler/283271) (Feb 19, 2024, accessed 2025-11-13)

**Meta Production Debugging:**
- [Performance Debugging of Production PyTorch Models at Meta](https://pytorch.org/blog/performance-debugging-of-production-pytorch-models-at-meta/) (Sept 29, 2022, accessed 2025-11-13)

**Holistic Trace Analysis:**
- [Trace Analysis for Masses blog](https://pytorch.org/blog/trace-analysis-for-masses/) (accessed 2025-11-13)
- [HTA ReadTheDocs](https://hta.readthedocs.io/) (accessed 2025-11-13)

**arr-coc-0-1 CUPTI Investigation:**
- Local investigation experience from arr-coc-0-1 training infrastructure debugging (2025-11-13)

**Additional Research:**
- [GPU Profiling Under the Hood survey](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/) (April 21, 2025, accessed 2025-11-13)
- [Argonne National Lab: PyTorch Profiler for AI](https://www.alcf.anl.gov/sites/default/files/2025-05/PyTorchProfiler_HZheng.pdf) (May 7, 2025, accessed 2025-11-13)
