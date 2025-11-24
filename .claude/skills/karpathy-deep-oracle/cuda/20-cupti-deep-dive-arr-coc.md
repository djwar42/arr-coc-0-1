# CUPTI Deep Dive: What Is It, When Do You Need It, arr-coc-0-1 Analysis

**Production-validated knowledge from CUPTI investigation**: Understanding CUPTI (CUDA Profiling Tools Interface), its use cases, overhead characteristics, and production deployment decisions. Includes specific analysis for arr-coc-0-1 VLM training project.

**Key question answered**: Do we need CUPTI for arr-coc-0-1 training and inference?

---

## Section 1: What is CUPTI? (~100 lines)

### CUPTI Architecture Overview

From [NVIDIA CUPTI Developer Page](https://developer.nvidia.com/cupti) (accessed 2025-11-13):

> "The NVIDIA CUDA Profiling Tools Interface (CUPTI) is a library that enables the creation of profiling and tracing tools that target CUDA applications."

**CUPTI is NOT a profiler itself** - it's an **interface** that profilers use to collect GPU data.

**CUPTI API Categories**:

```
CUPTI Library (libcupti.so)
├── Activity API - GPU timeline tracing (kernel start/end, memory transfers)
├── Callback API - CUDA runtime/driver API call interception
├── Profiling API - Hardware performance counters (new, replaces legacy)
├── Range Profiling API - Metrics for concurrent kernel launches
├── PC Sampling API - Program counter sampling (instruction-level profiling)
├── SASS Metric API - Assembly-level performance metrics
├── PM Sampling API - Power management sampling
└── Checkpoint API - Save/restore profiling state
```

**Who uses CUPTI**:
- PyTorch Profiler (via Kineto library)
- NVIDIA NSight Systems (partial - uses driver hooks primarily)
- NVIDIA NSight Compute (standalone, doesn't require CUPTI installation)
- TensorBoard Profiler Plugin
- Custom profiling tools

From [PyTorch Kineto GitHub](https://github.com/pytorch/kineto) (accessed 2025-11-13):

> "Kineto is PyTorch's CPU+GPU profiling library that provides low-overhead GPU timeline tracing. GPU-side support is needed to create the trace. NVIDIA provides this visibility via the CUPTI library. Kineto is the subsystem within Profiler that interfaces with CUPTI."

### CUPTI vs NSight Tools

From existing knowledge (cuda/11-pytorch-profiling-ecosystem.md):

**Key distinction**: NSight Systems does NOT require CUPTI for basic GPU profiling.

| Feature | CUPTI (via torch.profiler) | NSight Systems | NSight Compute |
|---------|---------------------------|----------------|----------------|
| **CUPTI required** | ✅ Yes (libcupti.so) | ❌ No (driver hooks) | ❌ No (standalone) |
| **PyTorch operator names** | ✅ Yes | ❌ No (NVTX workaround) | ❌ No |
| **GPU kernel timeline** | ✅ Yes | ✅ Yes | ❌ No (single kernel) |
| **GPU hardware counters** | ❌ Limited | ❌ No | ✅ Yes (full) |
| **Overhead** | 5-10% | 2-3% | Very high (replay) |
| **TensorBoard integration** | ✅ Yes | ❌ No | ❌ No |
| **Use case** | PyTorch training | System bottlenecks | Kernel optimization |

**Critical insight from production**: If you don't have CUPTI, you can still profile with NSight Systems.

### CUPTI Components Deep Dive

**1. Activity API** (what PyTorch uses):

```c
// What Kineto collects via CUPTI Activity API
- CUPTI_ACTIVITY_KIND_KERNEL          // GPU kernel execution timeline
- CUPTI_ACTIVITY_KIND_MEMCPY          // H2D, D2H, D2D transfers
- CUPTI_ACTIVITY_KIND_MEMSET          // cudaMemset operations
- CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL  // Overlapping kernel execution
- CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER  // UM page faults
- CUPTI_ACTIVITY_KIND_CUDA_EVENT      // cudaEvent timing
```

From [NVIDIA CUPTI Documentation](https://docs.nvidia.com/cupti/) (accessed 2025-11-13):

**Activity API overhead characteristics**:
- Minimal overhead (~1-3%) when collecting kernel timelines only
- Moderate overhead (~5-10%) when collecting full trace (kernels + memory + Python stacks)
- High overhead (~15-25%) when collecting Python stack traces for every operator

**2. Callback API**:

```c
// CUDA runtime/driver API call interception
CUPTI_CBID_RUNTIME_CUDALAUNCH        // cudaLaunch callbacks
CUPTI_CBID_RUNTIME_CUDAMALLOC        // cudaMalloc callbacks
CUPTI_CBID_RUNTIME_CUDAMEMCPY        // cudaMemcpy callbacks
```

**Use case**: Intercept CUDA API calls for custom profiling logic.

**3. Profiling API** (replaces legacy Event/Metric APIs in CUDA 13.0+):

From [CUPTI 13.0 Updates](https://developer.nvidia.com/cupti) (accessed 2025-11-13):

> "The CUPTI Profiling API from the header cupti_profiler_target.h and the Perfworks Metric API from the header nvperf_host.h are deprecated in the CUDA 13.0 release and will be removed in a future CUDA release. It is recommended to use the CUPTI Range Profiling API as an alternative."

**Hardware metrics available**:
- SM occupancy (warps/SM, registers/thread)
- Memory bandwidth (DRAM, L2 cache)
- Instruction throughput (FP32, FP64, Tensor Core)
- Warp stall reasons (memory dependency, execution dependency)

### What CUPTI Provides That Nothing Else Does

**Unique capabilities**:

1. **GPU-side timeline with low overhead** (~2-5%)
   - NSight Systems has similar capability (driver hooks, no CUPTI needed)
   - CUPTI advantage: PyTorch operator attribution

2. **Python stack trace correlation**
   - Map GPU kernels → PyTorch operators → Python source code
   - NSight Systems cannot do this (needs NVTX annotations)

3. **Programmatic profiling control**
   - Start/stop profiling from within PyTorch code
   - NSight Systems requires external launch

4. **TensorBoard integration**
   - PyTorch Profiler → TensorBoard plugin
   - NSight Systems has no TensorBoard export

From [PyTorch Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) (Sept 5, 2023, accessed 2025-11-13):

> "GPU kernels execute asynchronously, and GPU-side support is needed to create the trace. NVIDIA provides this visibility via the CUPTI library. Kineto is the subsystem within Profiler that interfaces with CUPTI."

### CUPTI Installation Location

From existing knowledge (cuda/13-nvidia-container-cuda-packaging.md):

**CUPTI lives in `cuda-libraries-dev-{version}` apt package, NOT in runtime!**

```bash
# Runtime image (no CUPTI)
nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Missing: /usr/lib/x86_64-linux-gnu/libcupti.so.12

# Devel image (has CUPTI)
nvidia/cuda:12.4.0-devel-ubuntu22.04
# Includes: /usr/lib/x86_64-linux-gnu/libcupti.so.12
```

**Critical discovery from arr-coc-0-1 investigation**:
- CUPTI headers in `/usr/local/cuda/extras/CUPTI/include/` (devel image)
- CUPTI shared libraries in `/usr/lib/x86_64-linux-gnu/libcupti.so*` (from apt package)
- Copying `/usr/local/cuda/extras/CUPTI/` to runtime image FAILS (headers only, no .so files)

**Solution**: Install `cuda-libraries-dev-12-4` package or extract `libcupti.so` files.

---

## Section 2: CUPTI Use Cases (~100 lines)

### Profiling Workflows That Require CUPTI

**1. PyTorch Profiler (torch.profiler)**

From [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html) (accessed 2025-11-13):

```python
import torch
from torch.profiler import profile, ProfilerActivity

# REQUIRES CUPTI for GPU profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    model(inputs)
    loss.backward()

# Export to TensorBoard
prof.export_chrome_trace("trace.json")
```

**What works without CUPTI**:
- ✅ CPU profiling (`ProfilerActivity.CPU` only)
- ✅ Python stack traces
- ❌ GPU kernel timelines
- ❌ GPU memory events
- ❌ CUDA API tracing

**2. Dynolog On-Demand Profiling**

From [PyTorch Automated Trace Collection](https://pytorch.org/blog/automated-trace-collection/) (accessed 2025-11-13):

**Production workflow at Meta**:

```
1. PyTorch app registers with Dynolog daemon (KINETO_USE_DAEMON=True)
2. Engineer triggers profiling via Dynolog CLI: dyno gputrace
3. Dynolog sends profiling config to PyTorch app over network
4. PyTorch Profiler (via CUPTI) collects 3-5 iterations
5. Trace saved to shared storage (NFS/S3)
6. HTA (Holistic Trace Analysis) analyzes traces automatically
```

**CUPTI dependency**: Entire workflow requires CUPTI for GPU trace collection.

**Key innovation**: Zero code instrumentation - profiling triggered externally.

**3. Custom Profiling Tools**

From [CUPTI User Guide](https://docs.nvidia.com/cupti/) (accessed 2025-11-13):

**Example use cases**:
- Training framework performance monitoring (MLPerf)
- Distributed training debuggers (NCCL trace analysis)
- Memory leak detectors (CUDA malloc/free tracking)
- Automatic mixed precision profilers (FP16 vs FP32 kernel detection)

**What they use CUPTI for**:
- Callback API: Intercept `cudaMalloc`/`cudaFree` for memory tracking
- Activity API: Collect NCCL kernel timelines for communication analysis
- Profiling API: Measure SM occupancy to validate AMP effectiveness

### Profiling Workflows WITHOUT CUPTI

**1. NSight Systems (Recommended CUPTI Alternative)**

From existing knowledge (cuda/11-pytorch-profiling-ecosystem.md):

```bash
# Profile PyTorch training WITHOUT CUPTI
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
- ✅ Multi-process profiling (DDP training)
- ❌ No PyTorch operator names (workaround: add NVTX annotations)
- ❌ No automatic TensorBoard export

**Overhead**: ~2-3% (lower than torch.profiler's 5-10%)

**2. CPU-Only PyTorch Profiling**

```python
from torch.profiler import profile, ProfilerActivity

# NO CUPTI NEEDED - CPU profiling only
with profile(activities=[ProfilerActivity.CPU]) as prof:
    model(inputs)
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

**Use case**: Identify CPU bottlenecks (data loading, preprocessing, Python overhead).

**3. Manual CUDA Event Timing**

```python
import torch

# NO CUPTI NEEDED - Zero dependencies
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
outputs = model(inputs)
end.record()
torch.cuda.synchronize()

forward_time = start.elapsed_time(end)  # milliseconds
print(f"Forward pass: {forward_time:.2f}ms")
```

**Overhead**: ~0.1% (minimal)

**Use case**: Production monitoring (always-on, low-overhead).

### CUPTI Overhead Measurements

From [CUPTI 13.0 Documentation](https://docs.nvidia.com/cupti/main/main.html) (accessed 2025-11-13):

**Activity API overhead** (collecting GPU timeline):
- **Minimal config** (kernel timeline only): 1-3% overhead
- **Standard config** (kernels + memory + CUDA API): 5-10% overhead
- **Full config** (+ Python stacks + shapes): 15-25% overhead

**Profiling API overhead** (hardware counters):
- **Single metric collection**: 3-5% overhead
- **Multiple metrics** (10+ counters): 10-20% overhead
- **Full metric set** (100+ counters): 30-50% overhead

From [PyTorch Kineto profiling overhead research](https://github.com/pytorch/kineto) (accessed 2025-11-13):

**Kineto (PyTorch Profiler) overhead benchmarks**:
- CPU profiling alone: ~2-5% overhead
- GPU profiling (with CUPTI): ~5-10% overhead
- Memory profiling: Additional ~3-5% overhead
- Python stack traces: Additional ~10-15% overhead

**Production recommendation**: Profile for short bursts (3-5 iterations) rather than continuous profiling.

### When You NEED CUPTI vs When You DON'T

**✅ NEED CUPTI:**

1. **PyTorch Profiler with GPU support** (`ProfilerActivity.CUDA`)
2. **TensorBoard Profiler Plugin** (requires PyTorch Profiler traces)
3. **Dynolog automated profiling** (Meta's production workflow)
4. **Distributed training analysis** (Holistic Trace Analysis - HTA)
5. **Python stack trace → GPU kernel mapping**
6. **Custom profiling tools** (using CUPTI APIs directly)

**❌ DON'T NEED CUPTI:**

1. **NSight Systems profiling** (uses driver hooks, not CUPTI)
2. **NSight Compute profiling** (standalone tool)
3. **Production inference** (no profiling needed)
4. **CPU-only profiling** (`ProfilerActivity.CPU`)
5. **Manual timing** (torch.cuda.Event)
6. **Basic monitoring** (GPU utilization via nvidia-smi)

---

## Section 3: Production Decision Tree (~100 lines)

### When You NEED CUPTI (Development & Deep Profiling)

**Scenario 1: Development/Training Phase**

**Use case**: Optimize PyTorch model training performance.

**Why you need CUPTI**:
- Identify slow operators (PyTorch Profiler)
- Analyze GPU utilization per layer
- Debug data loading bottlenecks (CPU vs GPU time)
- Profile distributed training (DDP, FSDP)

**Workflow**:
```bash
# Development Docker image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# CUPTI included in devel image
# Full PyTorch Profiler support available

python train.py  # Profile with torch.profiler
```

**Overhead acceptable**: 5-10% slowdown during profiling is fine in development.

**Scenario 2: Production Training at Scale**

**Use case**: Meta-scale distributed training with automated profiling.

From [PyTorch Automated Trace Collection](https://pytorch.org/blog/automated-trace-collection/) (accessed 2025-11-13):

**Why you need CUPTI**:
- On-demand profiling via Dynolog (no code changes)
- Analyze thousands of training jobs automatically
- Detect GPU underutilization across fleet
- Debug performance regressions in production

**Workflow**:
```bash
# Training service launches job with KINETO_USE_DAEMON=True
# Engineer triggers profiling remotely:
dyno gputrace --pid <pytorch_pid> --duration 30

# CUPTI required for GPU trace collection
# HTA analyzes traces to identify bottlenecks
```

**Trade-off**: 5-10% overhead during 30-second profiling burst vs continuous monitoring.

**Scenario 3: Custom Performance Tools**

**Use case**: Build framework-specific profiling (e.g., MLPerf compliance).

**Why you need CUPTI**:
- Collect hardware performance counters (SM occupancy, memory bandwidth)
- Track CUDA API call patterns (malloc/free for memory analysis)
- Correlate CPU and GPU timelines programmatically

**Implementation**: Direct CUPTI API usage (Callback API + Activity API).

### When You DON'T NEED CUPTI (Production Deployment)

**Scenario 1: Production Inference**

**Use case**: Deploy PyTorch model for serving.

**Why you DON'T need CUPTI**:
- No profiling in production inference
- Monitoring via prometheus + nvidia-smi (no CUPTI)
- Manual timing if needed (torch.cuda.Event - no CUPTI)

**Docker image strategy**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# NO CUPTI - smaller image, faster startup
# Runtime libraries only (libcudart, libcublas, etc.)
# Image size: ~1.5GB vs 4.5GB (devel)
```

**Benefit**: 3GB smaller Docker image, faster deployment.

**Scenario 2: System-Level Profiling**

**Use case**: Debug multi-GPU bottlenecks, I/O issues, CPU-GPU interaction.

**Why you DON'T need CUPTI**:
- Use NSight Systems instead (no CUPTI required)
- Lower overhead (2-3% vs 5-10%)
- System-wide view (CPU + GPU + network + disk)

**Workflow**:
```bash
# NO CUPTI needed
nsys profile --trace=cuda,nvtx,osrt python train.py

# Add NVTX annotations for visibility
import torch.cuda.nvtx as nvtx
nvtx.range_push("Forward")
output = model(data)
nvtx.range_pop()
```

**Trade-off**: Manual NVTX annotations vs automatic PyTorch operator names.

**Scenario 3: Basic Performance Monitoring**

**Use case**: Track training progress, detect slowdowns.

**Why you DON'T need CUPTI**:
- Simple timing with torch.cuda.Event (0.1% overhead)
- GPU metrics via nvidia-smi (no overhead)
- Loss/accuracy tracking (native PyTorch)

**Production monitoring**:
```python
import torch
import time

class ProductionMonitor:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_times = []

    def measure_step(self):
        return self.StepTimer(self)

    class StepTimer:
        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def __exit__(self, *args):
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start) * 1000
            self.monitor.step_times.append(elapsed)

# NO CUPTI - works everywhere
monitor = ProductionMonitor()
for step, batch in enumerate(dataloader):
    with monitor.measure_step():
        train_step(batch)
```

**Overhead**: <0.5% (minimal impact on training throughput).

### Docker Image Size Trade-offs

From arr-coc-0-1 CUPTI investigation:

**Option 1: Runtime image (no CUPTI)**
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# Size: ~1.5GB
# CUPTI: ❌ Not available
# Use case: Production inference, no profiling
```

**Option 2: Devel image (full CUPTI)**
```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# Size: ~4.5GB
# CUPTI: ✅ Full support
# Use case: Development, profiling, debugging
```

**Option 3: Runtime + minimal CUPTI**
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install CUPTI only (~20MB overhead)
RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-libraries-dev-12-4 && \
    cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
    apt-get remove -y cuda-libraries-dev-12-4 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Size: ~1.52GB (+20MB for CUPTI)
# CUPTI: ✅ PyTorch Profiler works
# Use case: Production training with optional profiling
```

**Decision matrix**:
- Development: Use devel image (easy, full tooling)
- Production inference: Use runtime image (minimal size)
- Production training: Use runtime + minimal CUPTI (best of both worlds)

---

## Section 4: arr-coc Analysis (~100 lines)

### arr-coc-0-1 Project Context

**What is arr-coc-0-1?**

From project README and investigation context:

- **VLM training project** (vision-language model)
- **Adaptive relevance realization** architecture (Vervaekean cognitive framework)
- **Training infrastructure**: Vertex AI + W&B Launch (Google Cloud)
- **Image building**: Cloud Build compiles PyTorch from source (2-4 hour builds)
- **Production deployment**: HuggingFace Spaces (Gradio demo)

**Training characteristics**:
- Multi-channel vision processing (13-channel texture arrays)
- Variable LOD (64-400 tokens per patch)
- Query-aware compression (dynamic relevance realization)
- Qwen3-VL backbone integration

### Do We Need CUPTI for arr-coc-0-1 Training?

**Answer: NO for initial development, YES for optimization phase.**

**Reasoning**:

**Phase 1: Initial training (current)**
- ❌ **CUPTI NOT needed**
- Focus: Validate architecture, verify training convergence
- Profiling: Basic timing (torch.cuda.Event), W&B metrics
- Docker: Runtime image (smaller, faster deployment)
- Benefit: 3GB smaller images, faster Cloud Build

**Phase 2: Optimization (future)**
- ✅ **CUPTI NEEDED**
- Focus: Optimize training throughput, reduce costs
- Profiling: PyTorch Profiler → TensorBoard, identify bottlenecks
- Docker: Runtime + minimal CUPTI (20MB overhead)
- Workflow:
  ```python
  # Enable profiling when needed
  with torch.profiler.profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      record_shapes=True
  ) as prof:
      # Profile 5 iterations
      for i in range(5):
          train_step(batch)
          prof.step()

  prof.export_chrome_trace("arr_coc_trace.json")
  ```

**When to add CUPTI to arr-coc-0-1**:

✅ **Add CUPTI when**:
- Training is slow (>expected time per epoch)
- GPU utilization is low (<80%)
- Need to optimize specific layers (texture processing, LOD allocation)
- Debugging distributed training issues (multi-GPU)
- Preparing for production scale-up

❌ **Skip CUPTI when**:
- Initial prototyping (architecture validation)
- Single-GPU development (easy to debug without profiling)
- Inference-only deployment (HuggingFace Space demo)

### Do We Need CUPTI for arr-coc-0-1 Inference?

**Answer: NO - inference doesn't need profiling.**

**Reasoning**:

**Production inference (HuggingFace Spaces)**:
- Gradio demo application
- User uploads image + text query
- Model generates response
- No profiling needed in production

**Development inference testing**:
- Manual timing sufficient:
  ```python
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  start.record()
  output = model(image, query)
  end.record()
  torch.cuda.synchronize()

  inference_time = start.elapsed_time(end)
  print(f"Inference: {inference_time:.2f}ms")
  ```
- No CUPTI dependency
- Works on any PyTorch installation

**Docker image for inference**:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# NO CUPTI - production inference doesn't profile
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /torch

# Minimal image size for fast deployment
# Image size: ~1.5GB (no CUPTI overhead)
```

### Recommendation for arr-coc Deployment

**Training Docker Image Strategy**:

```dockerfile
# Multi-stage build
# Stage 1: Build PyTorch from source (Cloud Build)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
WORKDIR /build

# Compile PyTorch (takes 2-4 hours on Cloud Build)
# CUPTI headers available during build
RUN python -m pip install torch --no-binary torch

# Stage 2: Runtime with optional CUPTI
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Copy compiled PyTorch
COPY --from=builder /usr/local/lib/python3.10/site-packages/torch /usr/local/lib/python3.10/site-packages/torch

# OPTION A: No CUPTI (Phase 1 - initial training)
# Image size: ~1.5GB
# Profiling: torch.cuda.Event only

# OPTION B: Minimal CUPTI (Phase 2 - optimization)
# Uncomment below to add CUPTI support:
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends cuda-libraries-dev-12-4 && \
#     cp /usr/lib/x86_64-linux-gnu/libcupti.so.12* /usr/local/lib/ && \
#     apt-get remove -y cuda-libraries-dev-12-4 && \
#     apt-get autoremove -y && \
#     rm -rf /var/lib/apt/lists/*
# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Image size: ~1.52GB (+20MB)
# Profiling: Full PyTorch Profiler support

WORKDIR /app
COPY arr_coc/ /app/arr_coc/
COPY training/ /app/training/

CMD ["python", "training/train.py"]
```

**Recommended approach**:

1. **Start without CUPTI** (Phase 1)
   - Use runtime-only image
   - Monitor with W&B metrics + basic timing
   - 3GB smaller images → faster Cloud Build → lower costs

2. **Add CUPTI when needed** (Phase 2)
   - Uncomment minimal CUPTI installation
   - Profile specific bottlenecks
   - Only 20MB overhead (vs 3GB devel image)

3. **Remove CUPTI for production** (Phase 3)
   - After optimization complete
   - Deploy inference-only image
   - Minimal size for HuggingFace Spaces

### CUPTI and Cloud Build Timeouts

From arr-coc-0-1 context:

**Cloud Build constraints**:
- PyTorch compilation: 2-4 hours (from source)
- Default timeout: 1 hour (too short)
- Extended timeout: 4 hours (cloudbuild.yaml setting)

**CUPTI impact on build time**:
- ❌ **No impact** - CUPTI doesn't affect PyTorch compilation
- CUPTI detection happens at PyTorch build time (CMake finds headers)
- Runtime CUPTI installation is fast (~1 minute via apt)

**Build strategy**:
```yaml
# cloudbuild.yaml
timeout: 14400s  # 4 hours (for PyTorch compilation)

steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--target=builder', '-t', 'pytorch-builder', '.']
    timeout: 14400s  # Full 4 hours for PyTorch build

  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--target=runtime', '-t', 'arr-coc-training', '.']
    timeout: 600s    # 10 minutes for runtime stage (fast)
```

**CUPTI doesn't cause timeouts** - PyTorch source compilation does.

### arr-coc-0-1 Profiling Workflow (When Needed)

**Step 1: Enable CUPTI in Docker image**

Uncomment minimal CUPTI installation in Dockerfile (20MB overhead).

**Step 2: Add profiling to training script**

```python
# training/train.py
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Profile every 100 epochs for 5 iterations
def train_epoch(epoch, dataloader, model, optimizer):
    # Enable profiling every 100 epochs
    if epoch % 100 == 0:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(
                f"gs://arr-coc-traces/trace_epoch_{epoch}.json"
            ),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Profile 5 iterations only
                    break
                train_step(batch, model, optimizer)
                prof.step()
    else:
        # Normal training (no profiling overhead)
        for batch in dataloader:
            train_step(batch, model, optimizer)
```

**Step 3: Analyze traces**

```bash
# Download trace from GCS
gsutil cp gs://arr-coc-traces/trace_epoch_100.json .

# View in TensorBoard
tensorboard --logdir=. --port=6006

# Or use Chrome trace viewer
# Open chrome://tracing and load trace_epoch_100.json
```

**Step 4: Optimize based on findings**

- Identify slow operators (e.g., texture processing bottleneck)
- Check GPU utilization (idle time vs compute time)
- Analyze memory transfers (H2D, D2H overhead)
- Validate LOD allocation performance

**Overhead**: 5-10% slowdown during profiled epochs (every 100 epochs), negligible overall impact.

---

## Sources

**Source Documents:**
- [cuda/11-pytorch-profiling-ecosystem.md](../cuda/11-pytorch-profiling-ecosystem.md) - PyTorch Profiler, Kineto, CUPTI dependencies
- [cuda/13-nvidia-container-cuda-packaging.md](../cuda/13-nvidia-container-cuda-packaging.md) - CUPTI location in Docker images

**Web Research:**
- [NVIDIA CUPTI Developer Page](https://developer.nvidia.com/cupti) (accessed 2025-11-13) - Official CUPTI overview
- [CUPTI 13.0 Documentation](https://docs.nvidia.com/cupti/main/main.html) (accessed 2025-11-13) - API details, overhead analysis
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html) (accessed 2025-11-13) - CUPTI usage in PyTorch
- [PyTorch Kineto GitHub](https://github.com/pytorch/kineto) (accessed 2025-11-13) - CUPTI integration architecture
- [PyTorch Automated Trace Collection blog](https://pytorch.org/blog/automated-trace-collection/) (Sept 5, 2023, accessed 2025-11-13) - Meta's production profiling workflow

**arr-coc-0-1 Project Context:**
- arr-coc-0-1 CUPTI investigation (2025-11-13) - Production Docker build experience
- arr-coc-0-1 training infrastructure - Vertex AI + Cloud Build workflow
