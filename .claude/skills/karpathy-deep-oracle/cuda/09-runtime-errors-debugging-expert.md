# CUDA Runtime Errors & Memory Failures: Expert-Level Debugging

## Overview

Runtime CUDA errors are among the most challenging debugging scenarios in GPU programming. Unlike compilation errors that fail fast, runtime errors can manifest as cryptic error codes, silent failures, or catastrophic memory corruption. This document covers expert-level diagnosis and solutions for out-of-memory (OOM) errors, illegal memory access, kernel launch failures, and advanced debugging techniques.

**Why Runtime Errors are Hard:**
- **Asynchronous execution**: Errors may not appear until `cudaDeviceSynchronize()` or `cudaStreamSynchronize()`
- **Cryptic error codes**: CUDA error 700, 77, 2, 30 require deep understanding
- **Memory corruption**: Invalid pointers cause segfaults with no stack trace
- **Device-side failures**: Printf debugging doesn't work; need cuda-gdb, compute-sanitizer
- **Multi-GPU complexity**: Peer access failures, NCCL timeouts, gradient sync issues

From [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) (accessed 2025-11-13):
> "Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a pin_memory() method, that returns a copy of the object, with data put in a pinned region."

**Related Knowledge:**
- See [cuda/01-memory-management-unified.md](01-memory-management-unified.md) for memory allocation strategies
- See [cuda/08-compilation-troubleshooting-expert.md](08-compilation-troubleshooting-expert.md) for build-time errors
- See [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) for PyTorch-specific compilation

---

## Section 1: Out of Memory (OOM) Errors (~100 lines)

### Understanding PyTorch OOM

**Classic Error Message:**
```python
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 10.76 GiB total capacity; 4.29 GiB already allocated; 10.12 MiB free; 4.46 GiB reserved in total by PyTorch)
```

From [Stack Overflow: How to avoid "CUDA out of memory" in PyTorch](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-11-13):
> "Although `torch.cuda.empty_cache()` provides a good alternative for clearing the occupied cuda memory and we can also manually clear the not in use variables by using `import gc; del variables; gc.collect()`. But still after using these commands, the error might appear again because pytorch doesn't actually clear the memory instead clears the reference to the memory occupied by the variables."

**Key Insight**: PyTorch uses a caching allocator that reserves more memory than actively allocated. The error message shows:
- **Total capacity**: GPU physical memory (10.76 GiB)
- **Already allocated**: Active PyTorch tensors (4.29 GiB)
- **Free**: Available for allocation (10.12 MiB) ← actual bottleneck
- **Reserved**: Cached by PyTorch allocator (4.46 GiB)

### OOM Debugging Workflow

**Step 1: Check actual memory usage**
```python
import torch

# Current memory usage
allocated = torch.cuda.memory_allocated()  # Bytes in tensors
reserved = torch.cuda.memory_reserved()     # Bytes reserved by caching allocator
free = torch.cuda.mem_get_info()[0]        # Free GPU memory (CUDA API)

print(f"Allocated: {allocated / 1e9:.2f} GB")
print(f"Reserved: {reserved / 1e9:.2f} GB")
print(f"Free: {free / 1e9:.2f} GB")

# Memory summary (detailed)
print(torch.cuda.memory_summary(device=None, abbreviated=False))
```

**Step 2: Identify memory leaks**
```python
# Track peak memory
torch.cuda.reset_peak_memory_stats()
model(input)
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")

# Find what's using memory
import gc
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(type(obj), obj.size(), obj.device)
```

**Step 3: Common causes and solutions**

From [Stack Overflow: How to avoid CUDA OOM](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-11-13):

**1. Batch size too large**
```python
# BAD: batch_size=128 on 8GB GPU
dataloader = DataLoader(dataset, batch_size=128)

# GOOD: Reduce batch size
dataloader = DataLoader(dataset, batch_size=16)

# BETTER: Gradient accumulation (simulate large batch)
accumulation_steps = 8
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2. Gradient accumulation without clearing**
```python
# BAD: Accumulating gradients + computation graph
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Keeps computation graph!

# GOOD: Detach from graph
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # Just the value
```

**3. Not clearing optimizer gradients**
```python
# BAD: Gradients accumulate indefinitely
for epoch in range(100):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        # Missing: optimizer.zero_grad()

# GOOD: Clear gradients before backward
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()  # Clear before backward
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Memory Fragmentation

**The Problem**: PyTorch caching allocator can fragment memory over time, causing OOM even when total free memory > requested allocation.

From [PyTorch Forums: Systematically debugging out-of-memory issue](https://discuss.pytorch.org/t/systematically-debugging-out-of-memory-issue/175034) (accessed 2025-11-13):
> "I'm having a recurring out-of-memory issue that seems to be caused by memory fragmentation: torch.cuda.OutOfMemoryError: CUDA out of memory."

**Solutions:**

**1. Environment variable for memory allocation**
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

**2. Expandable segments (PyTorch 2.0+)**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

**3. Manual cache clearing**
```python
# After large operations
torch.cuda.empty_cache()

# Force garbage collection
import gc
del large_tensor
gc.collect()
torch.cuda.empty_cache()
```

**4. Restart kernel as last resort**

From [Stack Overflow comments](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-11-13):
> "I had the same problem using Kaggle. It worked fine with batches of 64 and then once I tried 128 and got the error nothing worked. Even the batches of 64 gave me the same error. Tried resetting a few times. Instead first disable the GPU, then restart the kernel, and reactivate the GPU. This worked for me."

---

## Section 2: Illegal Memory Access (CUDA Error 700, 77) (~100 lines)

### Understanding Illegal Memory Access

**Error codes:**
- **CUDA error 700**: `an illegal memory access was encountered`
- **CUDA error 77**: `an illegal memory access was encountered` (older CUDA versions)
- **CUDA error 2**: `out of memory` (can be triggered by invalid pointer)

From [Stack Overflow: PyTorch CUDA error: an illegal memory access](https://stackoverflow.com/questions/68106457/pytorch-cuda-error-an-illegal-memory-access-was-encountered) (accessed 2025-11-13):
> "It was partially said by the answer of the OP, but the problem under the hood with illegal memory access is that the GPU runs out of memory."

**Key Insight**: Illegal memory access is GPU's equivalent of CPU segfault. Common causes:
- **Buffer overrun**: Indexing beyond array bounds
- **Invalid pointer**: Accessing freed memory, null pointer
- **Misaligned access**: Accessing unaligned addresses
- **Race condition**: Multiple threads writing to same location
- **OOM masquerading**: Out of memory triggers illegal access

### Debugging Illegal Memory Access

**Step 1: Enable synchronous execution**
```bash
# Force synchronous kernel launches
export CUDA_LAUNCH_BLOCKING=1
```

This makes errors appear at the exact line that causes them (vs. appearing later at synchronization point).

**Step 2: Use compute-sanitizer (replaces cuda-memcheck)**
```bash
# Memory error detection
compute-sanitizer --tool memcheck ./my_program

# Race condition detection
compute-sanitizer --tool racecheck ./my_program

# Initialization checking
compute-sanitizer --tool initcheck ./my_program

# Example output
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x150 in kernel(int*, int)
=========     by thread (31,0,0) in block (0,0,0)
=========     Address 0x7f8a40000000 is out of bounds
```

**Step 3: Common patterns and fixes**

**1. Buffer overrun**
```cpp
// BAD: Indexing error
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2.0f;  // No bounds check!
}

// GOOD: Bounds checking
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // Guard against overrun
        data[idx] = idx * 2.0f;
    }
}
```

**2. Shared memory race condition**
```cpp
// BAD: Race condition
__global__ void kernel(float *output) {
    __shared__ float shared[256];
    int tid = threadIdx.x;

    shared[tid] = tid;
    // Missing __syncthreads()!
    output[tid] = shared[tid + 1];  // Race!
}

// GOOD: Proper synchronization
__global__ void kernel(float *output) {
    __shared__ float shared[256];
    int tid = threadIdx.x;

    shared[tid] = tid;
    __syncthreads();  // Wait for all threads
    if (tid < 255) {
        output[tid] = shared[tid + 1];
    }
}
```

**3. PyTorch tensor device mismatch**
```python
# BAD: Tensor on CPU, accessing on GPU
tensor_cpu = torch.randn(100)
model = MyModel().cuda()
output = model(tensor_cpu)  # CUDA error!

# GOOD: Move tensor to GPU first
tensor_gpu = torch.randn(100).cuda()
model = MyModel().cuda()
output = model(tensor_gpu)
```

### Advanced: Using cuda-gdb

```bash
# Compile with debug symbols
nvcc -g -G my_kernel.cu -o my_program

# Run under cuda-gdb
cuda-gdb ./my_program

# Set breakpoint
(cuda-gdb) break my_kernel
(cuda-gdb) run

# When error occurs
(cuda-gdb) cuda thread
(cuda-gdb) cuda block
(cuda-gdb) print *data
```

---

## Section 3: Kernel Launch Failures (~100 lines)

### Understanding Launch Failures

**Common launch error codes:**
- **CUDA error 9**: `invalid configuration argument`
- **CUDA error 719**: `unspecified launch failure`
- **CUDA error 4**: `launch out of resources`

From [NVIDIA Developer Forums: How to debug kernel launch errors](https://forums.developer.nvidia.com/t/cudalaunchkernel-failed-to-launch-kernel/211624) (accessed 2025-11-13):
> "This error is related to the GPU memory and not the general memory => Try using a smaller batch size. If you use Pytorch: do you keep all the training data on the GPU all the time? make sure you don't drag the grads too far, check the sizes of your hidden layer."

### Diagnosing Launch Failures

**Step 1: Check kernel configuration**
```cpp
// Query device limits
int maxThreadsPerBlock;
cudaDeviceGetAttribute(&maxThreadsPerBlock,
                       cudaDevAttrMaxThreadsPerBlock, 0);

int maxBlockDimX;
cudaDeviceGetAttribute(&maxBlockDimX,
                       cudaDevAttrMaxBlockDimX, 0);

printf("Max threads per block: %d\n", maxThreadsPerBlock);
printf("Max block dim X: %d\n", maxBlockDimX);
```

**Step 2: Common launch configuration errors**

**1. Too many threads per block**
```cpp
// BAD: Exceeds limit (usually 1024)
dim3 threads(64, 32, 1);  // 64 * 32 = 2048 threads > 1024!
kernel<<<blocks, threads>>>();

// GOOD: Within limits
dim3 threads(32, 32, 1);  // 32 * 32 = 1024 threads
kernel<<<blocks, threads>>>();
```

**2. Too many blocks**
```cpp
// BAD: Grid dimension overflow
int N = 10000000;
int blocks = N;  // May exceed maxGridSize!
kernel<<<blocks, 256>>>();

// GOOD: Check grid limits
int maxGridSizeX;
cudaDeviceGetAttribute(&maxGridSizeX,
                       cudaDevAttrMaxGridDimX, 0);
int blocks = min(N, maxGridSizeX);
kernel<<<blocks, 256>>>();
```

**3. Shared memory exceeds limit**
```cpp
// Query shared memory limit
int sharedMemPerBlock;
cudaDeviceGetAttribute(&sharedMemPerBlock,
                       cudaDevAttrMaxSharedMemoryPerBlock, 0);

// BAD: Requesting too much
__global__ void kernel() {
    __shared__ float data[100000];  // May exceed 48KB limit!
}

// GOOD: Check before launch
size_t sharedMemNeeded = 100000 * sizeof(float);
if (sharedMemNeeded > sharedMemPerBlock) {
    printf("ERROR: Shared memory exceeds limit\n");
    return;
}
```

**4. Register pressure**
```cpp
// Check register usage
nvcc --ptxas-options=-v my_kernel.cu

// Output shows:
// ptxas info: Used 63 registers, 0 bytes smem, 360 bytes cmem[0]

// If registers > 255, kernel won't launch
// Solution: Reduce local variables, use shared memory
```

### PyTorch-Specific Launch Issues

**1. DataLoader num_workers**
```python
# BAD: Too many workers cause GPU memory fragmentation
dataloader = DataLoader(dataset, batch_size=32, num_workers=16)

# GOOD: Reduce workers, use pin_memory
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Fewer workers
    pin_memory=True     # Faster CPU->GPU transfer
)
```

**2. Multi-stream conflicts**
```python
# BAD: Race condition between streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    output1 = model(input1)

with torch.cuda.stream(stream2):
    output2 = model(input1)  # Race on input1!

# GOOD: Use events for synchronization
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(stream1):
    output1 = model(input1.clone())
    event.record(stream1)

with torch.cuda.stream(stream2):
    event.wait(stream2)  # Wait for stream1
    output2 = model(input1)
```

---

## Section 4: Advanced Debugging Techniques (~100 lines)

### Nsight Compute for Kernel Analysis

From [NVIDIA Developer Blog: Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels) (accessed 2025-11-13):

**Profiling memory access patterns:**
```bash
# Collect memory metrics
nv-nsight-cu-cli \
  --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  ./my_program

# Example output
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     request    65,536
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      sector     2,097,152

# Ratio: 2,097,152 / 65,536 = 32 transactions per request
# Optimal: 4:1 ratio (coalesced access)
# This shows: 32:1 ratio (uncoalesced, poor efficiency)
```

**Memory coalescing check:**
```bash
# Verify coalesced access
nv-nsight-cu-cli --query-metrics | grep coalesced

# Or use GUI
nv-nsight-cu --metrics all ./my_program
```

### PyTorch Memory Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True  # Capture stack traces
) as prof:
    model(input)

# Print memory usage by operation
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=10
))

# Export to Chrome trace for visualization
prof.export_chrome_trace("trace.json")
# Open chrome://tracing and load trace.json
```

**Output example:**
```
---------------------------------  ------------  ------------
Name                               CPU Memory    CUDA Memory
---------------------------------  ------------  ------------
aten::addmm                        0 b           1.00 Gb
aten::empty                        0 b           512.00 Mb
aten::copy_                        0 b           256.00 Mb
---------------------------------  ------------  ------------
```

### Debugging Device-Side Assertions

**Enable device-side assertions:**
```cpp
// In kernel code
__global__ void kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Device-side assertion
    assert(idx < N);  // Will print to stdout on failure

    data[idx] = idx * 2.0f;
}

// Compile with -G flag for device debug
// nvcc -G my_kernel.cu -o my_program
```

**Using printf in kernels:**
```cpp
__global__ void debug_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug print (first thread only to avoid spam)
    if (idx == 0) {
        printf("Kernel launched: N=%d, blockDim=%d\n",
               N, blockDim.x);
    }

    if (idx < N) {
        data[idx] = idx * 2.0f;
    }
}

// Output appears in stdout after cudaDeviceSynchronize()
```

### Automated Error Checking Macros

```cpp
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

kernel<<<blocks, threads>>>(d_data);
CUDA_CHECK(cudaGetLastError());  // Check launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Check execution errors
```

**PyTorch error checking:**
```python
import torch

def check_cuda_error():
    """Check for pending CUDA errors"""
    err = torch.cuda.get_last_error()
    if err != torch.cuda.cudaError.success:
        raise RuntimeError(f"CUDA error: {err}")

# Use after suspicious operations
model(input)
torch.cuda.synchronize()
check_cuda_error()
```

### Production Debugging Checklist

**Before deployment:**
- ✅ Run compute-sanitizer on all kernels
- ✅ Test with CUDA_LAUNCH_BLOCKING=1
- ✅ Profile with Nsight Compute
- ✅ Verify memory coalescing (4:1 ratio)
- ✅ Check kernel configurations against device limits
- ✅ Test OOM recovery (reduce batch size)
- ✅ Monitor GPU memory with nvidia-smi

**Monitoring commands:**
```bash
# Real-time GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.free,utilization.memory \
           --format=csv -l 1

# Watch for OOM patterns
watch -n 1 'nvidia-smi | grep -A 5 "Processes:"'

# Check for zombie processes holding GPU memory
fuser -v /dev/nvidia*
```

**Emergency recovery:**
```bash
# Reset GPU if stuck
sudo nvidia-smi --gpu-reset

# Or reset specific GPU
sudo nvidia-smi --gpu-reset -i 0
```

---

## Diagnostic Command Reference

**Quick diagnostic workflow:**
```bash
# 1. Check GPU status
nvidia-smi

# 2. Check for runtime errors with blocking
CUDA_LAUNCH_BLOCKING=1 python my_script.py

# 3. Memory error detection
compute-sanitizer --tool memcheck python my_script.py

# 4. Race condition detection
compute-sanitizer --tool racecheck python my_script.py

# 5. Profile kernel performance
nv-nsight-cu-cli --metrics all python my_script.py
```

**PyTorch memory diagnostics:**
```python
# Complete memory diagnostic
import torch
import gc

def diagnose_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Detailed summary
    print(torch.cuda.memory_summary(abbreviated=False))

    # List tensors
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(f"{type(obj)}, {obj.size()}, {obj.device}")

# Run after suspected memory leak
diagnose_memory()
```

---

## Sources

**PyTorch Documentation:**
- [CUDA semantics - PyTorch](https://pytorch.org/docs/stable/notes/cuda.html) (accessed 2025-11-13)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) (accessed 2025-11-13)

**Stack Overflow & Forums:**
- [How to avoid "CUDA out of memory" in PyTorch](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-11-13)
- [PyTorch CUDA error: an illegal memory access](https://stackoverflow.com/questions/68106457/pytorch-cuda-error-an-illegal-memory-access-was-encountered) (accessed 2025-11-13)
- [PyTorch Forums: Systematically debugging OOM](https://discuss.pytorch.org/t/systematically-debugging-out-of-memory-issue/175034) (accessed 2025-11-13)
- [PyTorch Forums: CUDA error illegal memory access](https://discuss.pytorch.org/t/cuda-error-an-illegal-memory-access-was-encountered-with-reproduction/135532) (accessed 2025-11-13)

**NVIDIA Documentation:**
- [Using Nsight Compute to Inspect your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels) (accessed 2025-11-13)
- [NVIDIA Developer Forums: Kernel launch failures](https://forums.developer.nvidia.com/t/cudalaunchkernel-failed-to-launch-kernel/211624) (accessed 2025-11-13)

**Source Documents:**
- [cuda/01-memory-management-unified.md](01-memory-management-unified.md) - Memory allocation strategies
- [cuda/02-pytorch-build-system-compilation.md](02-pytorch-build-system-compilation.md) - PyTorch build system

---

**Document Version**: 1.0
**Created**: 2025-11-13
**Coverage**: OOM errors, illegal memory access (error 700/77), kernel launch failures, Nsight Compute profiling, compute-sanitizer, PyTorch memory debugging
