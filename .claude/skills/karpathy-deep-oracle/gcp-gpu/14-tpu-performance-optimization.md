# TPU Performance Optimization

## Overview

TPU (Tensor Processing Unit) performance optimization requires understanding JAX JIT compilation, XLA compiler optimizations, TPU memory hierarchy, and profiling tools. This guide covers optimization techniques for maximizing TPU utilization in training and inference workloads.

**Key Concepts:**
- JAX JIT compilation transforms Python functions into highly optimized XLA code
- XLA compiler performs fusion, layout optimization, and kernel specialization
- TPU memory hierarchy: HBM (High Bandwidth Memory) + on-chip SRAM
- Matrix Unit (MXU) efficiency is critical for achieving peak FLOPS
- TensorBoard profiling reveals performance bottlenecks

From [JAX JIT Compilation Documentation](https://docs.jax.dev/en/latest/jit-compilation.html) (accessed 2025-11-16):
- JAX uses XLA compiler to translate code into optimized machine code for CPU, GPU, or TPU
- JIT compilation happens on first call, subsequent calls use cached compiled code
- Traced values can only affect control flow via static attributes (shape, dtype), not values

From [Google TPU v6e Performance Analysis](https://introl.com/blog/google-tpu-v6e-vs-gpu-4x-better-ai-performance-per-dollar-guide) (accessed 2025-11-16):
- Operator fusion reduces memory bandwidth requirements
- Layout optimization ensures efficient utilization of tensor cores
- TPU v6e shows 4x better performance per dollar compared to GPUs

---

## Section 1: JAX JIT Compilation for TPU (~100 lines)

### What is JIT Compilation?

Just-In-Time (JIT) compilation transforms Python functions into highly optimized XLA code that executes directly on TPU hardware without Python interpreter overhead.

From [JAX Documentation](https://docs.jax.dev/en/latest/jit-compilation.html):

**How JIT Works:**
```python
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Without JIT: 3.6 ms per iteration
x = jnp.arange(1000000)
selu(x).block_until_ready()

# With JIT: 280 μs per iteration (12.8x faster!)
selu_jit = jax.jit(selu)
selu_jit(x).block_until_ready()
```

**JIT Compilation Process:**
1. **Tracing**: JAX wraps arguments in tracers, records all operations
2. **jaxpr Creation**: Sequence of primitives representing computation
3. **XLA Compilation**: jaxpr compiled into optimized TPU code
4. **Caching**: Compiled code cached for subsequent calls
5. **Execution**: Cached code executes directly on TPU

### JAX Tracing and jaxpr

From [JAX JIT Documentation](https://docs.jax.dev/en/latest/jit-compilation.html):

```python
def log2(x):
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2

# View the jaxpr (JAX intermediate representation)
print(jax.make_jaxpr(log2)(3.0))
# Output:
# { lambda ; a:f32[]. let
#     b:f32[] = log a
#     c:f32[] = log 2.0:f32[]
#     d:f32[] = div b c
#   in (d,) }
```

**Key Points:**
- jaxpr captures function as sequence of primitives
- No Python side-effects appear in jaxpr (only pure computation)
- jaxpr is input to XLA compiler for optimization

### Static vs Dynamic Arguments

From [JAX JIT Documentation](https://docs.jax.dev/en/latest/jit-compilation.html):

**Problem: Control flow on traced values fails**
```python
def f(x):
    if x > 0:  # ERROR: can't condition on traced value!
        return x
    else:
        return 2 * x

jax.jit(f)(10)  # TracerBoolConversionError
```

**Solution: Mark arguments as static**
```python
# Use static_argnums for specific argument indices
f_jit = jax.jit(f, static_argnums=0)
print(f_jit(10))  # Works!

# Use static_argnames for named arguments
@partial(jax.jit, static_argnames=['n'])
def loop_fn(x, n):
    i = 0
    while i < n:
        i += 1
    return x + i

print(loop_fn(10, 20))  # Works!
```

**Tradeoffs:**
- Static args enable control flow but trigger recompilation per unique value
- Only use for arguments with limited value sets (flags, small integers)
- Excessive static args cause compilation overhead

---

## Section 2: XLA Compiler Optimization Passes (~150 lines)

### XLA Compilation Pipeline

From [OpenXLA Documentation](https://openxla.org/xla) (accessed 2025-11-16):

**XLA Compilation Stages:**
1. **HLO Generation**: High-Level Operations from jaxpr
2. **Optimization Passes**: Fusion, layout, constant folding
3. **Backend Code Generation**: TPU-specific machine code
4. **Kernel Specialization**: Optimized for specific shapes

**Key Optimizations:**
- **Operator Fusion**: Combine multiple ops into single kernel (reduce memory traffic)
- **Layout Optimization**: Arrange tensors for efficient TPU memory access
- **Constant Folding**: Pre-compute constant expressions at compile time
- **Algebraic Simplification**: Simplify mathematical expressions
- **Dead Code Elimination**: Remove unused computations

### Operator Fusion

From [Modular AI Compiler Analysis](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers) (accessed 2025-11-16):

**What is Fusion?**
Combines multiple operations into a single kernel to minimize memory bandwidth.

**Example:**
```python
# Unfused: 3 separate kernels
x = a + b      # Kernel 1: Load a,b -> Store x
y = x * c      # Kernel 2: Load x,c -> Store y
z = jnp.tanh(y) # Kernel 3: Load y -> Store z
# Total: 6 memory accesses (3 loads + 3 stores)

# XLA Fused: 1 kernel
z = jnp.tanh((a + b) * c)
# Total: 3 memory accesses (1 load a,b,c -> 1 compute -> 1 store z)
# 2x reduction in memory traffic!
```

**Fusion Benefits:**
- Reduces global memory bandwidth (critical on TPU)
- Increases arithmetic intensity (compute/memory ratio)
- Enables compiler to keep intermediate values in registers/SRAM

### Layout Optimization

From [Google TPU v6e Analysis](https://introl.com/blog/google-tpu-v6e-vs-gpu-4x-better-ai-performance-per-dollar-guide):

**Memory Layout Matters:**
TPU Matrix Units (MXU) require specific tensor layouts for maximum efficiency.

**XLA Layout Transformations:**
```python
# Row-major (inefficient for TPU MXU)
A = jnp.array([[1, 2, 3], [4, 5, 6]])  # [2, 3] shape

# XLA automatically transposes for TPU MXU efficiency
# Column-major or tiled layouts optimized for 128x128 MXU tiles
```

**Layout Considerations:**
- TPU v5e/v5p have 128x128 systolic arrays
- XLA tiles matrices into 128x128 blocks
- Padding added to align with tile boundaries
- Minimize layout conversions (expensive memory operations)

### Constant Folding and Simplification

**Compile-Time Optimization:**
```python
# Before optimization
def model(x):
    scale = 2.0 * 3.0  # Constant expression
    return x * scale

# After XLA optimization
def model_optimized(x):
    return x * 6.0  # Pre-computed at compile time
```

**Algebraic Simplification:**
```python
# Before
y = x * 1.0 + 0.0

# After XLA optimization
y = x  # Identity operations removed
```

---

## Section 3: TPU Memory Layout and HBM Optimization (~150 lines)

### TPU Memory Hierarchy

From [TPU Architecture Technical Analysis](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf) (accessed 2025-11-16):

**Memory Levels:**
1. **HBM (High Bandwidth Memory)**: Main memory (16-96 GB depending on TPU generation)
2. **Vector Memory (VMEM)**: On-chip SRAM for intermediate activations
3. **Scalar Memory (SMEM)**: On-chip SRAM for small scalars, loop indices
4. **Matrix Registers**: Fast registers directly feeding Matrix Units

**TPU v5e Memory Specs:**
- HBM: 16 GB per chip, 819 GB/s bandwidth
- VMEM: 256 MB on-chip SRAM
- Peak FLOPS: 197 TFLOPs (BF16)

**TPU v5p Memory Specs:**
- HBM: 95 GB per chip, 2765 GB/s bandwidth
- VMEM: Larger on-chip SRAM (exact size undisclosed)
- Peak FLOPS: 459 TFLOPs (BF16)

From [HBM Scaling Analysis](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm) (accessed 2025-11-16):

**HBM Evolution:**
- HBM2e: 3.2 Gbps per pin (current TPU v4)
- HBM3: 6.4 Gbps per pin (TPU v5e)
- HBM3e: 9.6 Gbps per pin (TPU v5p, future v6)
- HBM4 (2026): 12.8 Gbps per pin

### Optimizing for HBM Bandwidth

**Memory-Bound vs Compute-Bound:**
```python
# Memory-bound: Low arithmetic intensity
y = x + 1  # 1 FLOP per memory access
# HBM bandwidth is bottleneck

# Compute-bound: High arithmetic intensity
y = jnp.matmul(A, B)  # 2*M*N*K FLOPs for M*K + K*N memory
# Matrix Unit utilization is bottleneck
```

**Arithmetic Intensity Calculation:**
```
AI = (Total FLOPs) / (Total Bytes Transferred)

For matmul [M, K] @ [K, N] = [M, N]:
FLOPs = 2 * M * N * K
Bytes = (M*K + K*N + M*N) * sizeof(dtype)

Example: M=N=K=4096, BF16 (2 bytes)
AI = 2*4096^3 / (3*4096^2*2) = ~2731 FLOPs/byte

TPU v5p HBM bandwidth: 2765 GB/s
Peak achievable: 2765 * 2731 = 7.5 PFLOPs
Actual TPU peak: 459 TFLOPs
Conclusion: Matmul is compute-bound on TPU!
```

### Data Loading Patterns

From [Google Cloud TPU Data Loading Guide](https://cloud.google.com/tpu/docs/performance-guide):

**Efficient Data Loading:**
```python
# BAD: Eager data loading (HBM thrashing)
for batch in dataset:
    batch = jax.device_put(batch)  # Synchronous transfer
    loss = train_step(batch)

# GOOD: Asynchronous prefetch
dataset_iter = iter(dataset)
batch = next(dataset_iter)
batch = jax.device_put(batch)  # Async transfer

for i in range(num_steps):
    next_batch = next(dataset_iter)
    next_batch = jax.device_put(next_batch)  # Overlap with compute
    loss = train_step(batch)  # Compute while next batch transfers
    batch = next_batch
```

**VMEM Usage Patterns:**
- Store frequently accessed activations in VMEM
- Minimize HBM read/write operations
- Use Pallas custom kernels for explicit VMEM control

---

## Section 4: Matrix Unit Utilization and MXU Efficiency (~100 lines)

### TPU Matrix Units (MXU)

From [TPU Architecture Analysis](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf):

**MXU Architecture:**
- **Systolic Array**: 128x128 grid of multiply-accumulate units
- **BF16 Operations**: 128×128×128 = 2M BF16 ops per cycle
- **INT8 Operations**: Higher throughput for quantized models
- **Pipelining**: Multiple operations in-flight simultaneously

**Peak Performance:**
```
TPU v5e (BF16):
Clock: 1.05 GHz
MXU: 128×128 array
FLOPs/cycle = 2 * 128 * 128 = 32,768
Peak = 32,768 * 1.05 GHz = 197 TFLOPs

TPU v5p (BF16):
Clock: Higher frequency (undisclosed)
Peak = 459 TFLOPs
```

### Maximizing MXU Utilization

**Tile Size Matters:**
```python
# BAD: Small matmul (poor MXU utilization)
A = jnp.ones([64, 64], dtype=jnp.bfloat16)
B = jnp.ones([64, 64], dtype=jnp.bfloat16)
C = A @ B  # Only 50% MXU utilization (64 < 128)

# GOOD: Large matmul (high MXU utilization)
A = jnp.ones([4096, 4096], dtype=jnp.bfloat16)
B = jnp.ones([4096, 4096], dtype=jnp.bfloat16)
C = A @ B  # 95%+ MXU utilization
```

**Batch Operations:**
```python
# BAD: Sequential small matmuls
for i in range(batch_size):
    C[i] = A[i] @ B[i]  # Low utilization per iteration

# GOOD: Batched matmul
C = jax.vmap(lambda a, b: a @ b)(A, B)  # XLA fuses into large op
# Or use einsum:
C = jnp.einsum('bik,bkj->bij', A, B)  # Batch matrix multiply
```

### Mixed Precision Training

From [JAX Mixed Precision Guide](https://apxml.com/posts/reasons-to-learn-jax) (accessed 2025-11-16):

**BF16 Training Benefits:**
```python
# FP32: 19.5 TFLOPs on TPU v4
# BF16: 275 TFLOPs on TPU v4 (14x faster!)

import jax
import jax.numpy as jnp

# Automatic mixed precision
def train_step(params, batch):
    # Compute in BF16
    logits = model_bf16(params, batch['inputs'])
    loss = jnp.mean((logits - batch['labels'])**2)
    # Accumulate gradients in FP32 for stability
    return loss

# Explicit dtype control
params_bf16 = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
```

**Precision Recommendations:**
- **Weights**: BF16 (8-bit exponent like FP32, better range than FP16)
- **Activations**: BF16
- **Gradients**: FP32 accumulation (prevent underflow)
- **Loss**: FP32 (precision matters for small values)

---

## Section 5: JAX Profiling with TensorBoard (~100 lines)

### Profiling TPU Workloads

From [Google Cloud TPU Profiling Guide](https://cloud.google.com/tpu/docs/profile-tpu-vm) (accessed 2025-11-16):

**XProf Profiler:**
XProf is the core profiling tool for TPU workloads, integrated with TensorBoard.

**Capture Profile:**
```python
import jax
import jax.numpy as jnp
from jax import random

# Your training code
def train_step(params, batch):
    # ... training logic ...
    return loss, new_params

# Profile the training step
jax.profiler.start_trace("./tensorboard_logs")

for i in range(100):
    loss, params = train_step(params, batch)
    if i == 10:  # Capture after warmup
        jax.profiler.start_trace("./tensorboard_logs")
    if i == 20:
        jax.profiler.stop_trace()

jax.profiler.stop_trace()
```

**View in TensorBoard:**
```bash
tensorboard --logdir=./tensorboard_logs
# Open browser: http://localhost:6006
```

From [TensorBoard Profiler Plugin Guide](https://pypi.org/project/tensorboard-plugin-profile/):

**Key Metrics to Monitor:**
1. **TPU Utilization**: % of time TPU is actively computing
2. **MXU Utilization**: % of Matrix Unit capacity used
3. **Memory Bandwidth**: HBM read/write throughput
4. **Step Time**: Total time per training iteration
5. **Compilation Time**: XLA compilation overhead

### Profiling Visualizations

**Trace Viewer:**
- Timeline of TPU operations
- Identify idle time (compilation, data loading)
- See operation dependencies

**Op Profile:**
- Which operations consume most time
- Memory bandwidth per operation
- FLOP utilization per operation

**Memory Profile:**
- HBM allocation over time
- Peak memory usage
- Memory fragmentation

**Example Bottleneck Identification:**
```
Trace Viewer shows:
- 60% idle time between steps -> Data loading bottleneck
  Solution: Async data prefetch

- High "all_reduce" time -> Communication bottleneck
  Solution: Gradient accumulation to reduce sync frequency

- Low MXU utilization (30%) -> Small batch size
  Solution: Increase batch size or sequence length
```

---

## Section 6: Common TPU Performance Bottlenecks (~100 lines)

### Bottleneck 1: Host-to-TPU Transfer

**Symptom:**
Trace viewer shows long gaps between training steps.

**Diagnosis:**
```python
# Measure transfer time
import time

batch = next(dataloader)
start = time.time()
batch_device = jax.device_put(batch)
jax.block_until_ready(batch_device)
transfer_time = time.time() - start
print(f"Transfer time: {transfer_time:.3f}s")
```

**Solutions:**
```python
# 1. Async prefetch (shown earlier)
# 2. Use TensorFlow Datasets with proper prefetch
import tensorflow_datasets as tfds

ds = tfds.load('imagenet2012', split='train')
ds = ds.prefetch(tf.data.AUTOTUNE)  # Automatic async prefetch

# 3. Multi-process data loading
import multiprocessing as mp
with mp.Pool(8) as pool:
    batches = pool.map(preprocess, raw_data)
```

### Bottleneck 2: Compilation Overhead

**Symptom:**
First iteration very slow, subsequent iterations fast.

**Diagnosis:**
```python
# Time compilation
@jax.jit
def train_step(params, batch):
    # ... training logic ...
    return loss, new_params

# First call: compile + execute
start = time.time()
loss, params = train_step(params, batch)
jax.block_until_ready(loss)
first_time = time.time() - start

# Second call: execute only (from cache)
start = time.time()
loss, params = train_step(params, batch)
jax.block_until_ready(loss)
second_time = time.time() - start

print(f"Compilation overhead: {first_time - second_time:.3f}s")
```

**Solutions:**
```python
# 1. Persistent compilation cache
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

# 2. Ahead-of-time compilation
from jax import jit, aot

compiled_fn = jit(train_step).lower(params, batch).compile()
# Save compiled artifact for reuse

# 3. Avoid recompilation in loops (shown in JIT caching section)
```

### Bottleneck 3: Small Batch Sizes

**Symptom:**
Low MXU utilization in profiler.

**Diagnosis:**
```python
# Check effective batch size
print(f"Batch shape: {batch['inputs'].shape}")
# [32, 224, 224, 3] -> Batch size 32 might be too small
```

**Solutions:**
```python
# 1. Increase batch size
batch_size = 256  # instead of 32

# 2. Gradient accumulation
accumulation_steps = 8
effective_batch = 32 * 8 = 256

def train_step_accumulated(params, batches):
    grad_acc = jax.tree_map(jnp.zeros_like, params)
    for batch in batches:
        grads = jax.grad(loss_fn)(params, batch)
        grad_acc = jax.tree_map(lambda a, b: a + b, grad_acc, grads)
    return jax.tree_map(lambda g: g / len(batches), grad_acc)

# 3. Increase sequence length (for transformers)
seq_len = 2048  # instead of 512
```

---

## Section 7: Advanced Optimization Techniques (~100 lines)

### Custom Kernels with Pallas

From [Pallas Custom Kernels Article](https://medium.com/data-science/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a) (accessed 2025-11-16):

**When to Use Pallas:**
- Operations not efficiently supported by XLA
- Need explicit VMEM management
- Custom fused kernels for domain-specific ops

**Example: Flash Attention on TPU**
```python
import jax
from jax.experimental import pallas as pl

def flash_attention_kernel(q_ref, k_ref, v_ref, o_ref):
    # Custom kernel with explicit VMEM usage
    # ... low-level TPU operations ...
    pass

# Use Pallas kernel
@pl.pallas_call(
    out_shape=jax.ShapeDtypeStruct((seq_len, d_model), jnp.bfloat16),
    grid=(batch_size, num_heads)
)
def flash_attention(q, k, v):
    return flash_attention_kernel(q, k, v)
```

### TPU Pod Slice Optimization

From [TPU Multi-Host Training](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm):

**Multi-Host Sharding:**
```python
import jax
from jax.experimental import multihost_utils

# Shard data across TPU hosts
def shard_batch(batch, num_hosts):
    local_batch = jax.tree_map(
        lambda x: x[jax.process_index()::num_hosts],
        batch
    )
    return local_batch

# Synchronize across hosts
loss = train_step(params, local_batch)
global_loss = multihost_utils.process_allreduce(loss, op='mean')
```

**Communication Optimization:**
- Use `jax.pmap` for data parallelism across devices
- Use `jax.sharding` for model parallelism
- Minimize cross-host communication (expensive)

### Memory Optimization

**Gradient Checkpointing:**
```python
from jax import checkpoint

# Recompute activations instead of storing (trade compute for memory)
@checkpoint
def transformer_block(x, params):
    # ... expensive computation ...
    return output

# Saves ~50% activation memory at cost of 33% more compute
```

**Dynamic Batch Sizing:**
```python
# Adjust batch size based on sequence length
def get_batch_size(seq_len):
    # Fit within HBM limit
    max_tokens = 1_000_000
    return max_tokens // seq_len

batch_size = get_batch_size(sequence_length)
```

---

## Section 8: TPU-Specific Best Practices (~50 lines)

### Recommended Settings

**JAX Configuration:**
```python
import jax

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

# Enable XLA optimizations
jax.config.update("jax_xla_backend", "tpu")

# Disable CPU fallback (catch TPU-incompatible ops)
jax.config.update("jax_platforms", "tpu")

# BF16 matmul precision
jax.config.update("jax_default_matmul_precision", "bfloat16")
```

**Data Format:**
```python
# Use NHWC format for images (TPU-optimized)
images = jnp.ones([batch, height, width, channels])  # Not NCHW

# Use BF16 dtype
params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
```

### Debugging Performance Issues

**Quick Checklist:**
1. ✓ Compilation cached? (persistent cache enabled)
2. ✓ Batch size large? (>128 for most models)
3. ✓ Using BF16? (14x faster than FP32)
4. ✓ Async data loading? (overlap transfer & compute)
5. ✓ MXU utilization >80%? (profile with TensorBoard)
6. ✓ Minimal host-TPU transfers? (keep data on device)
7. ✓ Avoid dynamic shapes? (triggers recompilation)

**Profile-Guided Optimization:**
```
1. Run with profiler: jax.profiler.start_trace()
2. Identify top bottleneck in TensorBoard
3. Fix bottleneck (see common bottlenecks section)
4. Re-profile to verify improvement
5. Repeat until target performance achieved
```

---

## Sources

### Source Documents

- [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) - Tensor Core patterns (GPU comparison context)

### Web Research

**Primary Sources:**

- [JAX JIT Compilation Documentation](https://docs.jax.dev/en/latest/jit-compilation.html) (accessed 2025-11-16)
  - JIT compilation workflow and tracing mechanism
  - Static vs dynamic arguments
  - Caching behavior and performance implications

- [Google TPU v6e Performance Guide](https://introl.com/blog/google-tpu-v6e-vs-gpu-4x-better-ai-performance-per-dollar-guide) (accessed 2025-11-16)
  - Operator fusion and layout optimization
  - 4x performance per dollar vs GPUs
  - Real-world deployment metrics

- [TPU Architecture Technical Paper](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf) (accessed 2025-11-16)
  - HBM integration and memory hierarchy
  - Matrix Unit architecture (128x128 systolic array)
  - TPU v5e/v5p specifications

- [HBM Scaling and Roadmap Analysis](https://newsletter.semianalysis.com/p/scaling-the-memory-wall-the-rise-and-roadmap-of-hbm) (accessed 2025-11-16)
  - HBM2e/HBM3/HBM3e/HBM4 evolution
  - Memory bandwidth scaling for AI workloads
  - Future TPU memory projections

- [Google Cloud TPU Profiling Guide](https://cloud.google.com/tpu/docs/profile-tpu-vm) (accessed 2025-11-16)
  - XProf profiler usage
  - TensorBoard integration for TPU
  - Performance metrics and visualizations

- [PyTorch/XLA TPU Profiling](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm) (accessed 2025-11-16)
  - Multi-framework profiling support
  - XProf trace capture for PyTorch workloads

- [TensorBoard Profiler Plugin](https://pypi.org/project/tensorboard-plugin-profile/) (accessed 2025-11-16)
  - Profile visualization tools
  - Trace viewer and op profile features

- [OpenXLA Compiler Documentation](https://openxla.org/xla) (accessed 2025-11-16)
  - XLA optimization passes
  - Backend code generation for TPU

- [Modular AI Compiler Analysis](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers) (accessed 2025-11-16)
  - XLA optimization priorities
  - TPU vs GPU compiler differences

- [Pallas Custom Kernels Article](https://medium.com/data-science/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a) (accessed 2025-11-16)
  - JAX extension for custom TPU kernels
  - GPU and TPU support via Pallas

- [JAX Learning Resources](https://apxml.com/posts/reasons-to-learn-jax) (accessed 2025-11-16)
  - JIT compilation benefits
  - Mixed precision training on accelerators

**Additional References:**

- Stack Overflow: [Check TPU workload utilization](https://stackoverflow.com/questions/52427141/check-tpu-workload-utilization) - Community TPU profiling tips
- Google Cloud Blog: [PyTorch/XLA 2.4 Pallas improvements](https://cloud.google.com/blog/products/ai-machine-learning/pytorch-xla-2-4-improves-pallas-and-adds-eager-mode/)
- Research: [Leveraging Compute-in-Memory for TPU Architecture](https://arxiv.org/pdf/2503.00461) (arXiv:2503.00461)

---

## Practical Recommendations

### Optimization Priority Order

**1. Enable JIT Compilation (First!):**
- Wrap training step in `@jax.jit`
- 10-100x speedup with minimal code changes

**2. Use BF16 Precision:**
- 14x faster on TPU vs FP32
- Minimal accuracy impact for most models

**3. Optimize Batch Size:**
- Target >80% MXU utilization
- Profile to find sweet spot (balance memory & throughput)

**4. Async Data Loading:**
- Overlap data transfer with compute
- Use TensorFlow Datasets with prefetch

**5. Profile and Iterate:**
- Use TensorBoard profiler to find bottlenecks
- Fix top bottleneck, re-profile, repeat

### When to Use TPU vs GPU

**Choose TPU when:**
- Large matrix multiplications dominate (transformers, LLMs)
- Training large models at scale (multi-host pods)
- BF16 precision acceptable
- JAX/TensorFlow ecosystem

**Choose GPU when:**
- Custom CUDA kernels required
- PyTorch ecosystem preferred
- Mixed workloads (training + inference)
- FP64 precision needed

### TPU Performance Targets

**Utilization Goals:**
- MXU: >80% during training steps
- HBM bandwidth: >70% of peak (for memory-bound ops)
- Compilation overhead: <5% of total training time
- Data loading overhead: <10% of step time

**Benchmark Comparison:**
```
ResNet-50 Training (ImageNet, batch=1024, BF16):
- TPU v5e: ~8,000 images/sec
- TPU v5p: ~18,000 images/sec
- A100 GPU: ~6,000 images/sec

GPT-3 Training (175B params, BF16):
- TPU v4 Pod (4096 chips): ~500 TFLOPs/chip sustained
- A100 GPU (8-way): ~200 TFLOPs/GPU sustained
```

### Common Pitfalls to Avoid

**❌ Don't:**
- Define JIT functions inside loops (kills caching)
- Use dynamic shapes (triggers recompilation)
- Transfer small batches frequently (HBM thrashing)
- Ignore profiler warnings (hints at inefficiencies)

**✓ Do:**
- Cache compiled functions globally
- Use static shapes throughout pipeline
- Batch operations aggressively
- Profile regularly during development
