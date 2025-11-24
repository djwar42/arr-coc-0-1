# TPU Programming Fundamentals

**The essentials of programming Google's Tensor Processing Units for machine learning workloads**

## Overview

Tensor Processing Units (TPUs) are Google's custom-designed, application-specific integrated circuits (ASICs) optimized for machine learning workloads. Unlike GPUs which are general-purpose accelerators, TPUs are purpose-built for tensor operations with a focus on matrix multiplication - the core operation in neural network training and inference.

**Key insight**: A TPU is essentially a matrix multiplication machine connected to fast memory. Understanding this simple model is crucial for writing efficient TPU code.

From [Introduction to Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) (Google Cloud Documentation, accessed 2025-11-13):
- TPUs are custom ASICs designed specifically for accelerating machine learning workloads
- Available through Google Cloud Platform as Cloud TPU resources
- Support both training and inference with different optimized versions

From [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) (JAX ML Scaling Book, accessed 2025-11-13):
> "A TPU is basically a compute core that specializes in matrix multiplication (called a TensorCore) attached to a stack of fast memory (called high-bandwidth memory or HBM)."

## TPU Architecture

### Core Components

**TensorCore** - The main compute unit containing:

1. **MXU (Matrix Multiply Unit)**
   - Core computational engine performing matrix multiplication
   - For most TPU generations: performs `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` every 8 cycles
   - Uses systolic array architecture (128x128 for v4/v5, 256x256 for v6e Trillium)
   - TPU v5e: ~5e13 bf16 FLOPs/s per MXU at 1.5GHz
   - Most TensorCores have 2-4 MXUs

2. **VPU (Vector Processing Unit)**
   - Performs general mathematical operations: ReLU, pointwise add/multiply, reductions
   - 2D SIMD vector machine with shape (8, 128)
   - 128 dimension = lane axis, 8 dimension = sublane axis
   - Each (lane, sublane) pair contains 4 floating-point ALUs
   - TPU v5p: ~1.4e13 FLOPs/s (much smaller than MXU at ~2e14 FLOPs/s)

3. **VMEM (Vector Memory)**
   - On-chip scratchpad memory close to compute units
   - Much smaller than HBM (e.g., 128 MiB on TPU v5e)
   - Much higher bandwidth to MXU than HBM
   - Programmer-controlled (not automatic like CPU cache)
   - Data must be copied from HBM to VMEM before computation

**High Bandwidth Memory (HBM)** - Main memory storage:
- Stores tensors for use by TensorCore
- Capacity: tens of gigabytes (16GB for v5e, 32GB for v4/v6e, 96GB for v5p)
- Bandwidth: 1-2TB/sec HBM ↔ TensorCore
- Tensors stream from HBM through VMEM to MXU

### Memory Hierarchy and Bandwidth

Understanding the memory hierarchy is crucial for TPU programming:

```
Fastest    → VMEM ↔ MXU        (22x faster than HBM bandwidth)
           → HBM ↔ VMEM        (1-2 TB/s)
           → ICI (Inter-Chip)  (90-180 GB/s)
           → PCIe (Host)       (~16 GB/s)
Slowest    → DCN (Data Center) (~6.25 GB/s)
```

From [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) (JAX ML Scaling Book):
> "VMEM bandwidth is around 22x higher than HBM bandwidth which means an MXU operation reading from/writing to VMEM requires an arithmetic intensity of only 10-20 to achieve peak FLOPs utilization."

**Key implication**: If weights fit in VMEM, matrix multiplications can be FLOPs bound at much smaller batch sizes.

## TPU Generations Comparison

### TPU v4 vs TPU v5 Performance

From [ByteBridge GPU and TPU Comparative Analysis](https://bytebridge.medium.com/gpu-and-tpu-comparative-analysis-report-a5268e4f0d2a) (Medium, accessed 2025-11-13):
> "TPU v4 offers 1.2–1.7x better performance per watt compared to NVIDIA A100 GPUs, resulting in substantial energy savings."

### TPU v5 Improvements

From [Introducing Cloud TPU v5p](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) (Google Cloud Blog, December 6, 2023):
> "Compared to TPU v4, TPU v5p features more than 2X greater FLOPS and 3X more high-bandwidth memory (HBM)."

### Specifications Table

| Model | Pod Size | HBM/chip | HBM BW/chip | FLOPs/s (bf16) | FLOPs/s (int8) |
|-------|----------|----------|-------------|----------------|----------------|
| TPU v3 | 32x32 | 32GB | 9.0e11 | 1.4e14 | 1.4e14 |
| TPU v4p | 16x16x16 | 32GB | 1.2e12 | 2.75e14 | 2.75e14 |
| TPU v5p | 16x20x28 | 96GB | 2.8e12 | 4.59e14 | 9.18e14 |
| TPU v5e | 16x16 | 16GB | 8.1e11 | 1.97e14 | 3.94e14 |
| TPU v6e | 16x16 | 32GB | 1.6e12 | 9.20e14 | 1.84e15 |

**Key differences**:
- **v5p** (training): 96GB HBM, 3D torus networking, highest performance
- **v5e** (inference): 16GB HBM, 2D torus, cost-optimized
- **v6e Trillium** (latest): 256x256 systolic array (vs 128x128), 2x performance

From [TPUv5e: The New Benchmark](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) (SemiAnalysis, September 1, 2023):
> "The TPUv5e only has a single Tensor Core, unlike TPU v5 which includes two. Furthermore it has half the HBM stacks and at lower speeds."

## Programming Frameworks

### JAX on TPU

JAX is Google's primary framework for TPU programming, providing NumPy-like interface with automatic differentiation and XLA compilation.

From [JAX Documentation Quickstart](https://docs.jax.dev/en/latest/quickstart.html) (accessed 2025-11-13):
> "JAX provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings."

**JAX advantages for TPUs**:
- Native XLA compilation (XLA is the TPU compiler)
- Automatic batching with `vmap`
- Automatic differentiation with `grad`
- Easy parallelization with `pmap` and `pjit`

**Basic JAX TPU example**:

```python
import jax
import jax.numpy as jnp

# Automatically uses TPU if available
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x.T)

# Check device
print(jax.devices())  # [TpuDevice(id=0), TpuDevice(id=1), ...]
```

From [Programming TPUs in JAX](https://jax-ml.github.io/scaling-book/jax-stuff/) (JAX ML Scaling Book):
- JAX makes TPU programming straightforward
- Free TPU access available through Google Colab and TPU Research Cloud
- Supports data sharding, JIT compilation, and manual parallelism

### PyTorch XLA on TPU

PyTorch XLA enables PyTorch models to run on TPUs through the XLA (Accelerated Linear Algebra) compiler.

From [PyTorch/XLA Documentation](https://docs.pytorch.org/xla/) (accessed 2025-11-13):
> "Create and train PyTorch models on TPUs, with only minimal changes required."

**Architecture**: PyTorch → XLA IR → TPU machine code

**Basic PyTorch XLA usage**:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Get TPU device
device = xm.xla_device()

# Move model and data to TPU
model = MyModel().to(device)
data = torch.randn(32, 3, 224, 224).to(device)

# Training step
output = model(data)
loss = criterion(output, labels)
loss.backward()

# CRITICAL: Mark step to trigger XLA compilation
xm.mark_step()
```

From [Learn about TPUs - PyTorch/XLA](https://docs.pytorch.org/xla/master/accelerators/tpu.html):
> "Google Cloud TPUs are custom-designed AI accelerators, which are optimized for training and inference of large AI models."

**Key PyTorch XLA concepts**:
1. **Lazy execution**: Operations are traced to a graph, then compiled
2. **mark_step()**: Triggers compilation and execution
3. **XLA compiler**: Converts PyTorch ops to XLA ops, then to TPU code

From [PyTorch/XLA Internal](https://www.youtube.com/watch?v=kdZwgRghy3M) (PyTorch Developer Day 2020):
> "PyTorch XLA converts PyTorch ops to XLA ops and then compiles the IR graph to machine code for execution on TPUs."

## XLA Compilation

**XLA (Accelerated Linear Algebra)** is the compiler that transforms high-level operations into efficient TPU code.

### How XLA Works

1. **Graph capture**: Framework operations → XLA IR graph
2. **Optimization**: Graph-level optimizations (fusion, constant folding, etc.)
3. **Code generation**: XLA IR → TPU-specific machine code
4. **Execution**: Compiled code runs on TPU hardware

**XLA optimizations**:
- **Operator fusion**: Combine multiple ops to reduce memory traffic
- **Layout optimization**: Arrange data for optimal TPU access patterns
- **Constant folding**: Evaluate constants at compile time
- **Buffer management**: Minimize HBM ↔ VMEM transfers

From [Deep Dive: Compiling deep learning models](https://www.youtube.com/watch?v=Oo07fFb-aH0) (YouTube, February 28, 2024):
> "Model compilation optimizes deep learning models by analyzing the graph, eliminating redundancy, and generating efficient code for different accelerators."

**Compilation considerations**:
- First run includes compilation overhead (can be seconds)
- Subsequent runs use cached compiled code
- Shape changes trigger recompilation
- Use fixed shapes when possible for best performance

## TPU Networking and Topologies

### ICI (Inter-Chip Interconnect)

TPUs connect to each other through direct chip-to-chip links forming a mesh topology.

**Connectivity patterns**:
- **TPU v3/v5e/v6e**: 4 nearest neighbors (2D torus)
- **TPU v4/v5p**: 6 nearest neighbors (3D torus)

From [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/):
> "TPU v4 and TPU v5p are connected to the nearest 6 neighbors (forming a 3D torus). Note these connections do not go through their hosts, they are direct links between chips."

**ICI bandwidth** (bidirectional):
- TPU v3: 2e11 bytes/s per link
- TPU v4p: 9e10 bytes/s per link
- TPU v5p: 1.8e11 bytes/s per link
- TPU v5e: 9e10 bytes/s per link
- TPU v6e: 1.8e11 bytes/s per link

**Topology implications**:
- Communication between distant chips requires multiple hops
- Wraparound links on axes of size 16+ reduce max distance
- Full cubes (4x4x4 for v4/v5p) get optical wraparound links

### Pod Sizes and Slices

**Pod** = Set of ICI-connected TPUs

Maximum pod sizes:
- TPU v4: 16x16x16 = 4,096 chips
- TPU v5p: 16x20x28 = 8,960 chips
- TPU v5e/v6e: 16x16 = 256 chips (2D only)

**Slice** = Subset of pod allocated to a job

Common slice sizes:
- Single host: 4x2 (v5e), 2x2x1 (v4/v5p)
- Small: 2x2x2, 4x4x4
- Large: Full pod (superpod)

From [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/):
> "A TPU chip typically (but not always) consists of two TPU cores which share memory and can be thought of as one large accelerator with twice the FLOPs (known as a 'megacore' configuration)."

### DCN (Data Center Network)

**DCN** connects different pods and hosts through standard networking.

**Bandwidth hierarchy**:
- HBM: 2.5 TB/s (fastest)
- ICI: 90-180 GB/s
- PCIe: 16 GB/s
- DCN: 6.25 GB/s (slowest)

**Multi-slice training**: Connect slices via DCN for training beyond single pod size. Since DCN is 100x slower than ICI, minimize cross-DCN communication.

## Systolic Array Deep Dive

The systolic array is the heart of the TPU's matrix multiplication performance.

### How Systolic Arrays Work

From [How to Think About TPUs - Appendix B](https://jax-ml.github.io/scaling-book/tpus/):

1. **Grid of ALUs**: 128x128 grid (16,384 ALUs) for v4/v5, 256x256 for v6e
2. **Data flow**: Weights flow down, activations flow in from left
3. **Diagonal loading**: Inputs loaded diagonally to maximize overlap
4. **Pipelined execution**: After initial bubble, continuous throughput

**Systolic array advantages**:
- Extremely high throughput for matrix multiplication
- Minimal data movement (weights stay local)
- Natural pipelining across multiple operations
- Energy efficient (data reuse)

**Performance**: One `bf16[8,128] @ bf16[128,128] -> f32[8,128]` per 8 cycles

**Key requirement**: Matrix dimensions should be multiples of 128 (256 for v6e) to fully utilize the array. Smaller dimensions are automatically padded.

### Pipeline Bubbles

Initial pipeline bubble occurs while loading first weights and activations diagonally. After this bubble, new inputs can be loaded without additional bubbles, enabling continuous throughput.

**Optimization**: Use large matrices (>>128) to amortize pipeline bubble overhead.

## Performance Optimization

### Memory Bandwidth Considerations

From [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/):

**Roofline model**: Performance is limited by either:
1. **FLOPs bound**: Compute capacity is bottleneck
2. **Memory bound**: HBM ↔ VMEM bandwidth is bottleneck

**Arithmetic intensity** = FLOPs / bytes transferred

For TPU v5e:
- HBM bandwidth: 8.1e11 bytes/s
- FLOPs capacity: 1.97e14 FLOPs/s (bf16)
- Required arithmetic intensity: ~240 FLOPs/byte to be FLOPs bound

**Matrix multiplication arithmetic intensity**:
For `[B, D] @ [D, F] -> [B, F]`:
- FLOPs: 2*B*D*F
- Bytes: 2*B*D + 2*D*F + 2*B*F (assuming bf16)
- Intensity: 2*B*D*F / (2*(B*D + D*F + B*F))

For B << D and F = 4D (common in transformers):
- Intensity ≈ 8*B*D² / (8*D² + 2*B*D) ≈ B (for large D)
- Need B > 240 to be FLOPs bound on v5e

**VMEM optimization**:
- VMEM bandwidth is 22x higher than HBM bandwidth
- Loading from VMEM requires only ~10-20 arithmetic intensity
- If weights fit in VMEM: can be FLOPs bound at B > 11

### Precision and Performance

Lower precision = higher throughput:

**TPU v5e example**:
- bf16: 1.97e14 FLOPs/s
- int8: 3.94e14 FLOPs/s (2x faster)

**TPU v5p example**:
- bf16: 4.59e14 FLOPs/s
- int8: 9.18e14 FLOPs/s (2x faster)

**Trade-off**: Lower precision may reduce model quality. Use bf16 for training, consider int8 for inference.

### Batch Size Tuning

From worked problems in [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/):

**Example**: Matrix multiply `int8[16384, 4096] @ int8[B, 4096]` on TPU v5e
- FLOPs bound when B > 271 (from HBM)
- FLOPs bound when B > 11 (from VMEM)

**Guidelines**:
1. Calculate arithmetic intensity for your operation
2. Compare to required intensity (FLOPs/s ÷ HBM bandwidth)
3. Increase batch size if memory bound
4. Use VMEM prefetching for small batches

### Communication Optimization

**Minimize cross-chip communication**:
1. Keep frequently-accessed data on same chip
2. Use model parallelism to distribute large models
3. Overlap computation with communication when possible
4. Prefer ICI over DCN for inter-chip transfers

**ICI latency**: ~1μs per hop

**Data transfer time** = (bytes / bandwidth) + (hops × latency)

Example: Transfer 1.7e7 bytes from TPU{0,0} to TPU{3,3} in 4x4 v5e slice:
- Bandwidth: 9e10 bytes/s (both axes)
- Hops: 6 (no wraparound in 4x4)
- Latency: 6μs + 188μs = 194μs total

## JAX Programming Patterns

### Data Parallelism with pmap

```python
import jax
import jax.numpy as jnp
from jax import pmap

# Replicate model across all TPU cores
@pmap
def train_step(state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Batch data across devices
batch_per_device = jnp.reshape(batch, (n_devices, -1, ...))
state, loss = train_step(state, batch_per_device)
```

### Model Parallelism with pjit

```python
from jax.experimental import pjit
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P

# Define device mesh
mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))

# Shard large matrix across devices
@pjit(
    in_axis_resources=(P('x', None), P(None, 'y')),
    out_axis_resources=P('x', 'y')
)
def sharded_matmul(a, b):
    return jnp.dot(a, b)
```

### Efficient Compilation

```python
# Use static_argnums for shape-determining arguments
@jax.jit(static_argnums=(1,))
def process(data, length):
    return data[:length].sum()

# Avoid dynamic shapes
x = jnp.ones((100,))  # ✓ Static shape
y = jnp.ones(x.shape)  # ✗ Dynamic shape (triggers recompilation)

# Use fixed-size padding when needed
padded = jnp.pad(x, (0, max_len - len(x)))
```

## PyTorch XLA Programming Patterns

### Basic Training Loop

From [Training on a TPU in parallel using PyTorch XLA](https://medium.com/data-science/training-on-a-tpu-parallelly-using-pytorch-xla-4afef63ee7ac) (Medium, accessed 2025-11-13):

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
model = MyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Wrap dataloader for parallel loading
para_loader = pl.ParallelLoader(train_loader, [device])

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(para_loader.per_device_loader(device)):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Synchronize across cores
        xm.optimizer_step(optimizer)

        # Mark step for XLA compilation
        xm.mark_step()
```

### Multi-Core Training

```python
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(rank, flags):
    device = xm.xla_device()
    model = MyModel().to(device)
    # Training code here...

# Spawn processes on all TPU cores
xmp.spawn(_mp_fn, args=(flags,), nprocs=8)
```

### Performance Profiling

From [Profile PyTorch XLA workloads](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm) (Google Cloud):

```python
import torch_xla.debug.profiler as xp

# Start profiling
server = xp.start_server(9012)

# Training code here...

# View traces at http://localhost:9012
```

## TPU vs GPU Comparison

From [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) (CloudOptimo, April 15, 2025):

**TPU advantages**:
- **Matrix multiplication**: 2-5x faster than equivalent GPUs
- **Memory bandwidth**: Higher HBM bandwidth per FLOP
- **Power efficiency**: 1.2-1.7x better performance per watt
- **Scalability**: Direct chip-to-chip interconnects enable massive pods
- **Cost**: Often cheaper for equivalent training throughput

**GPU advantages**:
- **Flexibility**: Better for diverse workloads beyond matmul
- **Ecosystem**: More mature tooling and libraries
- **Debugging**: Better profiling and visualization tools
- **General compute**: Can handle non-ML workloads

**When to use TPUs**:
1. Training large transformer models (LLMs, vision transformers)
2. Workloads dominated by matrix multiplication
3. Need for massive scale (1000+ accelerators)
4. Cost-sensitive production inference

**When to use GPUs**:
1. Diverse workloads with varied compute patterns
2. Heavy reliance on ecosystem libraries
3. Need for CUDA-specific optimizations
4. Smaller scale deployments (<100 accelerators)

From [Inside Google's TPU: Architecture and GPU Comparisons](https://skymod.tech/inside-googles-tpu-and-gpu-comparisons/) (SkyMod, August 13, 2025):
> "In essence, TPUs pack more memory per system and deliver extremely high throughput for AI workloads, often surpassing equivalent GPU setups in training and inference tasks."

## Practical TPU Access

### Google Colab TPU

Free TPU access for prototyping:

```python
# In Colab notebook
# Runtime → Change runtime type → TPU

import jax
print(jax.devices())  # Should show TPU devices
```

From [Easy TPU Development: Getting Started](https://www.youtube.com/watch?v=0fzDm2vXynw) (YouTube, February 26, 2025):
> "Use Google Colab to run notebooks directly on TPU accelerators or Google Cloud Compute Engine for more flexibility and larger TPU configurations."

### Cloud TPU VM

For production workloads:

```bash
# Create TPU VM
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base

# SSH to TPU VM
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b

# Install dependencies
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

### TPU Research Cloud

Free TPU access for researchers through [TPU Research Cloud](https://sites.research.google/trc/about/).

From [Free TPU Access & JAX/PyTorch Setup](https://www.youtube.com/watch?v=PwYHoiB4Fag) (YouTube, November 11, 2024):
> "Set up TPU virtual machines on Google Cloud for research using the TPU Research Cloud program."

## Common Pitfalls and Best Practices

### Avoid Dynamic Shapes

```python
# ✗ BAD: Dynamic shape triggers recompilation every time
def bad_function(x):
    return x[:x.sum()]  # Output shape depends on data

# ✓ GOOD: Fixed shape with masking
def good_function(x, length):
    mask = jnp.arange(len(x)) < length
    return jnp.where(mask, x, 0)
```

### Minimize Host-Device Transfers

```python
# ✗ BAD: Transfers back to host every iteration
for i in range(1000):
    x = device_tensor.cpu().numpy()  # Slow!
    device_tensor = process(x)

# ✓ GOOD: Keep data on device
for i in range(1000):
    device_tensor = process(device_tensor)
final_result = device_tensor.cpu().numpy()  # Transfer once
```

### Use Appropriate Batch Sizes

```python
# ✗ BAD: Tiny batch, memory bound
batch_size = 4  # Too small, HBM bandwidth bottleneck

# ✓ GOOD: Large enough to be compute bound
batch_size = 512  # Arithmetic intensity sufficient

# Calculate minimum batch size:
# min_batch = (FLOPs/s) / (HBM_BW * arithmetic_intensity_per_sample)
```

### Pad to Multiples of 128

```python
# ✗ BAD: Irregular dimensions
hidden_dim = 1000  # Wastes systolic array capacity

# ✓ GOOD: Multiple of 128
hidden_dim = 1024  # 128 * 8, fully utilizes MXU

# For v6e Trillium, use multiples of 256
hidden_dim = 2048  # 256 * 8
```

### Mark Steps in PyTorch XLA

```python
# ✗ BAD: No mark_step(), compilation delayed
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    # XLA graph keeps growing!

# ✓ GOOD: Regular mark_step() calls
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    xm.mark_step()  # Trigger compilation and execution
```

## Advanced Topics

### VMEM Prefetching

Load weights into VMEM during other operations to hide latency:

```python
# In transformer forward pass:
# 1. Attention computation uses some weights
# 2. While attention runs, prefetch FFN weights to VMEM
# 3. FFN computation starts immediately (no HBM load wait)
```

This is typically handled by the XLA compiler, but understanding the concept helps with performance reasoning.

### Pallas for Custom Kernels

From [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html) (JAX Documentation):

Pallas enables writing custom TPU kernels in Python:

```python
import jax.experimental.pallas as pl

@pl.pallas_call(
    out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
    grid=1,
)
def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]
```

**Use cases**:
- Custom fusion patterns not handled by XLA
- Specialized algorithms with known access patterns
- Performance-critical kernels needing manual optimization

From [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html):
> "We believe that Pallas can make it easy to start writing TPU kernels, even without having a full understanding of the underlying hardware."

### Mixed Precision Training

```python
# JAX mixed precision
from jax.experimental import multihost_utils

@jax.jit
def train_step(state, batch):
    # Compute in bf16
    def loss_fn(params):
        logits = model.apply(params, batch['image'])
        return jnp.mean((logits - batch['label']) ** 2)

    # Gradients accumulated in fp32
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

**Benefits**:
- 2x throughput with bf16/int8 vs fp32
- Minimal accuracy degradation with bf16
- Important for large models (memory savings)

## Real-World Examples

### Training GPT-2 on TPU

From [Train a GPT2 model with JAX on TPU for free](https://developers.googleblog.com/en/train-gpt2-model-with-jax-on-tpu/) (Google Developers Blog, August 12, 2025):

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class GPT2(nn.Module):
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab_size, self.d_model)(tokens)
        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.d_model)(x)
        return nn.Dense(self.vocab_size)(x)

# Initialize on TPU
model = GPT2(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128), dtype=jnp.int32))

# Training loop on TPU v5e pod
for batch in dataloader:
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    params = update(params, grads)
```

### Scaling to Large Pods

From [Scaling deep learning workloads with PyTorch/XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm) (Google Cloud Blog, July 20, 2021):

**Multi-host training on v3-32 TPU Pod**:

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    device = xm.xla_device()

    # Load data from GCS, shard across hosts
    dataset = load_from_gcs(shard_id=index)

    # Model sharding across 32 TPU cores
    model = create_model().to(device)

    for epoch in range(num_epochs):
        for batch in dataset:
            # Forward, backward, optimize
            loss = train_step(model, batch)
            xm.mark_step()

            # Reduce metrics across all cores
            loss = xm.mesh_reduce('loss', loss, np.mean)

# Spawn on all cores
xmp.spawn(_mp_fn, nprocs=32)
```

## Performance Debugging

### XProf Profiling

```python
import torch_xla.debug.profiler as xp

# Capture trace
server = xp.start_server(9012)

# Run training
for i in range(100):
    train_step()
    if i == 10:
        xp.trace('localhost:9012', '/tmp/profile', duration_ms=10000)

# View in TensorBoard
# tensorboard --logdir=/tmp/profile
```

**Analyze**:
- Step time breakdown (compilation vs execution)
- Memory bandwidth utilization
- Compute utilization (% of peak FLOPs)
- Communication time (ICI transfers)

### Common Performance Issues

**Issue**: High compilation overhead
- **Solution**: Use static shapes, reduce unique graph shapes, cache compiled functions

**Issue**: Low compute utilization (<50%)
- **Solution**: Increase batch size, check arithmetic intensity, verify matmul dimensions are multiples of 128

**Issue**: High ICI communication time
- **Solution**: Reduce model parallelism, improve sharding strategy, increase computation per communication

**Issue**: OOM (Out of Memory) errors
- **Solution**: Reduce batch size, use gradient checkpointing, enable mixed precision, shard model across more chips

## Sources

**Official Documentation**:
- [Introduction to Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) - Google Cloud Documentation (accessed 2025-11-13)
- [Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/tpu) - Google Cloud (accessed 2025-11-13)
- [PyTorch/XLA master documentation](https://docs.pytorch.org/xla/) - PyTorch (accessed 2025-11-13)
- [JAX documentation](https://docs.jax.dev/en/latest/quickstart.html) - JAX (accessed 2025-11-13)

**Technical Deep Dives**:
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - JAX ML Scaling Book (accessed 2025-11-13)
- [Programming TPUs in JAX](https://jax-ml.github.io/scaling-book/jax-stuff/) - JAX ML Scaling Book (accessed 2025-11-13)
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html) - JAX Documentation (accessed 2025-11-13)

**Tutorials and Guides**:
- [Train a GPT2 model with JAX on TPU for free](https://developers.googleblog.com/en/train-gpt2-model-with-jax-on-tpu/) - Google Developers Blog (August 12, 2025)
- [Training on a TPU in parallel using PyTorch XLA](https://medium.com/data-science/training-on-a-tpu-parallelly-using-pytorch-xla-4afef63ee7ac) - Medium, Abhishek Swain (accessed 2025-11-13)
- [Profile PyTorch XLA workloads](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm) - Google Cloud (accessed 2025-11-13)
- [Scaling deep learning workloads with PyTorch/XLA and Cloud TPU VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm) - Google Cloud Blog (July 20, 2021)

**Video Resources**:
- [Easy TPU Development: Getting Started](https://www.youtube.com/watch?v=0fzDm2vXynw) - YouTube, Nodematic Tutorials (February 26, 2025)
- [Free TPU Access & JAX/PyTorch Setup with TPU Research Cloud](https://www.youtube.com/watch?v=PwYHoiB4Fag) - YouTube, Mashaan Alshammari (November 11, 2024)
- [PyTorch/XLA Internal](https://www.youtube.com/watch?v=kdZwgRghy3M) - YouTube, PyTorch Developer Day 2020 (November 25, 2020)

**Comparisons and Analysis**:
- [GPU and TPU Comparative Analysis Report](https://bytebridge.medium.com/gpu-and-tpu-comparative-analysis-report-a5268e4f0d2a) - Medium, ByteBridge (accessed 2025-11-13)
- [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) - CloudOptimo (April 15, 2025)
- [Inside Google's TPU: Architecture and GPU Comparisons](https://skymod.tech/inside-googles-tpu-and-gpu-comparisons/) - SkyMod (August 13, 2025)
- [Introducing Cloud TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) - Google Cloud Blog (December 6, 2023)
- [TPUv5e: The New Benchmark in Cost-Efficient Inference](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) - SemiAnalysis (September 1, 2023)

**Additional References**:
- [Tensor Processing Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) - Wikipedia (accessed 2025-11-13)
- [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Google Cloud (accessed 2025-11-13)
- [How to Use Cloud TPU for High-Performance Machine Learning on GCP](https://www.geeksforgeeks.org/devops/how-to-use-cloud-tpu-for-high-performance-machine-learning-on-gcp/) - GeeksforGeeks (July 23, 2025)

---

**Last updated**: 2025-11-13 (PART 16 execution)
