# Cloud TPU Architecture & Programming on GCP

**Deep dive into Google Cloud TPU architecture, programming models, and production deployment patterns**

## Overview

Cloud TPUs (Tensor Processing Units) are Google's custom-designed ASICs optimized specifically for machine learning workloads, particularly transformer-based models and large-scale training. Unlike GPUs which are general-purpose accelerators, TPUs are purpose-built for the matrix multiplication operations at the heart of neural networks, offering superior performance-per-watt and cost-efficiency for ML training at scale.

**Key Cloud TPU advantages**:
- **Matrix multiplication specialization**: Systolic arrays deliver 2-5× faster matmul vs equivalent GPUs
- **High bandwidth memory**: 1-2 TB/s HBM bandwidth with 16-96GB per chip
- **Direct chip interconnects**: ICI (Inter-Chip Interconnect) enables massive pod scaling
- **Cost efficiency**: 60-70% lower cost per FLOP compared to A100 GPUs
- **Managed integration**: Vertex AI provides seamless TPU deployment

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 5-18):
> "Tensor Processing Units (TPUs) are Google's custom-designed, application-specific integrated circuits (ASICs) optimized for machine learning workloads. A TPU is essentially a matrix multiplication machine connected to fast memory."

From [Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/tpu) (Google Cloud, accessed 2025-11-16):
> "Cloud TPU v5e provides up to 1.9x higher LLM fine-tuning performance per dollar compared to Cloud TPU v4."

## TPU Generations Comparison

### TPU v4 (Training Focused)

**Architecture**:
- **HBM**: 32GB per chip
- **Performance**: 2.75e14 FLOPs/s (bf16), 2.75e14 FLOPs/s (int8)
- **HBM Bandwidth**: 1.2e12 bytes/s per chip
- **Topology**: 3D torus (6 nearest neighbors)
- **Pod Size**: 16×16×16 = 4,096 chips maximum
- **Systolic Array**: 128×128 MXU

**Use cases**:
- Large-scale LLM pre-training
- Multi-node distributed training (requires 3D topology)
- Workloads requiring high HBM capacity

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 83-95):
> "TPU v4p: 16x16x16 = 4,096 chips maximum pod size, 32GB HBM, 1.2e12 HBM bandwidth/chip, 2.75e14 FLOPs/s (bf16)"

### TPU v5e (Cost-Optimized Training & Inference)

**Architecture**:
- **HBM**: 16GB per chip (single stack, lower cost)
- **Performance**: 1.97e14 FLOPs/s (bf16), 3.94e14 FLOPs/s (int8)
- **HBM Bandwidth**: 8.1e11 bytes/s per chip
- **Topology**: 2D torus (4 nearest neighbors)
- **Pod Size**: 16×16 = 256 chips (2D only)
- **Systolic Array**: 128×128 MXU (single TensorCore)

**Cost efficiency**:
- ~60-70% lower cost per FLOP vs TPU v4
- 1.9× better LLM fine-tuning performance per dollar vs v4
- Optimal for inference and cost-sensitive training

**Use cases**:
- Cost-effective LLM fine-tuning (7B-13B models)
- High-throughput inference serving
- Development and prototyping (lower hourly costs)
- Single-node or small pod training

From [TPUv5e: The New Benchmark in Cost-Efficient Inference](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) (SemiAnalysis, September 1, 2023):
> "The TPUv5e only has a single Tensor Core, unlike TPU v5 which includes two. Furthermore it has half the HBM stacks and at lower speeds."

From [Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/tpu) (Google Cloud, accessed 2025-11-16):
> "Cloud TPU v5e provides up to 1.9x higher LLM fine-tuning performance per dollar compared to Cloud TPU v4."

### TPU v5p (High-Performance Training)

**Architecture**:
- **HBM**: 96GB per chip (3× more than v4)
- **Performance**: 4.59e14 FLOPs/s (bf16), 9.18e14 FLOPs/s (int8)
- **HBM Bandwidth**: 2.8e12 bytes/s per chip
- **Topology**: 3D torus (6 nearest neighbors)
- **Pod Size**: 16×20×28 = 8,960 chips maximum
- **Systolic Array**: 128×128 MXU (dual TensorCores)

**Performance improvements**:
- 2× greater FLOPs vs v4
- 3× more HBM capacity (enables larger models)
- 2.8× faster at training LLMs vs v4

**Use cases**:
- Frontier model training (70B+ parameter models)
- Large-scale distributed training (thousands of chips)
- Research requiring massive memory capacity

From [Introducing Cloud TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) (Google Cloud Blog, December 6, 2023):
> "Compared to TPU v4, TPU v5p features more than 2X greater FLOPS and 3X more high-bandwidth memory (HBM)."

From [Google's latest AI chip is up to 2.8 times faster at training LLMs](https://www.reddit.com/r/singularity/comments/1ac0ax9/googles_latest_ai_chip_is_up_to_28_times_faster/) (Reddit, January 2024):
> "Google's v5p TPUs are up to 2.8 times faster at training large language models than TPU v4, and offer 2.1-times value-for-money."

### TPU v6e Trillium (Latest Generation)

**Architecture**:
- **HBM**: 32GB per chip
- **Performance**: 9.20e14 FLOPs/s (bf16), 1.84e15 FLOPs/s (int8)
- **HBM Bandwidth**: 1.6e12 bytes/s per chip
- **Topology**: 2D torus (4 nearest neighbors)
- **Pod Size**: 16×16 = 256 chips
- **Systolic Array**: 256×256 MXU (2× larger than v5e)

**Major improvements**:
- 4.7× performance increase over v5e (bf16)
- 256×256 systolic array (vs 128×128 in v5e)
- Second generation SparseCores for embeddings

**Availability**: Preview on Cloud TPU and Vertex AI

From [Trillium sixth-generation TPU is in preview](https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview) (Google Cloud Blog, October 30, 2024):
> "We're pleased to announce that Trillium, our sixth-generation TPU, is now available to Google Cloud customers in preview."

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 83-95):
> "TPU v6e: 16x16 = 256 chips (2D only), 32GB HBM, 1.6e12 HBM bandwidth, 9.20e14 FLOPs/s (bf16), 1.84e15 FLOPs/s (int8)"

### TPU Generations Comparison Table

| Model | Pod Size | HBM/chip | HBM BW/chip | FLOPs/s (bf16) | FLOPs/s (int8) | Topology | Use Case |
|-------|----------|----------|-------------|----------------|----------------|----------|----------|
| v4 | 16×16×16 | 32GB | 1.2 TB/s | 275 TF/s | 275 TF/s | 3D torus | Large-scale training |
| v5e | 16×16 | 16GB | 0.81 TB/s | 197 TF/s | 394 TF/s | 2D torus | Cost-optimized training/inference |
| v5p | 16×20×28 | 96GB | 2.8 TB/s | 459 TF/s | 918 TF/s | 3D torus | Frontier models |
| v6e | 16×16 | 32GB | 1.6 TB/s | 920 TF/s | 1,840 TF/s | 2D torus | Next-gen inference |

## TPU Architecture Deep Dive

### Core Components

**TensorCore** - The main compute unit:

1. **MXU (Matrix Multiply Unit)**:
   - Systolic array: 128×128 for v4/v5, 256×256 for v6e
   - Performs: `bf16[8,128] @ bf16[128,128] -> f32[8,128]` every 8 cycles
   - TPU v5e: ~5e13 bf16 FLOPs/s per MXU at 1.5GHz
   - Most TensorCores have 2-4 MXUs

2. **VPU (Vector Processing Unit)**:
   - 2D SIMD vector machine: shape (8, 128)
   - Performs: ReLU, pointwise add/multiply, reductions
   - TPU v5p: ~1.4e13 FLOPs/s (much smaller than MXU at ~2e14)

3. **VMEM (Vector Memory)**:
   - On-chip scratchpad: 128 MiB on TPU v5e
   - 22× higher bandwidth than HBM
   - Programmer-controlled (not automatic like CPU cache)
   - Data must be copied from HBM → VMEM before computation

**HBM (High Bandwidth Memory)**:
- Main memory: 16GB (v5e), 32GB (v4/v6e), 96GB (v5p)
- Bandwidth: 1-2 TB/s HBM ↔ TensorCore
- Stores model parameters, activations, gradients

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 19-51):
> "TensorCore components: MXU (Matrix Multiply Unit) - performs bf16[8,128] @ bf16[128,128] -> f32[8,128] every 8 cycles; VPU (Vector Processing Unit) - performs ReLU, pointwise operations; VMEM (Vector Memory) - on-chip scratchpad, 22× higher bandwidth than HBM"

### Memory Hierarchy and Performance

Understanding the memory hierarchy is critical for TPU optimization:

```
Fastest    → VMEM ↔ MXU        (22× faster than HBM bandwidth)
           → HBM ↔ VMEM        (1-2 TB/s)
           → ICI (Inter-Chip)  (90-180 GB/s)
           → PCIe (Host)       (~16 GB/s)
Slowest    → DCN (Data Center) (~6.25 GB/s)
```

**Arithmetic intensity requirement**:
- **TPU v5e**: Need ~240 FLOPs/byte to be FLOPs bound (from HBM)
- **With VMEM**: Need only ~10-20 FLOPs/byte to be FLOPs bound
- **Implication**: If weights fit in VMEM, much smaller batch sizes achieve peak utilization

**Batch size tuning example** (from TPU fundamentals):
- Matrix multiply `int8[16384, 4096] @ int8[B, 4096]` on TPU v5e:
  - FLOPs bound when B > 271 (reading from HBM)
  - FLOPs bound when B > 11 (reading from VMEM)

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 289-318):
> "For B << D and F = 4D (common in transformers): Intensity ≈ B (for large D). Need B > 240 to be FLOPs bound on v5e. With VMEM optimization: can be FLOPs bound at B > 11."

### Systolic Array Architecture

The systolic array is the heart of TPU's matrix multiplication performance.

**How it works**:
1. **Grid of ALUs**: 128×128 grid (16,384 ALUs) for v4/v5, 256×256 for v6e
2. **Data flow**: Weights flow down columns, activations flow in from left
3. **Diagonal loading**: Inputs loaded diagonally to maximize overlap
4. **Pipelined execution**: After initial bubble, continuous throughput

**Performance characteristics**:
- One `bf16[8,128] @ bf16[128,128] -> f32[8,128]` per 8 cycles
- **Key requirement**: Matrix dimensions should be multiples of 128 (256 for v6e)
- Smaller dimensions automatically padded (wastes compute)

**Pipeline bubbles**:
- Initial bubble occurs while loading first weights/activations diagonally
- After bubble, new inputs can be loaded without additional bubbles
- **Optimization**: Use large matrices (>>128) to amortize bubble overhead

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 258-286):
> "Systolic array: 128×128 grid (16,384 ALUs) for v4/v5, 256×256 for v6e. Data flow: Weights flow down, activations flow in from left. Performance: One bf16[8,128] @ bf16[128,128] -> f32[8,128] per 8 cycles."

## TPU Networking and Pod Architecture

### Inter-Chip Interconnect (ICI)

TPUs connect to each other through direct chip-to-chip links forming mesh topologies.

**Connectivity patterns**:
- **2D torus** (v3/v5e/v6e): 4 nearest neighbors
- **3D torus** (v4/v5p): 6 nearest neighbors

**ICI bandwidth** (bidirectional):
| TPU | Bandwidth per link |
|-----|-------------------|
| v3 | 200 GB/s |
| v4 | 90 GB/s |
| v5p | 180 GB/s |
| v5e | 90 GB/s |
| v6e | 180 GB/s |

**Topology implications**:
- Communication between distant chips requires multiple hops
- Wraparound links on axes of size 16+ reduce max distance
- Full cubes (4×4×4 for v4/v5p) get optical wraparound links

From [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) (Google Cloud Documentation, accessed 2025-11-16):
> "Slice. A slice is a collection of chips all located inside the same TPU Pod connected by high-speed inter chip interconnects (ICI)."

From [Inside the Ironwood TPU codesigned AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack) (Google Cloud Blog, November 6, 2025):
> "Inter-Chip Interconnect (ICI) links that form a direct 3D Torus topology. This creates an extremely dense, all-to-all network fabric."

### Pod Sizes and Slices

**Pod** = Set of ICI-connected TPUs in same data center

**Maximum pod sizes**:
- TPU v4: 16×16×16 = 4,096 chips
- TPU v5p: 16×20×28 = 8,960 chips
- TPU v5e/v6e: 16×16 = 256 chips (2D only)

**Slice** = Subset of pod allocated to a job

**Common slice sizes**:
| Slice Type | v5e (2D) | v4/v5p (3D) |
|-----------|---------|-------------|
| Single host | 4×2 | 2×2×1 |
| Small | 8-16 chips | 2×2×2, 4×4×4 |
| Medium | 32-64 chips | 8×8×8 |
| Large | 128-256 chips | 16×16×16 |

**Slice topology examples**:
- **v5e 4×4 slice**: 2D torus, no wraparound links (small slice)
- **v5e 16×16 slice**: 2D torus, wraparound on both axes (full pod)
- **v5p 4×4×4 slice**: 3D torus cube with optical wraparound
- **v5p 16×20×28 slice**: Full superpod with complex 3D topology

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 227-255):
> "Maximum pod sizes: TPU v4: 16×16×16 = 4,096 chips, TPU v5p: 16×20×28 = 8,960 chips, TPU v5e/v6e: 16×16 = 256 chips (2D only)"

From [TPU v4](https://docs.cloud.google.com/tpu/docs/v4) (Google Cloud Documentation, accessed 2025-11-16):
> "Some v4 3D torus slice shapes have the option to use what is known as a twisted torus topology. For example, two v4 cubes can be arranged as a 4×4×8 slice."

### Data Center Network (DCN)

**DCN** connects different pods and hosts through standard Ethernet networking.

**Bandwidth hierarchy**:
- HBM: 2.5 TB/s (fastest, on-chip)
- ICI: 90-180 GB/s (chip-to-chip within pod)
- PCIe: 16 GB/s (host-to-chip)
- DCN: 6.25 GB/s (slowest, between pods)

**Multi-slice training**:
- Connect slices via DCN for training beyond single pod size
- DCN is 100× slower than ICI → minimize cross-DCN communication
- Use data parallelism across DCN boundaries, model parallelism within ICI

**ICI latency**: ~1μs per hop

**Data transfer time** = (bytes / bandwidth) + (hops × latency)

Example from TPU fundamentals:
- Transfer 1.7e7 bytes from TPU{0,0} to TPU{3,3} in 4×4 v5e slice
- Bandwidth: 9e10 bytes/s (both axes)
- Hops: 6 (no wraparound in 4×4)
- Total time: 6μs + 188μs = 194μs

## JAX Programming on TPUs

### Why JAX for TPUs

JAX is Google's native framework for TPU programming, providing optimal performance through direct XLA compilation.

**JAX advantages**:
- **Native XLA**: Direct compilation to TPU machine code (no intermediate layers)
- **Automatic batching**: `vmap` for easy data parallelism
- **Automatic differentiation**: `grad` for training loops
- **Parallelization**: `pmap` and `pjit` for multi-chip distribution
- **NumPy compatibility**: Familiar API for ML researchers

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 100-132):
> "JAX is Google's primary framework for TPU programming, providing NumPy-like interface with automatic differentiation and XLA compilation. JAX advantages for TPUs: Native XLA compilation, automatic batching with vmap, automatic differentiation with grad, easy parallelization with pmap and pjit."

### JAX Quickstart

**Basic JAX TPU usage**:

```python
import jax
import jax.numpy as jnp

# Automatically uses TPU if available
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x.T)

# Check TPU devices
print(jax.devices())
# Output: [TpuDevice(id=0), TpuDevice(id=1), ...]

# Get device count
print(jax.device_count())
# Output: 8 (for single v5e host)
```

### Data Parallelism with pmap

**Replicate model across all TPU cores**:

```python
from jax import pmap
import jax.numpy as jnp

# Get number of TPU devices
n_devices = jax.local_device_count()
print(f"Training on {n_devices} TPU cores")

@pmap  # Replicate across all devices
def parallel_train_step(state, batch):
    """Training step replicated across TPU cores."""
    def loss_fn(params):
        logits = state.apply_fn(params, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], batch['labels'][:, 1:]
        ).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Gradients automatically averaged across devices via ICI
    state = state.apply_gradients(grads=grads)
    return state, loss

# Reshape batch for pmap: [n_devices, per_device_batch, seq_len]
batch_per_device = global_batch_size // n_devices
batched_data = batch.reshape(n_devices, batch_per_device, seq_len)

# Run parallel training step
state, loss = parallel_train_step(state, batched_data)
```

**pmap characteristics**:
- Batch dimension automatically sharded across devices
- Model parameters replicated on each device
- Gradients reduced via all-reduce over ICI (SPMD pattern)
- Suitable for models that fit on single TPU chip (≤16GB for v5e)

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 322-361):
> "pmap sharding strategy: Batch dimension automatically sharded across devices, model parameters replicated on each device, gradients reduced via all-reduce over ICI"

### Model Parallelism with pjit

**SPMD model + data parallelism for large models**:

```python
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit, PartitionSpec as P
from jax.sharding import Mesh

# Define 2D device mesh: data parallel × model parallel
devices = mesh_utils.create_device_mesh((4, 2))  # 4 data, 2 model
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding for model parameters
with mesh:
    @pjit(
        in_shardings=(P('data', None), P('data', None)),  # Inputs: data parallel
        out_shardings=P('data', None),  # Output: data parallel
    )
    def sharded_train_step(state, batch):
        def loss_fn(params):
            # FFN weights sharded: [d_model, 4*d_model] -> shard second dim
            # Enables tensor parallelism for large FFN layers
            logits = state.apply_fn(params, batch['input_ids'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1], batch['labels'][:, 1:]
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
```

**When to use pjit**:
- Model parameters > 16GB (single v5e chip capacity)
- Training 7B+ parameter models on v5e pods
- Training 13B-70B models on v5p pods
- Need both data and model parallelism

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 386-418):
> "Model parallelism with pjit: Define device mesh, shard large matrix across devices. Use when: model parameters > 16GB (single chip), training 7B+ models on pods"

### JIT Compilation Best Practices

**Static shapes are critical for TPU performance**:

```python
# ✗ BAD: Dynamic shape triggers recompilation
@jax.jit
def bad_function(x):
    return x[:x.sum()]  # Output shape depends on data

# ✓ GOOD: Fixed shape with masking
@jax.jit
def good_function(x, length):
    mask = jnp.arange(len(x)) < length
    return jnp.where(mask, x, 0)

# Use static_argnums for shape-determining arguments
@jax.jit(static_argnums=(1,))
def process(data, length):
    return data[:length].sum()
```

**Compilation overhead**:
- First run includes compilation (can be seconds)
- Subsequent runs use cached compiled code
- Shape changes trigger recompilation
- **Best practice**: Use fixed shapes whenever possible

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 406-418):
> "Efficient compilation: Use static_argnums for shape-determining arguments, avoid dynamic shapes, use fixed-size padding when needed"

## PyTorch XLA Programming on TPUs

### PyTorch XLA Architecture

PyTorch XLA enables PyTorch models on TPUs through XLA compilation.

**Architecture**: PyTorch → XLA IR → TPU machine code

**Key concepts**:
1. **Lazy execution**: Operations traced to graph, then compiled
2. **mark_step()**: Triggers compilation and execution
3. **XLA compiler**: Converts PyTorch ops to XLA ops, then to TPU code

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 161-175):
> "Key PyTorch XLA concepts: 1. Lazy execution: Operations traced to graph, then compiled. 2. mark_step(): Triggers compilation and execution. 3. XLA compiler: Converts PyTorch ops to XLA ops, then to TPU code."

### Basic PyTorch XLA Training

**CRITICAL: mark_step() is mandatory**:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Get TPU device
device = xm.xla_device()
print(f"Using device: {device}")

# Initialize model on TPU
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'].to(device),
        labels=batch['labels'].to(device)
    )
    loss = outputs.loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # CRITICAL: Use XLA-aware optimizer step
    xm.optimizer_step(optimizer)

    # CRITICAL: Mark step for XLA compilation
    xm.mark_step()

    # Logging (only occasionally to avoid CPU transfers)
    if step % 100 == 0:
        loss_value = loss.item()  # Transfers to CPU
        print(f"Step {step}, Loss: {loss_value:.4f}")
```

**Critical PyTorch XLA requirements**:
1. **`xm.mark_step()`**: MUST call after optimizer step to trigger XLA execution
2. **`xm.optimizer_step()`**: Use XLA-aware step for gradient synchronization
3. **Minimize device transfers**: Keep tensors on device, minimize `.item()` calls
4. **Static shapes**: Avoid dynamic shapes (variable sequence lengths)

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 439-507):
> "Critical PyTorch XLA requirements: 1. mark_step(): MUST call after optimizer step. 2. xm.optimizer_step(): Use XLA-aware optimizer. 3. Device transfer minimization. 4. Static shapes."

### Multi-Core PyTorch XLA Training

**Training across all TPU cores in a pod**:

```python
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    """Function executed on each TPU core."""
    device = xm.xla_device()
    print(f"Process {index} using device {device}")

    # Initialize model (replicated on each core)
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Load data with proper sharding
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,  # Per-core batch size
        sampler=train_sampler,
        num_workers=4
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)
            xm.mark_step()

            # Reduce metrics across all cores
            if step % 100 == 0:
                loss_reduced = xm.mesh_reduce('loss', loss, lambda x: sum(x) / len(x))
                if xm.is_master_ordinal():
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss_reduced:.4f}")

        # Save checkpoint (only on master)
        if xm.is_master_ordinal():
            xm.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

# Spawn processes on all TPU cores
if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(), nprocs=None)  # nprocs=None uses all cores
```

**Multi-core considerations**:
- Each core runs independent process
- Model replicated on all cores (data parallel)
- Gradients automatically synchronized via `xm.optimizer_step()`
- Use `xm.mesh_reduce()` for cross-core metric aggregation
- Only master ordinal (rank 0) saves checkpoints

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 509-576):
> "Multi-core considerations: Each core runs independent process, model replicated on all cores (data parallel), gradients synchronized via xm.optimizer_step()"

## TPU Performance Optimization

### Batch Size Optimization

**Finding optimal batch size for FLOPs-bound training**:

```python
def calculate_optimal_batch_size(
    model_params: int,
    seq_len: int,
    d_model: int,
    tpu_type: str = 'v5e'
):
    """Calculate minimum batch size for FLOPs-bound training."""

    # Memory bandwidth (bytes/s)
    hbm_bw = {
        'v5e': 8.1e11,
        'v5p': 2.8e12,
        'v6e': 1.6e12
    }[tpu_type]

    # FLOPs capacity (FLOPs/s, bf16)
    flops_capacity = {
        'v5e': 1.97e14,
        'v5p': 4.59e14,
        'v6e': 9.20e14
    }[tpu_type]

    # Required arithmetic intensity
    required_intensity = flops_capacity / hbm_bw

    # For transformer FFN: intensity ≈ batch_size (when seq_len, d_model >> batch)
    min_batch_size = int(required_intensity) + 1

    print(f"Minimum batch size for {tpu_type}: {min_batch_size}")
    print(f"Recommended batch size: {min_batch_size * 2}")

    return min_batch_size * 2

# Example usage
optimal_batch = calculate_optimal_batch_size(
    model_params=124_000_000,  # 124M params
    seq_len=1024,
    d_model=768,
    tpu_type='v5e'
)
# Output: Minimum batch size for v5e: 244
#         Recommended batch size: 488
```

**Batch size guidelines**:
- **v5e**: Minimum batch size ~240-250 for FLOPs bound
- **v5p**: Minimum batch size ~160-170 for FLOPs bound
- **v6e**: Minimum batch size ~180-190 for FLOPs bound
- **With VMEM**: Can be FLOPs bound at much smaller batches (~11-20)

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 334-346):
> "Example: Matrix multiply int8[16384, 4096] @ int8[B, 4096] on TPU v5e. FLOPs bound when B > 271 (from HBM), FLOPs bound when B > 11 (from VMEM)"

### Matrix Dimension Optimization

**Pad dimensions to multiples of 128 (256 for v6e)**:

```python
# ✗ BAD: Irregular dimensions waste systolic array
hidden_dim = 1000  # Wastes ~22% of MXU capacity
ffn_dim = 4000     # Wastes capacity

# ✓ GOOD: Multiples of 128 fully utilize MXU
hidden_dim = 1024  # 128 × 8, perfect for v5e
ffn_dim = 4096     # 128 × 32, perfect for v5e

# ✓ BEST: For v6e Trillium, use multiples of 256
hidden_dim = 2048  # 256 × 8
ffn_dim = 8192     # 256 × 32
```

**Why this matters**:
- Systolic array is 128×128 (v5e) or 256×256 (v6e)
- Non-aligned dimensions are automatically padded
- Padding wastes compute and memory bandwidth
- Can lose 10-30% performance with poor alignment

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 595-608):
> "Pad to multiples of 128: hidden_dim = 1024 # 128 × 8, fully utilizes MXU. For v6e Trillium, use multiples of 256: hidden_dim = 2048 # 256 × 8"

### Precision Optimization

**Use bf16 for training, int8 for inference**:

| TPU | bf16 FLOPs/s | int8 FLOPs/s | Speedup |
|-----|--------------|--------------|---------|
| v5e | 197 TF/s | 394 TF/s | 2× |
| v5p | 459 TF/s | 918 TF/s | 2× |
| v6e | 920 TF/s | 1,840 TF/s | 2× |

**Best practices**:
- **Training**: Use bf16 (minimal accuracy loss, 2× throughput vs fp32)
- **Inference**: Use int8 quantization (2× throughput vs bf16, acceptable quality)
- **Gradient accumulation**: Accumulate in fp32 for numerical stability

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 320-332):
> "Lower precision = higher throughput. TPU v5e: bf16: 1.97e14 FLOPs/s, int8: 3.94e14 FLOPs/s (2× faster). Trade-off: Lower precision may reduce model quality."

### Common Performance Issues

**Issue 1: Low compute utilization (<50%)**

```python
# Symptom: TPU underutilized, training slower than expected

# Solutions:
# 1. Increase batch size (most common fix)
batch_size = 512  # Increase until memory limit

# 2. Check matrix dimensions are multiples of 128
assert d_model % 128 == 0
assert ffn_dim % 128 == 0

# 3. Verify arithmetic intensity
# For v5e, need batch_size > 240 for FLOPs bound
```

**Issue 2: High compilation overhead**

```python
# Symptom: First steps very slow, frequent recompilations

# Solutions:
# 1. Use static shapes (avoid dynamic shapes)
# Bad:
x = x[:actual_length]  # Dynamic slice

# Good:
mask = jnp.arange(max_length) < actual_length
x = x * mask[:, None]  # Static shape with masking

# 2. Use static_argnums in jit
@jax.jit(static_argnums=(1,))
def process(data, length):
    return data[:length].sum()
```

**Issue 3: Out of Memory (OOM)**

```python
# Solutions:
# 1. Reduce batch size
batch_size = 256  # Instead of 512

# 2. Enable gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x)
    return x

# 3. Use mixed precision (bf16 instead of fp32)
# JAX: automatically uses bf16 on TPU
# PyTorch: use torch.autocast
with torch.autocast(device_type='xla', dtype=torch.bfloat16):
    outputs = model(inputs)
```

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 951-1035):
> "Common TPU training issues: Low compute utilization (<50%) - increase batch size, check matrix dimensions. High compilation overhead - use static shapes. OOM - reduce batch size, gradient checkpointing, mixed precision."

## TPU vs GPU Decision Framework

### Performance Comparison

**When TPUs excel**:
- **Matrix multiplication heavy**: Transformer models, ViT, BERT
- **Large batch sizes**: Can utilize batch sizes > 240 effectively
- **Long sequences**: Benefit from high HBM bandwidth
- **Cost-sensitive**: 60-70% lower cost per FLOP

**When GPUs excel**:
- **Small batch sizes**: Efficient at batch sizes < 64
- **Diverse operations**: Mix of matmul, convolutions, custom ops
- **Debugging needs**: Better profiling tools (Nsight, CUDA-GDB)
- **Ecosystem libraries**: More CUDA-optimized libraries

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 480-511):
> "TPU advantages: Matrix multiplication 2-5× faster, higher memory bandwidth, power efficiency 1.2-1.7× better, direct chip interconnects. GPU advantages: Flexibility for diverse workloads, mature ecosystem, better debugging tools."

### Cost Analysis

**Training cost comparison** (approximate, us-central2):

| Resource | Cost/hour | Memory/chip | Best for |
|----------|-----------|-------------|----------|
| v5e (1 chip) | $1.20 | 16GB | Dev, prototyping |
| v5e (4 chips) | $4.80 | 64GB | Small-scale prod |
| v5e (32 chips) | $38.40 | 512GB | Medium LLM training |
| v5p (8 chips) | $12.00 | 768GB | Frontier models |
| A100 40GB | $3.67 | 40GB | GPU baseline |
| A100 80GB | $5.89 | 80GB | Large model GPU |
| H100 80GB | $9.25 | 80GB | Latest GPU |

**Cost efficiency**:
- v5e: 60-70% lower cost per FLOP vs A100
- v5p: 40-50% lower cost per FLOP vs H100
- Preemptible TPUs: Additional 60-70% savings

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 733-758):
> "Cost efficiency analysis: v5e ~60-70% lower cost per FLOP vs A100, v5p ~40-50% lower vs H100. Best for transformer training, large batch sizes."

### Decision Tree

**Choose TPU when**:
1. Training transformers or vision transformers
2. Can use batch sizes > 240 (v5e) or > 160 (v5p)
3. Workload is matmul-dominated (>80% compute time)
4. Cost optimization is critical
5. Need to scale to 100+ accelerators

**Choose GPU when**:
1. Small batch sizes required (< 64)
2. Mixed workload (CNNs + transformers + custom ops)
3. Heavy debugging and profiling needed
4. Ecosystem libraries are CUDA-only
5. Need flexible deployment options

**Hybrid approach**:
- Use TPUs for training large transformers
- Use GPUs for inference and mixed workloads
- Use GPUs for development, TPUs for production training

## arr-coc-0-1 TPU Feasibility Analysis

### Current Implementation

The arr-coc-0-1 project currently uses **PyTorch on A100 GPUs**:

**Current architecture**:
- Vision encoder: Qwen3-VL (pretrained)
- Relevance scorer: Custom texture + knowing modules
- Training: A100 single-GPU (dev), 8×A100 (production)

**TPU feasibility assessment**:

✅ **Good fit**:
- Transformer-based architecture (ViT encoder)
- Large batch size training (K=200 patches × batch_size)
- Matrix multiplication heavy (attention, FFN layers)

⚠️ **Challenges**:
- Custom CUDA kernels in Qwen3-VL (may not port directly)
- Dynamic texture array sizes (would need padding)
- Mixed CPU/GPU operations in relevance scoring

### Migration Recommendation

**Don't migrate immediately** - Current PyTorch/A100 workflow is mature

**Consider for future scale** - If training >100 jobs/month, TPU cost savings become significant

**If migrating, recommended approach**:

**Phase 1**: JAX port of core modules (~2-3 weeks)
- Port knowing.py scorers to JAX/Flax
- Port balancing.py to JAX
- Port attending.py to JAX

**Phase 2**: Test on TPU v5e-4 (~1 week)
- Deploy to Cloud TPU with v5e-4
- Benchmark vs A100 single-GPU
- Validate numerical accuracy

**Phase 3**: Scale to v5e pod (~1 week)
- Data parallel training across 32 chips
- Cost comparison: v5e-32 vs 8×A100

**Estimated effort**: 4-6 weeks for full migration

### Cost-Benefit Analysis

**Training cost comparison** (arr-coc-0-1 typical run):

| Configuration | Hardware | Cost/hour | Hours | Total cost |
|--------------|----------|-----------|-------|------------|
| Current (dev) | A100 40GB × 1 | $3.67 | 10 | $36.70 |
| Current (prod) | A100 80GB × 8 | $47.12 | 4 | $188.48 |
| TPU option 1 | v5e-8 | $4.80 | 8 | $38.40 |
| TPU option 2 | v5e-32 | $38.40 | 2 | $76.80 |
| Preemptible v5e-32 | v5e-32 (spot) | $19.20 | 2.4 | $46.08 |

**Analysis**:
- **v5e-8**: Similar cost to single A100, 80% speedup potential = lower total cost
- **v5e-32**: 50% more expensive than 8×A100, but 2× speedup = comparable efficiency
- **Preemptible v5e-32**: 60% savings, best option if checkpointing is robust

**Recommendation**:
- **Prototype on v5e-4** first to validate JAX port before committing
- **Use preemptible v5e-32** if doing >20 training runs per month

From [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) (lines 1064-1140):
> "arr-coc-0-1 TPU feasibility: Good fit (transformer-based, large batches, matmul-heavy). Challenges (custom CUDA, dynamic shapes). Recommendation: Don't migrate immediately, consider for future scale, prototype on v5e-4 first."

## Sources

**Official Documentation**:
- [Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/tpu) - Google Cloud (accessed 2025-11-16)
- [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Google Cloud Documentation (accessed 2025-11-16)
- [TPU v4](https://docs.cloud.google.com/tpu/docs/v4) - Google Cloud Documentation (accessed 2025-11-16)
- [JAX documentation](https://docs.jax.dev/en/latest/quickstart.html) - JAX (accessed 2025-11-16)
- [PyTorch/XLA master documentation](https://docs.pytorch.org/xla/) - PyTorch (accessed 2025-11-16)

**Technical Deep Dives**:
- [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - Karpathy Deep Oracle (835 lines, comprehensive TPU architecture and programming)
- [vertex-ai-production/03-tpu-training-optimization.md](../vertex-ai-production/03-tpu-training-optimization.md) - Vertex AI TPU training patterns (1,174 lines)
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - JAX ML Scaling Book (accessed 2025-11-16)

**Blog Posts and Announcements**:
- [Introducing Cloud TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) - Google Cloud Blog (December 6, 2023)
- [Trillium sixth-generation TPU is in preview](https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview) - Google Cloud Blog (October 30, 2024)
- [Inside the Ironwood TPU codesigned AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack) - Google Cloud Blog (November 6, 2025)
- [TPUv5e: The New Benchmark in Cost-Efficient Inference](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) - SemiAnalysis (September 1, 2023)

**Community Resources**:
- [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) - CloudOptimo (April 15, 2025)
- [GPU and TPU Comparative Analysis Report](https://bytebridge.medium.com/gpu-and-tpu-comparative-analysis-report-a5268e4f0d2a) - Medium, ByteBridge (accessed 2025-11-16)
- [TPU Deep Dive](https://henryhmko.github.io/posts/tpu/tpu.html) - Henry Ko GitHub Pages (June 18, 2025)

**Research Papers**:
- [Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer](https://www.usenix.org/system/files/nsdi24-zu.pdf) - USENIX NSDI 2024 (Y. Zu et al., cited by 32)

**Search Results** (accessed 2025-11-16):
- Google Search: "Cloud TPU v5e v5p v4 comparison specifications 2024 2025"
- Google Search: "JAX TPU programming guide jit pmap pjit 2024"
- Google Search: "PyTorch XLA TPU training mark_step multi-core 2024 2025"
- Google Search: "TPU Pod slice topology 2D 3D torus ICI interconnect 2024"

---

**Last updated**: 2025-11-16 (PART 13 execution - GCP GPU & Cloud AI expansion)
