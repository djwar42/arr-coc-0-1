# TPU Multi-Host Distributed Training

## Overview

Multi-host TPU training distributes workloads across TPU Pod slices (multiple hosts/VMs, each with TPU chips), enabling training of models too large for a single host. JAX and PyTorch/XLA provide frameworks for coordinating computation, data sharding, and communication across hosts.

**Key Concept**: In multi-host TPU training, each host runs the same Python program as a separate **process** (also called a **controller**). JAX automatically coordinates these processes to form one logical cluster of TPU devices.

From [JAX Multi-Process Documentation](https://docs.jax.dev/en/latest/multi_process.html) (accessed 2025-11-16):
- Each Python process is a **controller** with its own set of **local devices** (TPU chips attached to that host)
- All devices across all processes are **global devices**
- A `jax.Array` can span all processes, enabling programming against one giant virtual machine

## TPU Pod Slice Configurations

### Available Slice Sizes

From [Google Cloud TPU v5p Documentation](https://docs.cloud.google.com/tpu/docs/v5p) (accessed 2025-11-16):

**TPU v5p Pod Slices**:
- **v5p-8**: 8 TensorCores (4 chips, single host)
- **v5p-16**: 16 TensorCores (8 chips, single host)
- **v5p-32**: 32 TensorCores (16 chips, 2 hosts)
- **v5p-64**: 64 TensorCores (32 chips, 4 hosts)
- **v5p-128**: 128 TensorCores (64 chips, 8 hosts)
- **v5p-256**: 256 TensorCores (128 chips, 16 hosts)
- **v5p-512**: 512 TensorCores (256 chips, 32 hosts)
- **v5p-1024**: 1,024 TensorCores (512 chips, 64 hosts)

**TPU v5e Pod Slices** (cost-optimized):
- v5e-1, v5e-4, v5e-8, v5e-16, v5e-32, v5e-64, v5e-128, v5e-256

**TPU v4 Pod Slices** (previous generation):
- v4-8, v4-16, v4-32, v4-64, v4-128, v4-256, v4-512, v4-1024, v4-2048, v4-4096

### Pod Topology

From [JAX Scaling Book - How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) (accessed 2025-11-16):

TPU v5p architecture:
- Each **chip** has 2 **TensorCores**
- TPU v5p: 5e14 bf16 FLOPs/sec/chip (2.5e14 per core)
- Single pod of 8,960 chips = **4.48 exaFLOPs** peak performance
- Chips connected via high-speed **ICI** (Inter-Chip Interconnect)
- Multi-host slices connected via **data center network**

## JAX Multi-Host Programming

### Initialization and Setup

From [JAX Multi-Process Documentation](https://docs.jax.dev/en/latest/multi_process.html):

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Initialize the distributed system
# On Cloud TPU, parameters are automatically detected
jax.distributed.initialize()

# Query devices
print(f"Process {jax.process_index()} of {jax.process_count()}")
print(f"Local devices: {jax.local_devices()}")  # TPUs on this host
print(f"Global devices: {jax.devices()}")       # All TPUs in pod

# Create a mesh spanning all processes
mesh = jax.make_mesh((jax.device_count(),), ('data',))
```

**Critical Rule**: `jax.distributed.initialize()` **must** be called before accessing any devices or running computations.

### Multi-Host JAX: pjit and Mesh Sharding

From [JAX Explicit Sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html):

```python
# Define a 2D mesh across all TPU devices
mesh = jax.make_mesh((8, 16), ('data', 'model'))

# Create sharding specifications
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))

# Create arrays sharded across all processes/devices
x = jax.device_put(jnp.ones((1024, 2048)), data_sharding)
weights = jax.device_put(jnp.ones((2048, 4096)), model_sharding)

# Computations automatically parallelize
@jax.jit
def train_step(x, w):
    return jnp.dot(x, w)

result = train_step(x, weights)
```

**Key Insight**: JAX automatically inserts **collective communication** (AllReduce, AllGather) when needed across hosts.

### GSPMD: General and Scalable Parallelization

From [JAX Changelog](https://docs.jax.dev/en/latest/changelog.html) (accessed 2025-11-16):

JAX uses **GSPMD** (General and Scalable Parallelization for ML Deployment) for automatic sharding:
- Compiler-level sharding propagation
- Automatic communication collective insertion
- Migrating to **Shardy** (new sharding system) as of 2025

From [Medium - 7 JAX Sharding Patterns](https://medium.com/@Nexumo_/7-jax-sharding-patterns-that-scale-without-pain-fb2ae656457a) (accessed 2025-11-16):

GSPMD enables:
- **Data parallelism**: Shard batch across hosts
- **Model parallelism**: Shard weights across hosts
- **Pipeline parallelism**: Different layers on different hosts
- **Hybrid strategies**: Combine all three

## Data Parallelism on TPU Pods

### Standard Data-Parallel Pattern

```python
# Each host loads different batch shard
batch_size = 1024
per_process_batch = batch_size // jax.process_count()

# Data-parallel mesh
mesh = jax.make_mesh((jax.device_count(),), ('batch',))
sharding = NamedSharding(mesh, P('batch'))

# Load process-local data (each host loads different examples)
local_batch = load_data(per_process_batch)  # numpy array

# Create global array from process-local data
global_batch = jax.make_array_from_process_local_data(
    sharding, local_batch
)

# Train step (automatically parallelized)
@jax.jit
def train_step(batch, model):
    loss = compute_loss(batch, model)
    grads = jax.grad(loss)(model)
    # Gradients automatically averaged across all devices
    return update_model(model, grads)
```

From [JAX Multi-Process Documentation](https://docs.jax.dev/en/latest/multi_process.html):

**Three ways to create process-spanning arrays**:
1. `jax.device_put()` - Load full array on all processes, then shard
2. `jax.make_array_from_process_local_data()` - Each process loads its shard
3. `jax.make_array_from_single_device_arrays()` - Most control, per-device data

## Model Parallelism on TPU Pods

### Tensor Parallelism with pjit

```python
# 2D mesh: data parallelism + tensor parallelism
mesh = jax.make_mesh((4, 32), ('data', 'model'))

# Shard model weights across 'model' axis
weight_sharding = NamedSharding(mesh, P(None, 'model'))
weights = jax.device_put(init_weights(), weight_sharding)

# Shard activations across 'data' axis
activation_sharding = NamedSharding(mesh, P('data', None))

@jax.jit
def forward(x, w):
    # x: sharded over 'data', w: sharded over 'model'
    # XLA automatically inserts AllReduce for matmul
    return jnp.dot(x, w)
```

### FSDP-style Sharding (Fully Sharded Data Parallel)

```python
# Shard model parameters across all devices
mesh = jax.make_mesh((jax.device_count(),), ('fsdp',))
param_sharding = NamedSharding(mesh, P('fsdp'))

# Shard parameters
sharded_params = jax.tree_map(
    lambda p: jax.device_put(p, param_sharding),
    params
)

# During forward pass, parameters are gathered
# During backward pass, gradients are reduced
```

## PyTorch/XLA Multi-Host Training

### PyTorch/XLA FSDP on TPU

From [PyTorch Blog - Scaling PyTorch Models on TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) (accessed 2025-11-16):

**FSDP on TPU v4** enables:
- 16B parameter GPT-2: 39% hardware utilization on v4-64
- 128B parameter models on v5p-1024

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

# Initialize TPU distributed environment
device = xm.xla_device()

# Wrap model with FSDP
model = MyTransformer()
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)

# Data loading with parallel loader
train_loader = pl.ParallelLoader(dataset, [device]).per_device_loader(device)

# Training loop
for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    xm.optimizer_step(optimizer)  # Syncs gradients across hosts
    xm.mark_step()  # Execute XLA graph
```

### PyTorch/XLA SPMD (Single Program Multiple Data)

From [PyTorch Blog - PyTorch/XLA SPMD](https://pytorch.org/blog/pytorch-xla-spmd/) (accessed 2025-11-16):

```python
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental.xla_sharding import XLAShardingSpec

# Define mesh for TPU pod
num_devices = xr.global_runtime_device_count()
mesh = Mesh(range(num_devices), (num_devices,), axis_names=('data',))

# Shard tensor
sharding_spec = XLAShardingSpec(mesh, ('data', None))
sharded_tensor = xs.mark_sharding(tensor, mesh, ('data', None))

# Computations automatically use GSPMD
output = model(sharded_tensor)
```

## Checkpoint Sharding for Large Models

### JAX Checkpoint Sharding

From [GitHub - JAX Multi-Host Checkpointing Discussion](https://github.com/google/jax/discussions/21290) (accessed 2025-11-16):

```python
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec as P

# Save sharded checkpoint (each host saves its shard)
checkpointer = ocp.PyTreeCheckpointer()
checkpoint_manager = ocp.CheckpointManager(
    'checkpoint_dir',
    checkpointer,
    options=ocp.CheckpointManagerOptions(
        save_interval_steps=1000,
        max_to_keep=3
    )
)

# Save (automatically distributed)
checkpoint_manager.save(
    step=global_step,
    items={'model': sharded_params, 'optimizer': opt_state}
)

# Restore (automatically distributed)
restored = checkpoint_manager.restore(
    step=global_step,
    items={'model': sharded_params, 'optimizer': opt_state}
)
```

### PyTorch/XLA Distributed Checkpointing

```python
import torch_xla.core.xla_model as xm
from torch.distributed.checkpoint import save, load

# Save checkpoint from all hosts
state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
save(state_dict, checkpoint_dir='checkpoints/step_1000')

# Restore to all hosts
load(state_dict, checkpoint_dir='checkpoints/step_1000')
```

## Inter-Host Communication Patterns

### Collective Operations on TPU Pods

From [JAX Scaling Book](https://jax-ml.github.io/scaling-book/tpus/):

**Common collectives in multi-host training**:
- **AllReduce**: Sum gradients across all devices (synchronous SGD)
- **AllGather**: Gather data from all devices to all devices
- **ReduceScatter**: Reduce then scatter results
- **AllToAll**: Permute data across devices (used in pipeline parallelism)

**Network topology matters**:
- **Intra-host** (ICI): 600 GB/s bandwidth on v5p
- **Inter-host** (data center network): Ethernet, slower than ICI
- GSPMD tries to minimize inter-host communication

### Network Optimization

```python
# Prefer intra-host communication by structuring mesh carefully
# Good: Keep frequently-communicating ops on same host
mesh = jax.make_mesh((num_hosts, devices_per_host), ('inter', 'intra'))

# Shard to minimize inter-host traffic
weight_sharding = NamedSharding(mesh, P('intra', None))  # Within-host sharding
```

## Profiling Multi-Host Training

### JAX Profiling on TPU Pods

```python
import jax.profiler as profiler

# Start profiling (collect from all hosts)
profiler.start_trace('/tmp/tensorboard')

# Run training steps
for step in range(100):
    train_step(batch)

# Stop profiling
profiler.stop_trace()

# View in TensorBoard (merge traces from all hosts)
# tensorboard --logdir=/tmp/tensorboard
```

### PyTorch/XLA Profiling

```python
import torch_xla.debug.profiler as xp

# Profile multi-host execution
server = xp.start_server(9012)

# Training loop
for batch in train_loader:
    with xp.Trace('train'):
        loss = model(batch)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()

# View traces at http://localhost:9012
```

## Fault Tolerance and Elastic Training

### Checkpoint-Resume on Preemption

```python
# Save checkpoint frequently for fault tolerance
if step % checkpoint_interval == 0:
    checkpoint_manager.save(step, {'model': params, 'step': step})

# Resume from latest checkpoint
latest_step = checkpoint_manager.latest_step()
if latest_step is not None:
    restored = checkpoint_manager.restore(latest_step)
    params = restored['model']
    start_step = restored['step']
```

### Elastic Training (Dynamic Worker Scaling)

From [Kubeflow Training Operator PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch/) (accessed 2025-11-16):

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: tpu-training
spec:
  elasticPolicy:
    minReplicas: 2
    maxReplicas: 8
  pytorchReplicaSpecs:
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/my-project/tpu-trainer
            resources:
              limits:
                cloud-tpus.google.com/v5p: 4
```

## Performance Optimization

### Reduce Cross-Host Communication

1. **Gradient Accumulation**: Reduce AllReduce frequency
```python
accumulation_steps = 4
accumulated_grads = jax.tree_map(jnp.zeros_like, params)

for micro_batch in micro_batches:
    grads = compute_grads(micro_batch)
    accumulated_grads = jax.tree_map(jnp.add, accumulated_grads, grads)

# Single AllReduce after accumulation
averaged_grads = jax.tree_map(lambda g: g / accumulation_steps, accumulated_grads)
params = update(params, averaged_grads)
```

2. **ZeRO-style Sharding**: Shard optimizer states
```python
# Shard optimizer states across devices to reduce memory
optimizer_sharding = NamedSharding(mesh, P('fsdp'))
opt_state = jax.tree_map(
    lambda x: jax.device_put(x, optimizer_sharding),
    opt_state
)
```

3. **Pipeline Parallelism**: Overlap communication and computation
```python
# Use jax.lax.scan for pipelined execution
def pipeline_step(carry, layer_input):
    activations = layer_forward(carry, layer_input)
    return activations, activations

final_activations, intermediates = jax.lax.scan(
    pipeline_step, init_state, layer_inputs
)
```

### Memory Optimization

**Rematerialization (Gradient Checkpointing)**:
```python
from jax import checkpoint as jax_checkpoint

@jax_checkpoint
def transformer_layer(x, params):
    # Forward pass without saving all activations
    return compute_layer(x, params)

# Memory usage reduced at cost of recomputation
```

## Common Pitfalls and Debugging

### Deadlocks in Multi-Host Training

From [JAX Multi-Process Documentation](https://docs.jax.dev/en/latest/multi_process.html):

**Critical Rule**: All processes must run the same computation in the same order on process-spanning arrays, or deadlocks occur.

```python
# ❌ BAD: Different processes run different code
if jax.process_index() == 0:
    result = model(batch_a)  # Only process 0
else:
    result = model(batch_b)  # Other processes
# DEADLOCK: Collective communication waits for all processes

# ✅ GOOD: All processes run same code
result = model(batch)  # All processes participate
if jax.process_index() == 0:
    print(result)  # Only print on one process
```

### Printing Process-Spanning Arrays

```python
# ❌ BAD: Can't print distributed array directly
print(global_array)  # RuntimeError: spans non-addressable devices

# ✅ GOOD: Replicate first, then print on one process
replicated = jax.device_put(global_array, NamedSharding(mesh, P(None, None)))
if jax.process_index() == 0:
    print(replicated)

# ✅ ALTERNATIVE: Print local shards
for shard in global_array.addressable_shards:
    print(f"Device {shard.device}: {shard.data}")
```

### Host Synchronization

```python
# Ensure all hosts finish before continuing
jax.effects_barrier()

# Or use multihost_utils
from jax.experimental import multihost_utils
multihost_utils.sync_global_devices("step_complete")
```

## arr-coc-0-1 TPU Multi-Host Training

### Feasibility Analysis

**Current single-GPU setup**:
- Trains on A100 40GB
- Model: ~5B parameters (estimated)
- Batch size: 32

**TPU Pod v5p-128 training**:
- 128 TensorCores (64 chips across 8 hosts)
- 64 GB HBM per chip = 4 TB total memory
- Can train models up to 100B+ parameters with FSDP

**Migration path**:
1. Port JAX code to multi-host
2. Add `jax.distributed.initialize()`
3. Update data loading for per-process sharding
4. Configure mesh for 8-host, 16-device-per-host topology
5. Test on v5p-8 (single host) first
6. Scale to v5p-128

### Example Configuration

```python
# arr-coc-0-1 multi-host training setup
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Initialize TPU pod
jax.distributed.initialize()

# v5p-128: 8 hosts × 16 devices = 128 TensorCores
assert jax.device_count() == 128
assert jax.local_device_count() == 16

# Create 2D mesh: data parallel + model parallel
mesh = jax.make_mesh((8, 16), ('data', 'model'))

# Data sharding (batch across 'data' axis)
data_sharding = NamedSharding(mesh, P('data', None))

# Model sharding (weights across 'model' axis)
model_sharding = NamedSharding(mesh, P(None, 'model'))

# Load per-process batch (each host loads 1/8 of global batch)
global_batch_size = 256
per_process_batch_size = global_batch_size // jax.process_count()
local_batch = load_batch(per_process_batch_size)

# Create global batch
global_batch = jax.make_array_from_process_local_data(
    data_sharding, local_batch
)

# Training step (automatically parallelized)
@jax.jit
def train_step(batch, params):
    loss = compute_loss(batch, params)
    grads = jax.grad(loss)(params)
    return update_params(params, grads)

# Train loop
for step in range(num_steps):
    global_batch = load_and_shard_batch()
    params = train_step(global_batch, params)

    if step % 100 == 0 and jax.process_index() == 0:
        print(f"Step {step}: loss = {compute_loss(global_batch, params)}")
```

## Sources

**Official Documentation**:
- [JAX Multi-Process Programming](https://docs.jax.dev/en/latest/multi_process.html) (accessed 2025-11-16)
- [JAX Explicit Sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html) (accessed 2025-11-16)
- [Google Cloud TPU v5p Docs](https://docs.cloud.google.com/tpu/docs/v5p) (accessed 2025-11-16)
- [JAX Scaling Book - How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) (accessed 2025-11-16)

**PyTorch/XLA Resources**:
- [PyTorch Blog - Scaling Models on TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) (accessed 2025-11-16)
- [PyTorch Blog - PyTorch/XLA SPMD](https://pytorch.org/blog/pytorch-xla-spmd/) (accessed 2025-11-16)
- [PyTorch/XLA GitHub](https://github.com/pytorch/xla)

**Community Resources**:
- [Medium - 7 JAX Sharding Patterns That Scale](https://medium.com/@Nexumo_/7-jax-sharding-patterns-that-scale-without-pain-fb2ae656457a) (accessed 2025-11-16)
- [GitHub - JAX Multi-Host Checkpointing Discussion](https://github.com/google/jax/discussions/21290) (accessed 2025-11-16)
- [AstraBlog - Exploring Parallel Strategies with JAX](https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/) (accessed 2025-11-16)

**Additional References**:
- Kubeflow Training Operator documentation
- OpenXLA GSPMD documentation
- Cloud TPU training guides
