# Vertex AI TPU Training & Optimization

**Production TPU training on Google Cloud's managed ML platform**

## Overview

Vertex AI provides managed access to Google's Tensor Processing Units (TPUs) through Custom Training Jobs, enabling large-scale machine learning training with the performance benefits of custom AI accelerators. This guide covers deploying JAX and PyTorch/XLA workloads on Vertex AI TPUs, from v5e cost-optimized training to v5p high-performance pods.

**Key advantages of Vertex AI TPUs**:
- **Managed infrastructure**: No VM provisioning, automated scaling
- **Cost optimization**: Per-second billing, preemptible options, v5e efficiency
- **Integration**: Native W&B, TensorBoard, Model Registry support
- **Multi-framework**: JAX, PyTorch/XLA, TensorFlow support

From [Training with TPU accelerators | Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm) (Google Cloud Documentation, accessed 2025-11-14):
> "Vertex AI supports training with various frameworks and libraries using a TPU VM. When configuring compute resources, you can specify TPU v2, TPU v3, or TPU v5e VMs."

## TPU Generations on Vertex AI

### Available TPU Types

**TPU v5e** (Cost-Optimized Training & Inference):
- **Availability**: Generally available on Vertex AI
- **HBM**: 16GB per chip
- **Performance**: 1.97e14 FLOPs/s (bf16), 3.94e14 FLOPs/s (int8)
- **Topology**: 2D torus (4 nearest neighbors)
- **Use case**: Cost-effective training, high-throughput inference
- **Pricing**: ~60-70% lower cost per FLOP vs v4

From [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) (CloudOptimo, April 15, 2025):
> "TPU v5e / v5p: Cost-effective training at scale"

**TPU v5p** (High-Performance Training):
- **Availability**: Via Cloud TPU (can integrate with Vertex AI workloads)
- **HBM**: 96GB per chip (3× more than v4)
- **Performance**: 4.59e14 FLOPs/s (bf16), 9.18e14 FLOPs/s (int8)
- **Topology**: 3D torus (6 nearest neighbors)
- **Pod size**: Up to 8,960 chips
- **Use case**: Large-scale LLM training, frontier models

From [Introducing Cloud TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) (Google Cloud Blog, December 6, 2023):
> "Compared to TPU v4, TPU v5p features more than 2X greater FLOPS and 3X more high-bandwidth memory (HBM)."

**TPU v6e Trillium** (Latest Generation):
- **Availability**: Preview on Vertex AI
- **Systolic array**: 256×256 (vs 128×128 in v5e)
- **Performance**: 9.20e14 FLOPs/s (bf16), 4.7× faster than v5e
- **HBM**: 32GB per chip
- **Use case**: Next-gen production training and serving

From [Trillium sixth-generation TPU is in preview](https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview) (Google Cloud Blog, October 30, 2024):
> "We're pleased to announce that Trillium, our sixth-generation TPU, is now available to Google Cloud customers in preview."

### TPU Pod Slices

**Pod slice configurations** (subsets of full TPU pods):

| Generation | Single Host | Small Slice | Medium Slice | Large Slice | Full Pod |
|-----------|-------------|-------------|--------------|-------------|----------|
| v5e | 4 chips | 8-16 chips | 32-64 chips | 128 chips | 256 chips |
| v5p | 4 chips | 8-16 chips | 32-128 chips | 256-512 chips | 8,960 chips |
| v6e | 4 chips | 8-16 chips | 32-64 chips | 128 chips | 256 chips |

**Slice selection considerations**:
- Start small (4-8 chips) for prototyping
- Scale to 32-128 chips for production training
- Use full pods (256+ chips) for massive LLM training
- v5e: 2D mesh topology, simpler sharding
- v5p: 3D mesh topology, more complex but better scalability

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 227-255):
> "Maximum pod sizes: TPU v4: 16x16x16 = 4,096 chips, TPU v5p: 16x20x28 = 8,960 chips, TPU v5e/v6e: 16x16 = 256 chips (2D only)"

## Vertex AI Custom Training with TPUs

### Creating TPU Training Jobs

**Basic Vertex AI Custom Job with TPU v5e**:

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central2')

# Define custom training job with TPU v5e
job = aiplatform.CustomJob(
    display_name='jax-training-tpu-v5e',
    worker_pool_specs=[{
        'machine_spec': {
            'machine_type': 'ct5lp-hightpu-4t',  # 4-chip v5e
            'accelerator_type': 'TPU_V5_LITEPOD',
            'accelerator_count': 1
        },
        'replica_count': 1,
        'container_spec': {
            'image_uri': 'gcr.io/my-project/jax-trainer:latest',
            'args': [
                '--model=gpt2',
                '--batch_size=512',
                '--learning_rate=3e-4',
                '--steps=10000'
            ]
        },
    }]
)

job.run(
    service_account='training-sa@my-project.iam.gserviceaccount.com',
    tensorboard='projects/123/locations/us-central2/tensorboards/456'
)
```

**TPU v5e machine types on Vertex AI**:
- `ct5lp-hightpu-1t`: 1 chip (4 cores)
- `ct5lp-hightpu-4t`: 4 chips (16 cores) - single host
- `ct5lp-hightpu-8t`: 8 chips (32 cores)
- `ct5lp-hightpu-32t`: 32 chips (128 cores)
- `ct5lp-hightpu-256t`: 256 chips (1,024 cores) - full pod

From [Vertex AI release notes](https://docs.cloud.google.com/vertex-ai/docs/release-notes) (Google Cloud Documentation, accessed 2025-11-14):
> "Vertex AI custom training supports TPU v5e. For details, see Training with TPU accelerators. April 29, 2024."

### Multi-Host TPU Pod Training

**Training on 8-host v5e slice (32 chips)**:

```python
# Multi-host pod slice configuration
job = aiplatform.CustomJob(
    display_name='jax-llm-training-32chips',
    worker_pool_specs=[{
        'machine_spec': {
            'machine_type': 'ct5lp-hightpu-32t',  # 32-chip slice
            'accelerator_type': 'TPU_V5_LITEPOD',
            'accelerator_count': 1
        },
        'replica_count': 1,  # Vertex AI handles multi-host orchestration
        'container_spec': {
            'image_uri': 'gcr.io/my-project/jax-llm-trainer:latest',
            'args': [
                '--model=llama-7b',
                '--data_dir=gs://my-bucket/training-data',
                '--output_dir=gs://my-bucket/checkpoints',
                '--mesh_shape=8,4',  # 8 data parallel, 4 model parallel
                '--per_device_batch_size=32'
            ]
        },
        'disk_spec': {
            'boot_disk_size_gb': 100,
            'boot_disk_type': 'pd-ssd'
        }
    }]
)
```

**Key multi-host considerations**:
- All hosts share GCS data sources (no local data needed)
- JAX automatically detects multi-host topology
- PyTorch XLA requires `xmp.spawn()` for multi-process
- First host (host 0) is primary for checkpointing
- Use fast SSD boot disks for large dependency installations

## JAX Training on Vertex AI TPUs

### JAX Setup and Configuration

JAX is Google's native framework for TPU training, providing optimal performance through direct XLA compilation.

**Container setup for JAX on TPU v5e**:

```dockerfile
FROM gcr.io/cloud-tpus/tpu-vm-base:v2-base

# Install JAX with TPU support
RUN pip install --upgrade pip
RUN pip install "jax[tpu]>=0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install flax optax orbax-checkpoint

# Install additional ML libraries
RUN pip install datasets transformers einops
RUN pip install wandb tensorboard-plugin-profile

WORKDIR /app
COPY train.py /app/
COPY model.py /app/
COPY data.py /app/

ENTRYPOINT ["python", "train.py"]
```

**JAX version requirements**:
- TPU v5e: JAX 0.4.6+ (supports SparseCores)
- TPU v6e: JAX 0.4.20+ (Trillium optimizations)
- Update regularly for performance improvements

From [Train a model using TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e-training) (Google Cloud Documentation, accessed 2025-11-14):
> "TPU v5e supports JAX 0.4.6+, TensorFlow 2.15+, and PyTorch 2.1+"

### JAX Training Script for TPU

**Complete JAX training example** (referencing TPU fundamentals):

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Any

# Check TPU availability
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
# Expected: [TpuDevice(id=0), TpuDevice(id=1), ...]

class TransformerLM(nn.Module):
    """Transformer language model optimized for TPU."""
    vocab_size: int
    d_model: int = 1024  # Multiple of 128 for MXU efficiency
    n_heads: int = 8
    n_layers: int = 12

    @nn.compact
    def __call__(self, tokens):
        # Embedding
        x = nn.Embed(self.vocab_size, self.d_model)(tokens)

        # Transformer layers
        for _ in range(self.n_layers):
            # Multi-head attention
            x_norm = nn.LayerNorm()(x)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model
            )(x_norm)
            x = x + attn

            # FFN (4× expansion typical in transformers)
            x_norm = nn.LayerNorm()(x)
            ffn = nn.Dense(self.d_model * 4)(x_norm)
            ffn = nn.gelu(ffn)
            ffn = nn.Dense(self.d_model)(ffn)
            x = x + ffn

        # Output projection
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits

def create_train_state(rng, learning_rate, model, dummy_input):
    """Initialize training state."""
    params = model.init(rng, dummy_input)
    tx = optax.adamw(learning_rate, weight_decay=0.01)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit  # JIT compile for XLA optimization
def train_step(state, batch):
    """Single training step."""
    def loss_fn(params):
        logits = state.apply_fn(params, batch['input_ids'])
        # Shift for next-token prediction
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], batch['labels'][:, 1:]
        ).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
def train(config):
    # Initialize model
    model = TransformerLM(
        vocab_size=50257,
        d_model=1024,  # Multiple of 128 for TPU efficiency
        n_heads=8,
        n_layers=12
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
    state = create_train_state(rng, config['learning_rate'], model, dummy_input)

    # Training loop
    for step in range(config['total_steps']):
        batch = get_batch(config['batch_size'])  # Load from GCS
        state, loss = train_step(state, batch)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            # Log to W&B
            wandb.log({'loss': loss, 'step': step})

        if step % 1000 == 0:
            # Save checkpoint to GCS
            save_checkpoint(state, f"gs://my-bucket/checkpoints/step_{step}")

if __name__ == '__main__':
    config = {
        'learning_rate': 3e-4,
        'batch_size': 512,  # Large batch for TPU efficiency
        'total_steps': 10000
    }
    train(config)
```

**Key JAX TPU optimizations**:
1. **Batch size ≥ 240**: From TPU fundamentals, arithmetic intensity requires large batches
2. **Hidden dimensions**: Multiples of 128 (v5e) or 256 (v6e Trillium) for MXU utilization
3. **JIT compilation**: `@jax.jit` for XLA graph optimization
4. **Static shapes**: Avoid dynamic shapes to prevent recompilation

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 289-318):
> "For B << D and F = 4D (common in transformers): Intensity ≈ B (for large D). Need B > 240 to be FLOPs bound on v5e."

### Data Parallelism with pmap

**Multi-device data parallelism on TPU pod**:

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
    # Gradients automatically averaged across devices
    state = state.apply_gradients(grads=grads)
    return state, loss

# Reshape batch for pmap: [n_devices, per_device_batch, seq_len]
batch_per_device = global_batch_size // n_devices
batched_data = batch.reshape(n_devices, batch_per_device, seq_len)

# Run parallel training step
state, loss = parallel_train_step(state, batched_data)
```

**pmap sharding strategy**:
- Batch dimension automatically sharded across devices
- Model parameters replicated on each device
- Gradients reduced via all-reduce over ICI
- Suitable for models that fit on single TPU chip (≤16GB for v5e)

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
# Shard FFN weights across model dimension
with mesh:
    @pjit(
        in_shardings=(P('data', None), P('data', None)),  # Inputs: data parallel
        out_shardings=P('data', None),  # Output: data parallel
    )
    def sharded_train_step(state, batch):
        def loss_fn(params):
            # FFN weights sharded: [d_model, 4*d_model] -> shard second dim
            # This enables tensor parallelism for large FFN layers
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

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 364-418):
> "Model parallelism with pjit: Define device mesh, shard large matrix across devices"

## PyTorch XLA Training on Vertex AI TPUs

### PyTorch XLA Setup

PyTorch XLA enables PyTorch models on TPUs through XLA compilation, providing a familiar API for PyTorch developers.

**Container setup for PyTorch XLA**:

```dockerfile
FROM gcr.io/cloud-tpus/tpu-vm-pt-2.3:latest

# Install PyTorch XLA
RUN pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Install training dependencies
RUN pip install transformers datasets accelerate
RUN pip install wandb tensorboard

WORKDIR /app
COPY train_pytorch.py /app/
COPY model.py /app/

ENTRYPOINT ["python", "train_pytorch.py"]
```

**PyTorch XLA version compatibility**:
- TPU v5e: PyTorch 2.1+ with torch_xla 2.1+
- TPU v6e: PyTorch 2.3+ for Trillium support
- Use matching PyTorch and torch_xla versions

### PyTorch XLA Training Script

**Critical PyTorch XLA patterns**:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from transformers import GPT2LMHeadModel, GPT2Config

# Get TPU device
device = xm.xla_device()
print(f"Using device: {device}")

# Initialize model on TPU
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=1024,  # Multiple of 128 for TPU
    n_layer=12,
    n_head=8
)
model = GPT2LMHeadModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
def train(model, dataloader, optimizer, num_steps):
    model.train()

    # Wrap dataloader for TPU
    para_loader = pl.ParallelLoader(dataloader, [device])

    for step, batch in enumerate(para_loader.per_device_loader(device)):
        if step >= num_steps:
            break

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # CRITICAL: xm.optimizer_step for XLA
        xm.optimizer_step(optimizer)

        # CRITICAL: Mark step for XLA compilation
        xm.mark_step()

        # Logging (only on master device)
        if step % 10 == 0:
            loss_value = loss.item()  # Transfers to CPU
            print(f"Step {step}, Loss: {loss_value:.4f}")

# Run training
train(model, train_dataloader, optimizer, num_steps=10000)
```

**Critical PyTorch XLA requirements**:
1. **`xm.mark_step()`**: MUST call after optimizer step to trigger XLA graph execution
2. **`xm.optimizer_step()`**: Use XLA-aware optimizer step for gradient synchronization
3. **Device transfer minimization**: Keep tensors on device, minimize `.item()` calls
4. **Static shapes**: Avoid dynamic shapes (e.g., variable sequence lengths)

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 420-478):
> "Key PyTorch XLA concepts: 1. Lazy execution: Operations traced to graph, then compiled. 2. mark_step(): Triggers compilation and execution. 3. XLA compiler: Converts PyTorch ops to XLA ops, then to TPU code."

### Multi-Core PyTorch XLA Training

**Training across all TPU cores in a pod**:

```python
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    """Function executed on each TPU core."""
    device = xm.xla_device()
    print(f"Process {index} using device {device}")

    # Initialize model (replicated on each core)
    model = GPT2LMHeadModel(config).to(device)
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

    # Wrap for parallel loading
    para_loader = pl.ParallelLoader(train_loader, [device])

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(para_loader.per_device_loader(device)):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
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
            xm.save(model.state_dict(), f'gs://bucket/checkpoint_epoch_{epoch}.pt')

# Spawn processes on all TPU cores
if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(), nprocs=None)  # nprocs=None uses all available cores
```

**Multi-core considerations**:
- Each core runs independent process
- Model replicated on all cores (data parallel)
- Gradients automatically synchronized via `xm.optimizer_step()`
- Use `xm.mesh_reduce()` for cross-core metric aggregation
- Only master ordinal (rank 0) saves checkpoints

## TPU Performance Optimization for VLMs

### Vision Transformer Optimization

Vision transformers are well-suited for TPUs due to their matrix multiplication-heavy architecture.

**ViT optimization strategies**:

```python
# Optimize patch embedding for TPU MXU
class OptimizedViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_model=768, n_layers=12):
        super().__init__()

        # Ensure dimensions are multiples of 128 for v5e MXU
        assert d_model % 128 == 0, "d_model must be multiple of 128"

        # Patch embedding: [B, 3, H, W] -> [B, N, d_model]
        self.patch_embed = nn.Conv2d(
            3, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads=12)
            for _ in range(n_layers)
        ])

    def forward(self, images):
        # Batch size should be large for TPU efficiency
        # Recommended: B ≥ 240 for v5e
        x = self.patch_embed(images)  # [B, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]

        for block in self.transformer:
            x = block(x)

        return x
```

**ViT-specific optimizations**:
- Patch size 16×16 or 32×32 (reduces sequence length)
- Hidden dimension 768, 1024, 1536 (multiples of 128)
- Large batch sizes (256-512 images)
- Pre-compute patch embeddings for faster training

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 595-608):
> "Pad to multiples of 128: hidden_dim = 1024 # 128 × 8, fully utilizes MXU. For v6e Trillium, use multiples of 256."

### Language Model Optimization

**Transformer LM tuning for TPU**:

```python
class OptimizedTransformerLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_layers=24):
        super().__init__()

        # Embedding: ensure d_model is multiple of 128
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=8,
                ffn_dim=d_model * 4  # Standard 4× expansion
            )
            for _ in range(n_layers)
        ])

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        # Ensure batch_size × seq_len is large enough
        # Recommended: batch_size ≥ 64, seq_len ≥ 1024

        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)
        return logits
```

**LM optimization guidelines**:
1. **Sequence length**: 1024-2048 tokens (fixed length, no padding variation)
2. **Batch size**: 64-128 sequences (effective batch = batch × seq_len)
3. **FFN dimensions**: 4× hidden size (4096 for d_model=1024)
4. **Gradient checkpointing**: For very large models, trade compute for memory

### Multimodal Model Challenges

**VLM-specific TPU considerations**:

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_config, language_config):
        super().__init__()

        # Vision encoder (ViT)
        self.vision_encoder = OptimizedViT(
            img_size=224,
            patch_size=16,
            d_model=768
        )

        # Language decoder (Transformer LM)
        self.language_decoder = OptimizedTransformerLM(
            vocab_size=50257,
            d_model=1024,
            n_layers=12
        )

        # Cross-attention projector
        # Project vision features (768) to language dim (1024)
        # Ensure output is multiple of 128
        self.vision_proj = nn.Linear(768, 1024)

    def forward(self, images, input_ids):
        # Vision encoding (batch processing)
        vision_features = self.vision_encoder(images)  # [B, N_patches, 768]
        vision_features = self.vision_proj(vision_features)  # [B, N_patches, 1024]

        # Language decoding with vision context
        text_embeds = self.language_decoder.embed(input_ids)

        # Concatenate vision and text tokens
        combined = torch.cat([vision_features, text_embeds], dim=1)

        # Process through language decoder
        for block in self.language_decoder.blocks:
            combined = block(combined)

        logits = self.language_decoder.lm_head(combined)
        return logits
```

**VLM TPU challenges**:
- **Mixed sequence lengths**: Vision patches + text tokens
- **Cross-modal attention**: Additional compute overhead
- **Memory usage**: Both encoders loaded simultaneously
- **Batch size trade-offs**: Larger images = smaller batches

**Solutions**:
1. Fixed vision patch count (e.g., always 196 patches for 224×224 / 16)
2. Fixed text sequence length (pad to max length)
3. Stage training: Pre-train vision and language separately, then fine-tune
4. Use gradient checkpointing for memory efficiency

## Cost Optimization Strategies

### TPU vs GPU Cost Comparison

**Pricing comparison** (approximate, us-central2 region):

| Resource | Cost per hour | Memory/chip | Use case |
|----------|--------------|-------------|----------|
| v5e (1 chip) | $1.20 | 16GB | Cost-optimized training |
| v5e (4 chips) | $4.80 | 64GB | Small-scale production |
| v5e (32 chips) | $38.40 | 512GB | Medium-scale LLM |
| A100 40GB | $3.67 | 40GB | GPU baseline |
| A100 80GB | $5.89 | 80GB | Large model GPU |
| H100 80GB | $9.25 | 80GB | Latest GPU |

**Cost efficiency analysis**:
- v5e: ~60-70% lower cost per FLOP vs A100
- v5p: ~40-50% lower cost per FLOP vs H100
- Best for: Transformer training, large batch sizes
- Less efficient for: Small batches, irregular workloads

From [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) (CloudOptimo, April 15, 2025):
> "TPUs pack more memory per system and deliver extremely high throughput for AI workloads, often surpassing equivalent GPU setups in training and inference tasks."

### Preemptible TPU Training

**Using preemptible VMs for 60-80% cost savings**:

```python
job = aiplatform.CustomJob(
    display_name='preemptible-tpu-training',
    worker_pool_specs=[{
        'machine_spec': {
            'machine_type': 'ct5lp-hightpu-8t',
            'accelerator_type': 'TPU_V5_LITEPOD',
            'accelerator_count': 1
        },
        'replica_count': 1,
        'container_spec': {
            'image_uri': 'gcr.io/my-project/jax-trainer:latest'
        },
        # Enable preemptible for cost savings
        'spot': True,
        # Restart policy for preemptions
        'restart_job_on_worker_restart': True
    }],
    # Checkpointing configuration
    base_output_dir='gs://my-bucket/checkpoints'
)
```

**Preemptible best practices**:
1. **Frequent checkpointing**: Every 10-15 minutes
2. **Resume from checkpoint**: Automatic restart with `restart_job_on_worker_restart`
3. **Idempotent training**: Handle mid-batch preemptions gracefully
4. **Time of day**: Run during off-peak hours for lower preemption rates
5. **Acceptable for**: Long training runs (>4 hours) with good checkpointing

**Cost savings calculation**:
- Regular v5e-8: $4.80/hour
- Preemptible v5e-8: ~$1.20/hour (75% savings)
- With 20% preemption rate, effective cost: ~$1.44/hour (70% savings)

### Batch Size Tuning for Efficiency

**Finding optimal batch size**:

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
    # Need batch_size > required_intensity to be FLOPs bound
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

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 334-346):
> "Example: Matrix multiply int8[16384, 4096] @ int8[B, 4096] on TPU v5e. FLOPs bound when B > 271 (from HBM), FLOPs bound when B > 11 (from VMEM)"

## Integration with Vertex AI Services

### Weights & Biases Integration

**W&B tracking for Vertex AI TPU jobs**:

```python
import wandb

# Initialize W&B in training script
wandb.init(
    project='tpu-training',
    name=f'v5e-8-{datetime.now().strftime("%Y%m%d-%H%M")}',
    config={
        'tpu_type': 'v5e-8',
        'batch_size': 512,
        'learning_rate': 3e-4,
        'model': 'gpt2-medium'
    }
)

# Log metrics during training
def train_step_with_logging(state, batch, step):
    state, loss = train_step(state, batch)

    if step % 10 == 0:
        wandb.log({
            'loss': float(loss),
            'learning_rate': float(get_lr(step)),
            'step': step
        })

    return state, loss
```

**W&B Launch for managed training**:
- Submit jobs directly from W&B UI
- Automatic hyperparameter sweeps on TPUs
- Cost tracking per experiment
- Artifact versioning for datasets and models

### TensorBoard Profiling

**Enable TPU profiling in Vertex AI**:

```python
# In training script
import torch_xla.debug.profiler as xp

# Start profiler server
server = xp.start_server(9012)

# Training code here...

# Capture profile trace
if step == 1000:  # Profile specific step
    xp.trace(
        'localhost:9012',
        '/tmp/profile',
        duration_ms=10000
    )

# Upload profile to GCS
# Then view in TensorBoard: tensorboard --logdir=gs://bucket/profiles
```

**Profile analysis**:
- **Step time breakdown**: Compilation vs execution
- **Memory bandwidth**: HBM utilization percentage
- **Compute utilization**: Percentage of peak FLOPs achieved
- **ICI communication**: Multi-chip transfer times

From [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) (lines 759-796):
> "XProf Profiling: Capture trace, view in TensorBoard. Analyze: step time breakdown, memory bandwidth utilization, compute utilization, communication time"

### Model Registry Deployment

**Save trained model to Vertex AI Model Registry**:

```python
# After training completes
model_uri = 'gs://my-bucket/models/gpt2-finetuned'

# Upload model to registry
model = aiplatform.Model.upload(
    display_name='gpt2-tpu-finetuned',
    artifact_uri=model_uri,
    serving_container_image_uri='gcr.io/my-project/serving:latest',
    parameters_schema_uri=None,
    instance_schema_uri=None
)

# Deploy to endpoint (can use TPU for serving)
endpoint = model.deploy(
    machine_type='ct5lp-hightpu-1t',  # Single TPU chip for serving
    accelerator_type='TPU_V5_LITEPOD',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100
)
```

## Debugging and Troubleshooting

### Common TPU Training Issues

**Issue 1: Out of Memory (OOM)**

```python
# Symptom: ResourceExhaustedError during training

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

**Issue 2: Low compute utilization (<50%)**

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

**Issue 3: High compilation overhead**

```python
# Symptom: First steps are very slow, frequent recompilations

# Solutions:
# 1. Use static shapes (avoid dynamic shapes)
# Bad:
x = x[:actual_length]  # Dynamic slice

# Good:
mask = jnp.arange(max_length) < actual_length
x = x * mask[:, None]  # Static shape with masking

# 2. Use static_argnums in jit
@jax.jit(static_argnums=(1,))  # Mark shape args as static
def process(data, length):
    return data[:length].sum()

# 3. Reduce unique graph shapes
# Use consistent padding lengths
```

**Issue 4: Multi-host synchronization failures**

```python
# Symptom: Hanging during multi-host training

# Solutions:
# 1. Ensure all hosts run same number of steps
num_steps = min(num_steps, steps_on_shortest_host)

# 2. Use barriers for synchronization
xm.rendezvous('training_complete')  # PyTorch XLA
jax.experimental.multihost_utils.sync_global_devices('step_complete')  # JAX

# 3. Check GCS permissions on all hosts
# All hosts need read/write to checkpoint bucket
```

### Performance Debugging Workflow

**Step-by-step performance optimization**:

1. **Baseline metrics**:
   - Measure: steps/sec, memory usage, compute utilization
   - Target: >80% compute utilization for efficient training

2. **Profile single step**:
   ```python
   # Capture profile of one training step
   xp.trace('localhost:9012', '/tmp/profile', duration_ms=5000)
   ```

3. **Analyze profile**:
   - Check compilation time (<10% of step time ideal)
   - Verify memory bandwidth (>70% for memory-bound)
   - Check ICI communication (minimal for data parallel)

4. **Optimize**:
   - Increase batch size if compute-bound
   - Fix dynamic shapes if high compilation
   - Improve sharding if high ICI communication

5. **Iterate**:
   - Re-profile after changes
   - Track steps/sec improvement

## arr-coc-0-1 TPU Training Feasibility

### Current Implementation Analysis

The arr-coc-0-1 project currently uses **PyTorch on A100 GPUs**. TPU migration considerations:

**Current architecture** (arr-coc-0-1):
- Vision encoder: Qwen3-VL (pretrained)
- Relevance scorer: Custom texture + knowing modules
- Training: Fine-tuning on A100 (single-GPU dev, 8-GPU production)

**TPU feasibility assessment**:

✅ **Good fit**:
- Transformer-based architecture (ViT encoder)
- Large batch size training (K=200 patches × batch_size)
- Matrix multiplication heavy (attention, FFN layers)

⚠️ **Challenges**:
- Custom CUDA kernels in Qwen3-VL (may not port directly)
- Dynamic texture array sizes (would need padding)
- Mixed CPU/GPU operations in relevance scoring

### Migration Path to TPU v5e

**Recommended approach** (if pursuing TPU):

**Phase 1: JAX port of core modules** (~2-3 weeks)
```python
# Port knowing.py scorers to JAX
class InformationScorer(nn.Module):
    """JAX/Flax version of propositional scorer."""
    @nn.compact
    def __call__(self, texture_features):
        # Convert PyTorch ops to JAX
        entropy = -jnp.sum(
            texture_features * jnp.log(texture_features + 1e-10),
            axis=-1
        )
        return entropy

# Port balancing.py to JAX
# Port attending.py to JAX
```

**Phase 2: Test on TPU v5e (4 chips)** (~1 week)
- Deploy to Vertex AI with v5e-4
- Benchmark vs A100 single-GPU
- Validate numerical accuracy

**Phase 3: Scale to v5e pod (32 chips)** (~1 week)
- Data parallel training across 32 chips
- Cost comparison: v5e-32 vs A100-8

**Estimated effort**: 4-6 weeks for full migration

### Cost-Benefit Analysis

**Training cost comparison** (arr-coc-0-1 typical run):

| Configuration | Hardware | Cost/hour | Hours | Total cost |
|--------------|----------|-----------|-------|------------|
| Current (dev) | A100 40GB × 1 | $3.67 | 10 | $36.70 |
| Current (prod) | A100 80GB × 8 | $47.12 | 4 | $188.48 |
| TPU option 1 | v5e-8 | $4.80 | 8 | $38.40 |
| TPU option 2 | v5e-32 | $38.40 | 2 | $76.80 |

**Analysis**:
- **v5e-8**: Similar cost to single A100, 80% speedup potential = lower total cost
- **v5e-32**: 50% more expensive than A100-8, but 2× speedup = comparable efficiency
- **Preemptible v5e-32**: $19.20/hour × 2.4 hours = $46.08 (60% savings)

**Recommendation**:
- **Don't migrate immediately** - Current PyTorch/A100 workflow is mature
- **Consider for future scale** - If training >100 jobs/month, TPU cost savings significant
- **Prototype on v5e-4** first to validate JAX port before committing

## Sources

**Official Documentation**:
- [Training with TPU accelerators | Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm) - Google Cloud Documentation (accessed 2025-11-14)
- [Train a model using TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e-training) - Google Cloud Documentation (accessed 2025-11-14)
- [Vertex AI release notes](https://docs.cloud.google.com/vertex-ai/docs/release-notes) - Google Cloud Documentation (accessed 2025-11-14)
- [Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/tpu) - Google Cloud (accessed 2025-11-14)

**Technical References**:
- [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - Karpathy Deep Oracle knowledge base (835 lines, comprehensive TPU architecture, JAX/PyTorch XLA programming patterns, performance optimization)
- [practical-implementation/32-vertex-ai-gpu-tpu.md](../practical-implementation/32-vertex-ai-gpu-tpu.md) - Vertex AI GPU and TPU specifications

**Blog Posts and Announcements**:
- [Introducing Cloud TPU v5p and AI Hypercomputer](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) - Google Cloud Blog (December 6, 2023)
- [Trillium sixth-generation TPU is in preview](https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview) - Google Cloud Blog (October 30, 2024)
- [Inside the Ironwood TPU codesigned AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack) - Google Cloud Blog (November 6, 2025)
- [How Lightricks trains video diffusion models at scale with JAX on TPU](https://cloud.google.com/blog/products/media-entertainment/how-lightricks-trains-video-diffusion-models-at-scale-with-jax-on-tpu/) - Google Cloud Blog (3 days ago, accessed 2025-11-14)

**Community Resources**:
- [TPU vs GPU: What's the Difference in 2025?](https://www.cloudoptimo.com/blog/tpu-vs-gpu-what-is-the-difference-in-2025/) - CloudOptimo (April 15, 2025)
- [Train a GPT2 model with JAX on TPU for free](https://developers.googleblog.com/en/train-gpt2-model-with-jax-on-tpu/) - Google Developers Blog (August 12, 2025)
- [Google Cloud Platform Resources TPU](https://www.gcpweekly.com/gcp-resources/tag/tpu/) - GCP Weekly (accessed 2025-11-14)

**Search Results**:
- Google Search: "Vertex AI TPU v5e v5p v6e training 2024 2025" (accessed 2025-11-14)
- Google Search: "Vertex AI TPU pod slices PyTorch XLA JAX" (accessed 2025-11-14)
- Google Search: "Vertex AI TPU vs GPU cost comparison 2024" (accessed 2025-11-14)
- Google Search: "site:cloud.google.com Vertex AI TPU custom training job PyTorch XLA" (accessed 2025-11-14)
- Google Search: "JAX training on Vertex AI TPU v5p v5e 2024" (accessed 2025-11-14)

---

**Last updated**: 2025-11-14 (PART 4 execution - Vertex AI Advanced Integration expansion)
