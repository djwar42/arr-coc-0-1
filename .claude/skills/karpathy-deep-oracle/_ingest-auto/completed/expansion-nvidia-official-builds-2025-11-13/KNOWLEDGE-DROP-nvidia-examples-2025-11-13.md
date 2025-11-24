# KNOWLEDGE DROP: NVIDIA Deep Learning Examples

**Runner**: PART 2 - NVIDIA Deep Learning Examples Training Patterns
**Timestamp**: 2025-11-13 21:49 PST
**Status**: SUCCESS ✓

---

## Knowledge File Created

**File**: `cuda/15-nvidia-deep-learning-examples.md`
**Size**: ~400 lines (28.7 KB)
**Focus**: Official NVIDIA training patterns from DeepLearningExamples repository

---

## GitHub Repositories Analyzed

### Primary Repository

**NVIDIA/DeepLearningExamples**
- URL: https://github.com/NVIDIA/DeepLearningExamples
- Stars: 14,576
- Forks: 3,378
- Description: State-of-the-Art Deep Learning examples (easy to train/deploy)
- Last Updated: 2023-03-30

### Files Extracted (Direct Source Code)

1. **PyTorch/Classification/ConvNets/main.py** (678 lines, 20.3 KB)
   - Complete training script entry point
   - DDP initialization, checkpoint resume, AMP setup
   - Argument parsing, LR schedulers, model architecture selection

2. **PyTorch/Classification/ConvNets/image_classification/training.py** (435 lines, 13.1 KB)
   - Executor pattern (model + loss + AMP abstraction)
   - Trainer class (gradient accumulation + EMA)
   - Training/validation loop implementations
   - DDP with CUDA stream overlap

3. **PyTorch/Classification/ConvNets/image_classification/dataloaders.py** (577 lines, 17.4 KB)
   - DALI GPU/CPU pipelines (nvJPEG hardware decoding)
   - PyTorch DataLoader with prefetching
   - Fast collate function (uint8 → float optimization)
   - DistributedSampler patterns

---

## Key Discoveries from Official NVIDIA Code

### 1. DDP Initialization Pattern (Production-Ready)

```python
# NVIDIA's official DDP setup
if "WORLD_SIZE" in os.environ:
    args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.local_rank = int(os.environ["LOCAL_RANK"])

if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method="env://")
    args.world_size = torch.distributed.get_world_size()

# DDP model wrapping with CUDA stream overlap
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
torch.cuda.current_stream().wait_stream(s)
```

**Critical optimizations:**
- Environment variable detection (torch.distributed.launch standard)
- NCCL backend with `init_method="env://"`
- DDP initialization in separate stream (non-blocking)
- CPU affinity for NUMA optimization

### 2. Seed Management (Reproducibility)

```python
# Unique seed per rank and DataLoader worker
torch.manual_seed(args.seed + args.local_rank)
torch.cuda.manual_seed(args.seed + args.local_rank)
np.random.seed(seed=args.seed + args.local_rank)

def _worker_init_fn(id):
    np.random.seed(seed=args.seed + args.local_rank + id)
    random.seed(args.seed + args.local_rank + id)
```

**Pattern**: `seed + local_rank + worker_id` ensures no duplicate augmentations across GPUs/workers

### 3. DALI High-Performance Data Loading

```python
# Hardware JPEG decoder with memory padding
self.decode = ops.ImageDecoder(
    device="mixed",  # GPU-accelerated nvJPEG
    output_type=types.RGB,
    device_memory_padding=211025920,  # Pre-allocate 211MB GPU
    host_memory_padding=140544512,    # Pre-allocate 140MB CPU
)

# GPU-side augmentation pipeline
self.res = ops.RandomResizedCrop(device="gpu", ...)
self.cmnp = ops.CropMirrorNormalize(device="gpu", ...)  # Fused op
```

**Performance**: 2-3× faster than PyTorch DataLoader via:
- nvJPEG hardware decoding
- GPU-side augmentation (no CPU bottleneck)
- Fused operations (CropMirrorNormalize = 3 ops in 1 kernel)
- Pre-allocated buffers (no reallocations)

### 4. Prefetching Pattern (PyTorch DataLoader)

```python
stream = torch.cuda.Stream()
for next_input, next_target in loader:
    with torch.cuda.stream(stream):
        next_input = next_input.cuda(non_blocking=True)
        next_target = next_target.cuda(non_blocking=True)
        next_input = next_input.float()
        next_input = next_input.sub_(mean).div_(std)

    if not first:
        yield input, target

    torch.cuda.current_stream().wait_stream(stream)
    input = next_input
    target = next_target
```

**Optimization**: While GPU trains on batch N, CUDA stream loads batch N+1

### 5. Gradient Accumulation (Official Pattern)

```python
def train_step(self, input, target):
    loss = self.executor.forward_backward(input, target)
    self.steps_since_update += 1

    if self.steps_since_update == self.grad_acc_steps:
        if self.executor.scaler is not None:
            self.executor.scaler.step(self.optimizer)
            self.executor.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.steps_since_update = 0
```

**Usage**: Simulate larger batch sizes without OOM

### 6. Mixed Precision (AMP Configuration)

```python
scaler = torch.cuda.amp.GradScaler(
    init_scale=args.static_loss_scale,  # Configurable (default 1.0)
    growth_factor=2,
    backoff_factor=0.5,
    growth_interval=100,  # Conservative for stability
    enabled=args.amp,
)

# Forward-backward with AMP
with autocast(enabled=self.amp):
    loss = self.loss(self.model(input), target)
    loss /= self.divide_loss  # For gradient accumulation

self.scaler.scale(loss).backward()
```

**NVIDIA choices**: `growth_interval=100` (conservative), configurable `init_scale`

---

## Knowledge File Structure

### Section 1: Official Training Script Structure
- Main entry point pattern (`prepare_for_training` → `train_loop`)
- Distributed training initialization
- Seed management for reproducibility
- Checkpoint resume logic
- Mixed precision setup
- Learning rate scheduler options (step, cosine, linear)

### Section 2: Distributed Training & Mixed Precision
- Executor pattern (training abstraction)
- DDP with CUDA stream overlap
- Forward-backward with AMP
- Trainer class (gradient accumulation + EMA)
- Training loop with loss reduction
- Validation loop

### Section 3: Data Loading Optimization
- DALI pipelines (HybridTrainPipe, HybridValPipe)
- nvJPEG hardware decoding
- GPU-side augmentation
- PyTorch DataLoader with prefetching
- Fast collate function
- Memory format control (NCHW vs NHWC)

### Section 4: Production Patterns & Best Practices
- Checkpoint management (epoch + best)
- Command-line argument patterns
- Memory format optimization (NHWC for Tensor Cores)
- Early stopping
- Profiling mode
- GPU affinity optimization
- Timeout handler (spot instances)

---

## Official NVIDIA Optimizations Documented

1. **DDP CUDA Stream Overlap** - Non-blocking DDP initialization
2. **DALI Hardware Decoding** - nvJPEG with pre-allocated buffers
3. **Prefetching Pipeline** - H2D copy overlaps with compute
4. **Fast Collate** - Defer float conversion until GPU (4× smaller transfers)
5. **Memory Format** - NHWC for Tensor Cores on Ampere+
6. **Gradient Accumulation** - Official pattern with AMP support
7. **EMA Support** - Exponential Moving Average for better generalization
8. **CPU Affinity** - NUMA-aware process binding

---

## Production Patterns Extracted

### Training Configuration (ResNet-50 ImageNet Defaults)
- Epochs: 90
- Batch size: 256 per GPU
- Learning rate: 0.1
- Optimizer: SGD with momentum 0.9
- Weight decay: 1e-4
- LR schedule: Step decay at [30, 60, 80] epochs (0.1× multiplier)
- Warmup: 5 epochs (for large batch distributed training)

### Data Loading Best Practices
- DALI-GPU: Maximum throughput (3× faster than PyTorch)
- DALI-CPU: 1.5× faster than PyTorch
- PyTorch: Debugging, custom augmentations
- Prefetch factor: 2
- Persistent workers: True
- Pin memory: True
- Drop last: True (critical for DDP)

### Checkpoint Strategy
- Save epoch checkpoints: `checkpoint_0001.pth.tar`, `checkpoint_0002.pth.tar`
- Save best checkpoint: `model_best.pth.tar`
- Keep last N checkpoints to save disk
- Store: epoch, best_prec1, model state, optimizer state, EMA state

---

## Statistics

**Lines of Code Analyzed**: 1,690 lines
**Knowledge File**: 400 lines
**GitHub Files**: 3 core training scripts
**Repository**: 14.6k stars, production-validated

**Coverage:**
- Training loop architecture ✓
- DDP initialization ✓
- Mixed precision (AMP) ✓
- DALI data loading ✓
- PyTorch DataLoader optimization ✓
- Gradient accumulation ✓
- EMA support ✓
- Checkpointing ✓
- Production best practices ✓

---

## Next Steps (Other PARTs)

This PART 2 focused on **training examples and patterns**. Remaining PARTs:

- **PART 1**: NVIDIA PyTorch container builds (Dockerfiles, compilation flags)
- **PART 3**: CUDA compilation best practices (nvcc flags from samples)
- **PART 4**: Performance optimization patterns (CUTLASS, apex, kernel fusion)

---

**Completion**: PART 2 complete ✓ - Ready for oracle integration
