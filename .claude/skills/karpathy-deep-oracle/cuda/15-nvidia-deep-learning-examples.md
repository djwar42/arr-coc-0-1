# NVIDIA Deep Learning Examples: Official Training Patterns

**Source**: NVIDIA/DeepLearningExamples GitHub repository (accessed 2025-11-13)
**Focus**: Production-ready training scripts for ResNet, BERT, GPT - official NVIDIA patterns
**Repository**: https://github.com/NVIDIA/DeepLearningExamples (14.6k stars, 3.4k forks)

---

## Overview

NVIDIA/DeepLearningExamples provides State-of-the-Art Deep Learning examples that are **easy to train and deploy**, achieving the best reproducible accuracy and performance with NVIDIA CUDA-X software stack. These are NVIDIA's OFFICIAL production patterns used in NGC containers.

**What makes this special:**
- Monthly NGC container updates with latest NVIDIA optimizations
- Rigorous QA process ensuring best performance on Volta/Turing/Ampere/Hopper GPUs
- Complete examples: ResNet-50, EfficientNet, BERT, GPT, DLRM, Mask R-CNN
- Production-ready patterns for DDP, mixed precision, DALI data loading

---

## Section 1: Official Training Script Structure (main.py Analysis)

From [PyTorch/Classification/ConvNets/main.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/main.py)

### Main Entry Point Pattern

```python
def main(args, model_args, model_arch):
    exp_start_time = time.time()

    (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
        best_prec1,
    ) = prepare_for_training(args, model_args, model_arch)

    train_loop(
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs) if args.run_epochs != -1 else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
        keep_last_n_checkpoints=args.gather_checkpoints,
        topk=args.topk,
    )
```

**Key Insights:**
- **Separation of concerns**: `prepare_for_training()` handles setup, `train_loop()` handles execution
- **Flexible epoch control**: `--run_epochs` for checkpointing runs, early stopping support
- **Profiling mode**: `--prof` runs only N iterations for benchmarking
- **Evaluation mode**: `--evaluate` skips training, only validates checkpoint
- **Production ready**: All edge cases handled (resume, checkpointing, distributed)

### Distributed Training Initialization

```python
def prepare_for_training(args, model_args, model_arch):
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    affinity = set_affinity(args.gpu, mode=args.gpu_affinity)
    print(f"Training process {args.local_rank} affinity: {affinity}")
```

**NVIDIA's DDP pattern:**
- Environment variable detection (`WORLD_SIZE`, `LOCAL_RANK`)
- NCCL backend with `init_method="env://"` (torch.distributed.launch standard)
- CPU affinity optimization (critical for multi-GPU performance)
- GPU assignment: `local_rank % torch.cuda.device_count()` (handles multi-node)

### Seed Management for Reproducibility

```python
if args.seed is not None:
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    np.random.seed(seed=args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)

    def _worker_init_fn(id):
        # Worker process should inherit its affinity from parent
        affinity = os.sched_getaffinity(0)
        print(f"Process {args.local_rank} Worker {id} set affinity to: {affinity}")

        np.random.seed(seed=args.seed + args.local_rank + id)
        random.seed(args.seed + args.local_rank + id)
```

**Critical pattern**: `seed + local_rank + worker_id` ensures:
- Each rank gets different random state (no duplicate augmentations)
- Each DataLoader worker gets unique seed
- Reproducible training across runs with same seed

### Checkpoint Resume Logic

```python
if args.resume is not None:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(
            args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
        )
        start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]
        model_state = checkpoint["state_dict"]
        optimizer_state = checkpoint["optimizer"]
        if "state_dict_ema" in checkpoint:
            model_state_ema = checkpoint["state_dict_ema"]

        if start_epoch >= args.epochs:
            print(f"Launched training for {args.epochs}, checkpoint already run {start_epoch}")
            exit(1)
```

**NVIDIA's checkpoint pattern:**
- `map_location` ensures checkpoint loaded to correct GPU in multi-GPU setup
- EMA state dict support (`state_dict_ema`)
- Validation: Don't resume if checkpoint already finished training
- Stores epoch, best accuracy, model state, optimizer state, EMA state

### Mixed Precision Setup

```python
scaler = torch.cuda.amp.GradScaler(
    init_scale=args.static_loss_scale,
    growth_factor=2,
    backoff_factor=0.5,
    growth_interval=100,
    enabled=args.amp,
)

executor = Executor(
    model,
    loss(),
    cuda=True,
    memory_format=memory_format,
    amp=args.amp,
    scaler=scaler,
    divide_loss=batch_size_multiplier,
    ts_script=args.jit == "script",
)
```

**Official NVIDIA AMP pattern:**
- `GradScaler` with configurable `init_scale` (default 1.0, can set higher for stability)
- Growth interval: 100 iterations (conservative for stability)
- `enabled=args.amp` allows easy disable for debugging
- `divide_loss` for gradient accumulation support

### Learning Rate Scheduler Options

```python
if args.lr_schedule == "step":
    lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup)
elif args.lr_schedule == "cosine":
    lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, end_lr=args.end_lr)
elif args.lr_schedule == "linear":
    lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs)
```

**NVIDIA's scheduler choices:**
- **Step**: Decay at epochs 30, 60, 80 (ImageNet standard)
- **Cosine**: Smooth decay to `end_lr` (popular for modern training)
- **Linear**: Linear warmup + decay (stable for large batch sizes)
- All support warmup (critical for large batch distributed training)

---

## Section 2: Distributed Training & Mixed Precision (training.py Analysis)

From [PyTorch/Classification/ConvNets/image_classification/training.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/training.py)

### Executor: NVIDIA's Training Abstraction

```python
class Executor:
    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module],
        cuda: bool = True,
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
    ):
        assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"

        def xform(m: nn.Module) -> nn.Module:
            if cuda:
                m = m.cuda()
            m.to(memory_format=memory_format)
            return m

        self.model = xform(model)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        self.loss = xform(loss) if loss is not None else None
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None
```

**Why Executor pattern:**
- Encapsulates model, loss, AMP, memory format in one object
- Lazy compilation of forward/backward passes
- Supports TorchScript compilation (`torch.jit.script`)
- Memory format flexibility (NCHW vs NHWC for Tensor Cores)
- Clean separation between training logic and execution

### DDP Initialization with CUDA Stream Overlap

```python
def distributed(self, gpu_id):
    self.is_distributed = True
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
    torch.cuda.current_stream().wait_stream(s)
```

**Critical optimization:**
- DDP initialization in separate CUDA stream (non-blocking)
- Overlaps DDP setup with other initialization
- `wait_stream()` ensures synchronization before training starts
- Used in all NVIDIA's official examples

### Forward-Backward with AMP

```python
def _fwd_bwd_fn(
    self,
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    with autocast(enabled=self.amp):
        loss = self.loss(self.model(input), target)
        loss /= self.divide_loss

    self.scaler.scale(loss).backward()
    return loss
```

**NVIDIA's AMP pattern:**
- `autocast(enabled=self.amp)` allows runtime enable/disable
- Loss division for gradient accumulation (`divide_loss`)
- `scaler.scale(loss).backward()` handles overflow protection
- Returns unscaled loss for logging

### Validation Forward Pass

```python
def _forward_fn(
    self, input: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad(), autocast(enabled=self.amp):
        output = self.model(input)
        loss = None if self.loss is None else self.loss(output, target)

    return output if loss is None else loss, output
```

**Validation optimizations:**
- `torch.no_grad()` disables gradient tracking (memory savings)
- AMP enabled for consistency with training (same numerical behavior)
- Returns both loss and output for accuracy computation

### Trainer: Gradient Accumulation & EMA

```python
class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        self.optimizer.zero_grad(set_to_none=True)
        self.steps_since_update = 0
```

**Trainer features:**
- Gradient accumulation tracking (`grad_acc_steps`)
- Exponential Moving Average support (EMA model for better generalization)
- `set_to_none=True` for zero_grad (faster than setting to zero)
- Deep copy executor for EMA (separate model instance)

### Training Step with Gradient Accumulation

```python
def train_step(self, input, target, step=None):
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

    torch.cuda.synchronize()

    if self.use_ema:
        self.ema(self.executor.model, step=step)

    return loss
```

**Official gradient accumulation pattern:**
- Accumulate gradients for `grad_acc_steps` iterations
- Only optimizer step every N iterations (simulates larger batch size)
- `torch.cuda.synchronize()` ensures accurate timing
- EMA update after every step (not just optimizer steps)

### Training Loop with Loss Reduction

```python
def train(
    train_step,
    train_loader,
    lr_scheduler,
    grad_scale_fn,
    log_fn,
    timeout_handler,
    prof=-1,
    step=0,
):
    interrupted = False
    end = time.time()
    data_iter = enumerate(train_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr = lr_scheduler(i)
        data_time = time.time() - end

        loss = train_step(input, target, step=step + i)
        it_time = time.time() - end

        with torch.no_grad():
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.detach())
            else:
                reduced_loss = loss.detach()

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            lr=lr,
            loss=reduced_loss.item(),
            grad_scale=grad_scale_fn(),
        )

        end = time.time()
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break
```

**NVIDIA's training loop optimizations:**
- **Loss reduction**: `reduce_tensor()` averages loss across all GPUs
- **Throughput tracking**: Images per second (IPS) for compute only and total (data+compute)
- **Data time vs compute time**: Identifies data loading bottlenecks
- **Gradient scale logging**: Monitors AMP scaler behavior
- **Profiling mode**: `prof` parameter runs only N iterations
- **Timeout handler**: Graceful shutdown for preemptible instances

### Validation Loop

```python
def validate(infer_fn, val_loader, log_fn, prof=-1, with_loss=True, topk=5):
    top1 = log.AverageMeter()
    end = time.time()
    data_iter = enumerate(val_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        if with_loss:
            loss, output = infer_fn(input, target)
        else:
            output = infer_fn(input)

        with torch.no_grad():
            precs = utils.accuracy(output.data, target, topk=(1, topk))

            if torch.distributed.is_initialized():
                if with_loss:
                    reduced_loss = utils.reduce_tensor(loss.detach())
                precs = map(utils.reduce_tensor, precs)
            else:
                if with_loss:
                    reduced_loss = loss.detach()

        precs = map(lambda t: t.item(), precs)
        infer_result = {f"top{k}": (p, bs) for k, p in zip((1, topk), precs)}

        if with_loss:
            infer_result["loss"] = (reduced_loss.item(), bs)

        torch.cuda.synchronize()
        it_time = time.time() - end
        top1.record(infer_result["top1"][0], bs)

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            **infer_result,
        )

        end = time.time()

    return top1.get_val()
```

**Validation best practices:**
- Top-1 and Top-K accuracy (configurable K, default 5)
- Distributed accuracy reduction (all-reduce across GPUs)
- Same throughput metrics as training
- Returns final top-1 for checkpointing
- `torch.cuda.synchronize()` for accurate timing

---

## Section 3: Data Loading Optimization (dataloaders.py Analysis)

From [PyTorch/Classification/ConvNets/image_classification/dataloaders.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py)

### DALI: NVIDIA's High-Performance Data Loading

```python
class HybridTrainPipe(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        interpolation,
        crop,
        dali_cpu=False,
    ):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=True,
            pad_last_batch=True,
        )

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )
```

**DALI optimizations:**
- **nvJPEG hardware decoder**: `device="mixed"` uses GPU-accelerated JPEG decoding
- **Memory padding**: Pre-allocates buffers (211MB device, 140MB host) to avoid reallocations
- **Automatic sharding**: `shard_id=rank, num_shards=world_size` for DDP
- **Random shuffle**: Per-shard shuffle for each GPU
- **Pad last batch**: Ensures all ranks process same number of batches

### DALI Data Augmentation Pipeline

```python
self.res = ops.RandomResizedCrop(
    device=dali_device,
    size=[crop, crop],
    interp_type=interpolation,
    random_aspect_ratio=[0.75, 4.0 / 3.0],
    random_area=[0.08, 1.0],
    num_attempts=100,
    antialias=False,
)

self.cmnp = ops.CropMirrorNormalize(
    device="gpu",
    dtype=types.FLOAT,
    output_layout=types.NCHW,
    crop=(crop, crop),
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
)
self.coin = ops.CoinFlip(probability=0.5)

def define_graph(self):
    rng = self.coin()
    self.jpegs, self.labels = self.input(name="Reader")
    images = self.decode(self.jpegs)
    images = self.res(images)
    output = self.cmnp(images.gpu(), mirror=rng)
    return [output, self.labels]
```

**DALI pipeline advantages:**
- **GPU-side augmentation**: All ops run on GPU (faster than CPU)
- **Fused operations**: CropMirrorNormalize fuses 3 ops into 1 kernel
- **No PIL overhead**: Direct JPEG → Tensor on GPU
- **ImageNet standard**: Aspect ratio [0.75, 1.33], area [0.08, 1.0], 100 attempts
- **Mean/std in [0, 255]**: DALI works with uint8, normalizes to float

### PyTorch DataLoader with Prefetching

```python
class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target
```

**NVIDIA's prefetching pattern:**
- **Separate CUDA stream**: H2D copy overlaps with compute
- **Non-blocking transfer**: `cuda(non_blocking=True)` returns immediately
- **Pipeline**: While GPU trains on batch N, stream loads batch N+1
- **wait_stream()**: Ensures batch N+1 ready before yielding
- **One-hot encoding on GPU**: Fuses transfer + encoding

### PyTorch DataLoader Configuration

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    num_workers=workers,
    worker_init_fn=_worker_init_fn,
    pin_memory=True,
    collate_fn=partial(fast_collate, memory_format),
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=prefetch_factor,
)
```

**NVIDIA's DataLoader best practices:**
- **DistributedSampler**: Shards data across GPUs
- **pin_memory=True**: Enables faster H2D transfers
- **persistent_workers=True**: Keeps workers alive between epochs (faster startup)
- **prefetch_factor**: Number of batches pre-loaded per worker (default 2)
- **drop_last=True**: Ensures all batches same size (important for DDP)
- **fast_collate**: Custom collate function for uint8 → float conversion

### Fast Collate Function

```python
def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets
```

**Why fast_collate:**
- **Defers float conversion**: Keeps data as uint8 until GPU (4× smaller transfers)
- **Memory format control**: Supports NHWC for Tensor Cores
- **Batch allocation**: Single contiguous tensor (cache-friendly)
- **NumPy copy**: `nump_array.copy()` avoids PyTorch warning

---

## Section 4: Production Patterns & Best Practices

### Checkpoint Management

From official NVIDIA code:

```python
class Checkpointer:
    def save_checkpoint(self, checkpoint_state, is_best, filename):
        # Save epoch checkpoint
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, filename))

        # Save best checkpoint
        if is_best:
            shutil.copyfile(
                os.path.join(checkpoint_dir, filename),
                os.path.join(checkpoint_dir, 'model_best.pth.tar')
            )

        # Maintain last N checkpoints
        if keep_last_n > 0:
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth.tar')))
            if len(checkpoints) > keep_last_n:
                for old_checkpoint in checkpoints[:-keep_last_n]:
                    os.remove(old_checkpoint)
```

**NVIDIA's checkpoint strategy:**
- **Epoch checkpoints**: `checkpoint_0001.pth.tar`, `checkpoint_0002.pth.tar`, etc.
- **Best checkpoint**: `model_best.pth.tar` (separate file)
- **Cleanup**: Keep last N checkpoints to save disk space
- **EMA support**: Save both regular and EMA model states

### Command-Line Argument Patterns

```python
parser.add_argument("--arch", "-a", metavar="ARCH", default="resnet50", choices=model_names)
parser.add_argument("-j", "--workers", default=5, type=int, metavar="N")
parser.add_argument("--epochs", default=90, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR")
parser.add_argument("--lr-schedule", default="step", type=str, choices=["step", "linear", "cosine"])
parser.add_argument("--warmup", default=0, type=int, metavar="E")
parser.add_argument("--label-smoothing", default=0.0, type=float, metavar="S")
parser.add_argument("--mixup", default=0.0, type=float, metavar="ALPHA")
parser.add_argument("--optimizer", default="sgd", type=str, choices=("sgd", "rmsprop"))
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W")
parser.add_argument("--amp", action="store_true")
parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"])
```

**Production CLI design:**
- **Sensible defaults**: ResNet-50, 90 epochs, batch 256, LR 0.1, SGD, momentum 0.9
- **Flexible scheduling**: Step, linear, cosine (with warmup support)
- **Regularization**: Label smoothing, mixup, weight decay
- **Optimization**: AMP flag, memory format choice
- **Metavar hints**: Clear parameter units (N, E, LR, S, ALPHA)

### Memory Format: NCHW vs NHWC

```python
memory_format = (
    torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
)
model = model.to(memory_format=memory_format)
```

**When to use NHWC:**
- **Ampere/Hopper GPUs**: Tensor Cores perform better with NHWC
- **ResNet/EfficientNet**: Conv-heavy models benefit most
- **Trade-off**: ~5-10% speedup, but not all ops support NHWC
- **NVIDIA recommendation**: Test both, use NHWC if stable

### Early Stopping Pattern

```python
if early_stopping_patience > 0:
    if not is_best:
        epochs_since_improvement += 1
    else:
        epochs_since_improvement = 0
    if epochs_since_improvement >= early_stopping_patience:
        break
```

**Production early stopping:**
- Count epochs without validation improvement
- Stop if no improvement for N epochs
- Saves compute on non-converging runs

### Profiling Mode

```python
if prof > 0 and (i + 1 >= prof):
    time.sleep(5)  # Let profiler flush
    break
```

**Usage:**
```bash
# Run only 100 iterations for profiling
python main.py --prof 100 /data/imagenet

# Capture with nsys
nsys profile -o profile.qdrep python main.py --prof 100 /data/imagenet
```

### GPU Affinity Optimization

```python
affinity = set_affinity(args.gpu, mode=args.gpu_affinity)
print(f"Training process {args.local_rank} affinity: {affinity}")
```

**Why affinity matters:**
- **NUMA nodes**: Each GPU closer to specific CPU cores
- **Cross-NUMA penalty**: 2× memory access latency
- **NVIDIA pattern**: Bind process to cores near GPU's PCIe root complex
- **Performance impact**: 5-15% throughput improvement on multi-socket systems

### Timeout Handler for Spot Instances

```python
with utils.TimeoutHandler() as timeout_handler:
    for epoch in range(start_epoch, end_epoch):
        # Training code
        if ((i + 1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break
```

**Production pattern for preemptible VMs:**
- Monitor for SIGTERM/SIGINT
- Checkpoint at interruption
- Resume from checkpoint on restart
- Critical for GCP Spot, AWS Spot, Azure Spot

---

## Key Takeaways

### What Makes NVIDIA Examples Official

1. **NGC container tested**: Monthly QA with latest CUDA/cuDNN/NCCL
2. **Performance validated**: Benchmarked on DGX systems (A100, H100)
3. **Best practices embedded**: Years of production experience
4. **Reproducible**: Fixed seeds, deterministic ops, documented results

### Training Script Architecture

```
main.py                 # Entry point, argument parsing
├── prepare_for_training()  # Setup DDP, AMP, dataloaders
├── train_loop()            # Epoch loop, checkpointing
    ├── train()             # Single epoch training
    └── validate()          # Validation loop
```

### Core Optimizations

1. **DDP**: `init_process_group(backend="nccl", init_method="env://")`
2. **AMP**: `GradScaler` with configurable init_scale
3. **DALI**: nvJPEG + GPU augmentation (2-3× faster than PyTorch DataLoader)
4. **Prefetching**: Overlap H2D copy with compute
5. **Memory format**: NHWC for Tensor Cores on Ampere+
6. **Gradient accumulation**: Simulate larger batch sizes
7. **EMA**: Better generalization for free

### When to Use Each Data Backend

- **DALI-GPU**: Maximum throughput, GPU decode (3× faster than PyTorch)
- **DALI-CPU**: CPU decode, still faster than PyTorch (1.5× faster)
- **PyTorch**: Debugging, custom augmentations not in DALI
- **Synthetic**: Benchmarking compute only (no I/O)

### Production Checklist

- [ ] Use `DistributedSampler` for multi-GPU
- [ ] Set `pin_memory=True`, `persistent_workers=True`
- [ ] Enable AMP with `torch.cuda.amp.GradScaler`
- [ ] Add warmup to LR schedule (5-10 epochs)
- [ ] Use gradient accumulation for effective batch > GPU memory
- [ ] Save optimizer state in checkpoints
- [ ] Implement early stopping
- [ ] Monitor gradient scale for AMP stability
- [ ] Use `torch.cuda.synchronize()` for accurate timing
- [ ] Test NHWC memory format on Ampere+ GPUs

---

## Sources

**Primary Source:**
- [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) - Official NVIDIA repository (accessed 2025-11-13)

**Specific Files Analyzed:**
1. [PyTorch/Classification/ConvNets/main.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/main.py) - Main training script (678 lines)
2. [PyTorch/Classification/ConvNets/image_classification/training.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/training.py) - Executor/Trainer classes (435 lines)
3. [PyTorch/Classification/ConvNets/image_classification/dataloaders.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py) - DALI and PyTorch dataloaders (577 lines)

**Additional References:**
- [DeepLearningExamples README](https://github.com/NVIDIA/DeepLearningExamples/blob/master/README.md) - Repository overview
- NGC Deep Learning Containers - Monthly optimized builds
- NVIDIA Developer Documentation - Best practices guides
