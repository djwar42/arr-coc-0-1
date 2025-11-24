# Multi-GPU Distributed Training for SA-1B

## Overview

Training on the SA-1B dataset (11M images, 1.1B masks) requires distributed computing strategies to handle the massive data scale. This knowledge drop covers PyTorch's DistributedDataParallel (DDP), gradient accumulation, multi-node training, and NCCL optimization for efficient large-scale segmentation model training.

**Key Challenge**: SA-1B's scale (10TB+ data, ~100 masks per image) exceeds single-GPU memory and computation capacity, necessitating distributed training across multiple GPUs and nodes.

---

## Section 1: PyTorch DistributedDataParallel (DDP) Fundamentals

### What is DDP?

DistributedDataParallel is PyTorch's recommended approach for distributed training. Unlike DataParallel (DP), DDP:
- Creates one process per GPU (no GIL contention)
- Synchronizes gradients efficiently via all-reduce
- Scales near-linearly across GPUs and nodes

From [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html):
> "DDP is a powerful module in PyTorch that allows you to parallelize your model across multiple machines, making it perfect for large-scale deep learning."

### DDP vs DataParallel Comparison

| Feature | DataParallel (DP) | DistributedDataParallel (DDP) |
|---------|-------------------|-------------------------------|
| Parallelism | Single-process, multi-thread | Multi-process |
| GIL Impact | Significant bottleneck | No GIL issues |
| Scaling | Limited (~4 GPUs) | Linear scaling |
| Multi-node | Not supported | Fully supported |
| Recommended | Legacy only | Production use |

### Basic DDP Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """Initialize distributed process group."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group with NCCL backend for GPUs
    dist.init_process_group(
        backend='nccl',  # NVIDIA's optimized GPU communication
        rank=rank,
        world_size=world_size
    )

    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset, epochs=10):
    """Training function for each DDP process."""
    setup_ddp(rank, world_size)

    # Move model to GPU and wrap with DDP
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoader with distributed sampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,  # Per-GPU batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        # CRITICAL: Set epoch for proper shuffling
        sampler.set_epoch(epoch)

        for batch in dataloader:
            images = batch['image'].to(rank)
            masks = batch['masks'].to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = compute_loss(outputs, masks)
            loss.backward()
            optimizer.step()

    cleanup()

# Launch training
if __name__ == '__main__':
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_ddp,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    )
```

---

## Section 2: DistributedSampler for SA-1B

### Why DistributedSampler?

With SA-1B's 11M images, each GPU must process a unique subset to avoid redundant computation:

```python
from torch.utils.data.distributed import DistributedSampler

class SA1BDistributedSampler(DistributedSampler):
    """Custom sampler for SA-1B with tar-aware distribution."""

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=True):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

    def __iter__(self):
        # Get base indices
        indices = list(super().__iter__())

        # For SA-1B: Ensure tar file locality
        # Group indices by tar file to minimize I/O
        return iter(self._optimize_for_tar_locality(indices))

    def _optimize_for_tar_locality(self, indices):
        """Reorder indices to keep tar file reads sequential."""
        # Group by tar file (each tar has ~11,000 images)
        tar_groups = {}
        for idx in indices:
            tar_id = idx // 11000  # Approximate tar file
            if tar_id not in tar_groups:
                tar_groups[tar_id] = []
            tar_groups[tar_id].append(idx)

        # Flatten while maintaining locality
        optimized = []
        for tar_id in sorted(tar_groups.keys()):
            optimized.extend(tar_groups[tar_id])

        return optimized
```

### Handling Uneven Dataset Splits

From [PyTorch Forums](https://discuss.pytorch.org/t/best-practice-for-uneven-dataset-sizes-with-distributeddataparallel/67308):

```python
# Option 1: Drop last incomplete batch (recommended for training)
sampler = DistributedSampler(dataset, drop_last=True)

# Option 2: Pad to equal sizes (for validation)
sampler = DistributedSampler(dataset, drop_last=False)
# This pads the dataset to be evenly divisible

# SA-1B specific: 11M images / 8 GPUs = 1,375,000 per GPU
# With 1000 tar files, each GPU processes ~125 tar files
```

### Per-GPU Data Distribution for SA-1B

```python
def calculate_sa1b_distribution(world_size):
    """Calculate SA-1B data distribution across GPUs."""
    total_images = 11_000_000
    total_tars = 1000

    images_per_gpu = total_images // world_size
    tars_per_gpu = total_tars // world_size

    print(f"World size: {world_size}")
    print(f"Images per GPU: {images_per_gpu:,}")
    print(f"Tar files per GPU: {tars_per_gpu}")
    print(f"Approx memory per GPU: {images_per_gpu * 350 / 1e9:.1f} GB")

    return images_per_gpu, tars_per_gpu

# Example distributions:
# 8 GPUs:  1,375,000 images/GPU, 125 tars/GPU
# 32 GPUs: 343,750 images/GPU, 31 tars/GPU
# 64 GPUs: 171,875 images/GPU, 15 tars/GPU
```

---

## Section 3: Gradient Accumulation for Large Batch Sizes

### Why Gradient Accumulation?

SA-1B images with 100+ masks consume significant GPU memory. Gradient accumulation simulates larger batches:

From [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation):
> "Gradient accumulation is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory."

### Effective Batch Size Calculation

```
Effective Batch Size = per_gpu_batch × num_gpus × accumulation_steps

Example:
- Per-GPU batch: 2 images
- Number of GPUs: 8
- Accumulation steps: 4
- Effective batch: 2 × 8 × 4 = 64 images
```

### Implementation with DDP

```python
class GradientAccumulationTrainer:
    """Trainer with gradient accumulation for SA-1B."""

    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def train_step(self, batch, rank):
        """Single training step with accumulation."""
        images = batch['image'].to(rank)
        masks = batch['masks'].to(rank)

        # Scale loss by accumulation steps
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = self.model(images)
            loss = compute_segmentation_loss(outputs, masks)
            loss = loss / self.accumulation_steps

        # Backward pass
        loss.backward()

        self.step_count += 1

        # Update weights every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item() * self.accumulation_steps

def train_with_accumulation(rank, world_size, model, dataset):
    """Full training loop with gradient accumulation."""
    setup_ddp(rank, world_size)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Small per-GPU batch due to SA-1B memory requirements
    per_gpu_batch = 2
    accumulation_steps = 4
    effective_batch = per_gpu_batch * world_size * accumulation_steps

    print(f"Rank {rank}: Effective batch size = {effective_batch}")

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=1e-4 * (effective_batch / 256)  # Linear scaling
    )

    trainer = GradientAccumulationTrainer(
        ddp_model,
        optimizer,
        accumulation_steps
    )

    sampler = DistributedSampler(dataset, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = trainer.train_step(batch, rank)

    cleanup()
```

### Gradient Accumulation with no_sync()

For efficiency, skip gradient synchronization during accumulation:

From [PyTorch Forums](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593):

```python
def train_with_no_sync(ddp_model, dataloader, optimizer, accumulation_steps):
    """Efficient gradient accumulation using no_sync context."""
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Use no_sync for all but last accumulation step
        if (i + 1) % accumulation_steps != 0:
            with ddp_model.no_sync():
                loss = compute_loss(ddp_model(batch))
                (loss / accumulation_steps).backward()
        else:
            # Synchronize on last step
            loss = compute_loss(ddp_model(batch))
            (loss / accumulation_steps).backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## Section 4: Multi-Node Training with torchrun and SLURM

### Single-Node Multi-GPU with torchrun

From [PyTorch Multi-GPU Tutorial](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html):

```bash
# Launch on single node with 8 GPUs
torchrun --standalone --nproc_per_node=8 train_sa1b.py

# Equivalent to:
# python -m torch.distributed.launch --nproc_per_node=8 train_sa1b.py
```

### Multi-Node with torchrun

From [PyTorch Multinode Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode.html):

```bash
# Node 0 (master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train_sa1b.py

# Node 1
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train_sa1b.py
```

### SLURM Integration

From [Lambda AI Blog](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide):

```bash
#!/bin/bash
#SBATCH --job-name=sa1b_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))

# Launch with srun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_sa1b.py \
        --data_path /data/sa1b \
        --batch_size 2 \
        --accumulation_steps 4 \
        --epochs 10
```

### Environment Variables for Multi-Node

```python
import os

def get_distributed_info():
    """Get distributed training info from environment."""
    # Set by torchrun/SLURM
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main': rank == 0
    }

def init_distributed():
    """Initialize distributed training from environment."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return True
    return False
```

---

## Section 5: NCCL Backend Optimization

### What is NCCL?

NVIDIA Collective Communications Library (NCCL) is optimized for multi-GPU and multi-node communication:

From [Cerfacs PyTorch Multi-GPU Guide](https://cerfacs.fr/coop/pytorch-multi-gpu):
> "NCCL is optimized for NVIDIA GPUs and ensures better performance and scalability."

### NCCL vs Gloo Backend

| Backend | Use Case | Performance |
|---------|----------|-------------|
| NCCL | Multi-GPU (NVIDIA) | Best for GPUs |
| Gloo | CPU or mixed | Good fallback |
| MPI | HPC clusters | Specialized |

```python
# Always use NCCL for GPU training
dist.init_process_group(backend='nccl')

# Gloo for CPU-only operations
dist.init_process_group(backend='gloo')
```

### NCCL Environment Variables

```bash
# Optimize NCCL for your hardware
export NCCL_DEBUG=INFO                    # Debug output
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2               # GPU Direct RDMA level
export NCCL_P2P_LEVEL=NVL                 # NVLink for P2P
export NCCL_SOCKET_IFNAME=eth0            # Network interface

# For multi-node with InfiniBand
export NCCL_IB_HCA=mlx5_0                 # InfiniBand HCA
export NCCL_IB_GID_INDEX=3                # GID index

# Memory optimization
export NCCL_BUFFSIZE=2097152              # 2MB buffer
```

### NCCL Performance Tuning

```python
def configure_nccl_for_sa1b():
    """Configure NCCL for optimal SA-1B training performance."""
    import os

    # Enable async error handling
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

    # Tune buffer sizes for large gradients
    os.environ['NCCL_BUFFSIZE'] = str(16 * 1024 * 1024)  # 16MB

    # For NVLink-connected GPUs
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'

    # Reduce launch latency
    os.environ['NCCL_LAUNCH_MODE'] = 'GROUP'

# Initialize with timeout for large models
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(minutes=30)
)
```

---

## Section 6: Checkpoint Saving in Distributed Training

### Saving Checkpoints (Main Process Only)

```python
def save_checkpoint(model, optimizer, epoch, loss, rank, save_path):
    """Save checkpoint from main process only."""
    if rank == 0:
        # Get state dict from DDP model
        if isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    # Wait for save to complete before continuing
    dist.barrier()

def load_checkpoint(model, optimizer, checkpoint_path, rank):
    """Load checkpoint in distributed setting."""
    # Load on CPU first to save GPU memory
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model state
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']
```

### Checkpoint Strategies for SA-1B

```python
class SA1BCheckpointManager:
    """Manage checkpoints for long SA-1B training runs."""

    def __init__(self, save_dir, rank, keep_last_n=3):
        self.save_dir = Path(save_dir)
        self.rank = rank
        self.keep_last_n = keep_last_n
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, metrics):
        """Save checkpoint with rotation."""
        if self.rank != 0:
            dist.barrier()
            return

        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': metrics,
        }

        # Save with epoch number
        path = self.save_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, path)

        # Save as latest
        latest_path = self.save_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        dist.barrier()

    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pt'))
        while len(checkpoints) > self.keep_last_n:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]

    def load_latest(self, model, optimizer, scheduler):
        """Load the latest checkpoint."""
        latest_path = self.save_dir / 'checkpoint_latest.pt'
        if latest_path.exists():
            checkpoint = torch.load(latest_path, map_location='cpu')
            model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            return checkpoint['epoch'], checkpoint['metrics']
        return 0, {}
```

---

## Section 7: Fault Tolerance and Elastic Training

### Handling Node Failures

```python
import torch.distributed.elastic as elastic

def train_with_fault_tolerance(rank, world_size):
    """Training with automatic restart on failure."""
    try:
        setup_ddp(rank, world_size)

        # Load from checkpoint if resuming
        start_epoch = load_checkpoint_if_exists()

        for epoch in range(start_epoch, total_epochs):
            train_one_epoch(epoch)

            # Save checkpoint frequently
            if epoch % save_interval == 0:
                save_checkpoint(epoch)

    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        # Checkpoint will be used on restart
        raise
    finally:
        cleanup()
```

### Elastic Training with torchrun

```bash
# Elastic training allows nodes to join/leave
torchrun \
    --nnodes=2:4 \           # Min 2, max 4 nodes
    --nproc_per_node=8 \
    --rdzv_id=sa1b_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 \
    --max_restarts=3 \        # Auto-restart on failure
    train_sa1b.py
```

### Handling Stragglers

```python
def robust_all_reduce(tensor, timeout_seconds=60):
    """All-reduce with timeout for straggler detection."""
    try:
        work = dist.all_reduce(tensor, async_op=True)
        work.wait(timeout=datetime.timedelta(seconds=timeout_seconds))
    except RuntimeError as e:
        if "timeout" in str(e).lower():
            print("Warning: All-reduce timeout, possible straggler")
            # Log for investigation
        raise
```

---

## Section 8: ARR-COC-0-1 Integration - Distributed VLM Training

### Distributed Infrastructure for ARR-COC

ARR-COC's relevance realization training on SA-1B requires distributed computing:

```python
class ARRCOCDistributedTrainer:
    """Distributed trainer for ARR-COC VLM with SA-1B spatial grounding."""

    def __init__(self, config):
        self.config = config
        self.setup_distributed()

    def setup_distributed(self):
        """Initialize distributed training for VLM."""
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(minutes=30)
        )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(self.local_rank)

    def create_dataloaders(self):
        """Create distributed dataloaders for SA-1B + text pairs."""
        # SA-1B dataset with text annotations
        dataset = ARRCOCSpatialGroundingDataset(
            sa1b_path=self.config.sa1b_path,
            text_annotations=self.config.text_path,
            transform=self.get_transforms()
        )

        # Distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        # DataLoader optimized for SA-1B
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.per_gpu_batch,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        return loader, sampler

    def train(self):
        """Main training loop for distributed VLM."""
        # Model setup
        model = ARRCOCVisionLanguageModel(self.config)
        model = model.to(self.local_rank)
        model = DDP(model, device_ids=[self.local_rank])

        # Optimizer with gradient accumulation
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=0.01
        )

        dataloader, sampler = self.create_dataloaders()

        for epoch in range(self.config.epochs):
            sampler.set_epoch(epoch)
            self.train_epoch(model, dataloader, optimizer, epoch)

            if self.rank == 0:
                self.save_checkpoint(model, optimizer, epoch)

    def train_epoch(self, model, dataloader, optimizer, epoch):
        """Train one epoch with gradient accumulation."""
        model.train()
        accumulation_steps = self.config.accumulation_steps

        for i, batch in enumerate(dataloader):
            # Move to GPU
            images = batch['image'].to(self.local_rank)
            masks = batch['masks'].to(self.local_rank)
            text = batch['text']  # Text processed by tokenizer

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images, text)

                # Multi-task loss for spatial grounding
                loss = self.compute_spatial_grounding_loss(
                    outputs, masks, text
                )
                loss = loss / accumulation_steps

            # Backward with gradient accumulation
            if (i + 1) % accumulation_steps != 0:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def compute_spatial_grounding_loss(self, outputs, masks, text):
        """Compute loss for spatial relevance grounding."""
        # Segmentation loss
        seg_loss = F.binary_cross_entropy_with_logits(
            outputs['masks'], masks
        )

        # Text-region alignment loss
        align_loss = self.contrastive_alignment_loss(
            outputs['region_features'],
            outputs['text_features']
        )

        # Combined loss for relevance realization
        return seg_loss + 0.1 * align_loss
```

### Scaling Strategy for ARR-COC

```python
def plan_arrcoc_distributed_training():
    """Plan distributed training resources for ARR-COC on SA-1B."""

    # SA-1B scale
    total_images = 11_000_000
    total_masks = 1_100_000_000

    # Target effective batch size
    target_batch = 256

    # Memory constraints (per GPU)
    gpu_memory_gb = 80  # A100
    image_memory_mb = 50  # Approx with masks

    # Calculate configuration
    per_gpu_batch = 4  # Limited by memory
    num_gpus = 64  # 8 nodes x 8 GPUs
    accumulation = target_batch // (per_gpu_batch * num_gpus)

    # Training time estimate
    steps_per_epoch = total_images // target_batch
    seconds_per_step = 2.0  # Approx for VLM
    hours_per_epoch = (steps_per_epoch * seconds_per_step) / 3600

    print(f"Configuration:")
    print(f"  Per-GPU batch: {per_gpu_batch}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Accumulation steps: {accumulation}")
    print(f"  Effective batch: {per_gpu_batch * num_gpus * accumulation}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Hours per epoch: {hours_per_epoch:.1f}")
```

---

## Complete Training Script Example

```python
#!/usr/bin/env python3
"""
Distributed training script for SA-1B segmentation.

Usage:
    # Single node, 8 GPUs
    torchrun --standalone --nproc_per_node=8 train_distributed.py

    # Multi-node (run on each node)
    torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 \
        --master_addr=192.168.1.1 --master_port=29500 train_distributed.py
"""

import os
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=1)
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(minutes=30)
    )

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    args = parse_args()

    # Setup distributed
    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"Starting distributed training on {world_size} GPUs")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_segmentation_model()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create dataset and dataloader
    dataset = SA1BDataset(args.data_path)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    # Optimizer and scheduler
    effective_batch = args.batch_size * world_size * args.accumulation_steps
    lr = args.lr * (effective_batch / 256)  # Linear scaling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for i, batch in enumerate(dataloader):
            images = batch['image'].to(local_rank)
            masks = batch['masks'].to(local_rank)

            # Forward with mixed precision
            with autocast():
                outputs = model(images)
                loss = compute_loss(outputs, masks)
                loss = loss / args.accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Update weights
            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Save checkpoint
        if is_main and epoch % args.save_interval == 0:
            save_path = f"{args.output_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

---

## Sources

**PyTorch Documentation:**
- [DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Getting Started with DDP
- [Multi-GPU Training](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html) - Single node multi-GPU
- [Multinode Training](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode.html) - Multi-node setup

**Web Research (accessed 2025-11-20):**
- [Lambda AI Multi-Node Guide](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide)
- [Hugging Face Gradient Accumulation](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)
- [LearnOpenCV Distributed Training](https://learnopencv.com/distributed-parallel-training-pytorch-multi-gpu-setup/)
- [Cerfacs PyTorch Multi-GPU](https://cerfacs.fr/coop/pytorch-multi-gpu)
- [VLDB Paper: PyTorch Distributed](https://www.vldb.org/pvldb/vol13/p3005-li.pdf)

**Source Document:**
- PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md

**ARR-COC Integration:**
- Distributed training infrastructure for VLM spatial grounding
- Gradient accumulation for memory-constrained training
- Multi-node scaling for SA-1B's 11M image dataset
