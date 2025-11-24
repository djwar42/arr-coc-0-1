# KNOWLEDGE DROP: Memory Optimization Strategies for SA-1B

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 37)
**File Created**: `sa1b-dataset/36-memory-optimization-strategies.md`

---

## What Was Created

**Knowledge File**: Memory Optimization Strategies (~700 lines)

**8 Sections**:
1. Memory Breakdown Per Sample
2. Gradient Checkpointing
3. Mixed Precision Training (FP16/BF16)
4. Mask Subsampling Strategies
5. Streaming Data Loading
6. Memory Profiling Tools
7. Optimization Patterns for Large Datasets
8. **ARR-COC-0-1** (10%): Memory-efficient VLM training with dense annotations

---

## Key Insights

### Section 1: Memory Breakdown Per Sample

**Understanding SA-1B memory footprint per sample**:

From [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/):

SA-1B contains approximately 100 masks per image on average, creating substantial memory demands:

```python
# Memory breakdown per SA-1B sample
class SA1BMemoryAnalysis:
    """
    Analyze memory footprint of SA-1B samples.
    """

    def calculate_sample_memory(self, image_size=(1500, 2250), num_masks=100):
        """
        Calculate approximate memory per sample.

        Returns:
            dict: Memory breakdown in bytes
        """
        height, width = image_size

        # Image memory (RGB, uint8)
        image_bytes = height * width * 3  # ~10.1 MB

        # Single mask memory (binary, uint8)
        single_mask_bytes = height * width  # ~3.4 MB

        # All masks memory
        all_masks_bytes = single_mask_bytes * num_masks  # ~340 MB

        # Decoded masks as float32 for training
        masks_float32 = all_masks_bytes * 4  # ~1.35 GB

        # Annotations JSON (metadata, bboxes, etc.)
        annotation_bytes = num_masks * 500  # ~50 KB

        return {
            'image_uint8': image_bytes,
            'masks_uint8': all_masks_bytes,
            'masks_float32': masks_float32,
            'annotations': annotation_bytes,
            'total_uint8': image_bytes + all_masks_bytes + annotation_bytes,
            'total_float32': image_bytes + masks_float32 + annotation_bytes
        }

    def print_analysis(self):
        """Print human-readable memory analysis."""
        mem = self.calculate_sample_memory()

        print("SA-1B Memory Breakdown Per Sample")
        print("=" * 50)
        print(f"Image (RGB uint8):     {mem['image_uint8'] / 1e6:.1f} MB")
        print(f"Masks (100x uint8):    {mem['masks_uint8'] / 1e6:.1f} MB")
        print(f"Masks (100x float32):  {mem['masks_float32'] / 1e9:.2f} GB")
        print(f"Annotations:           {mem['annotations'] / 1e3:.1f} KB")
        print("-" * 50)
        print(f"Total (uint8):         {mem['total_uint8'] / 1e6:.1f} MB")
        print(f"Total (float32):       {mem['total_float32'] / 1e9:.2f} GB")

# Output:
# SA-1B Memory Breakdown Per Sample
# ==================================================
# Image (RGB uint8):     10.1 MB
# Masks (100x uint8):    337.5 MB
# Masks (100x float32):  1.35 GB
# Annotations:           50.0 KB
# --------------------------------------------------
# Total (uint8):         347.7 MB
# Total (float32):       1.36 GB
```

**Memory implications for batch training**:

```python
def calculate_batch_memory(batch_size, num_masks_per_sample=100):
    """
    Calculate GPU memory for a batch.

    Considerations:
    - Model weights
    - Activations
    - Gradients
    - Optimizer states
    """
    # Per-sample memory (float32 tensors)
    sample_memory_gb = 1.36

    # Batch data memory
    batch_data_gb = sample_memory_gb * batch_size

    # Model memory (SAM ViT-H example)
    model_weights_gb = 2.5  # 600M parameters * 4 bytes

    # Activations (roughly 2x model size for forward pass)
    activations_gb = model_weights_gb * 2 * batch_size

    # Gradients (same as weights)
    gradients_gb = model_weights_gb

    # Optimizer states (Adam: 2x weights for momentum + variance)
    optimizer_gb = model_weights_gb * 2

    total_gb = (batch_data_gb + model_weights_gb +
                activations_gb + gradients_gb + optimizer_gb)

    return {
        'batch_data': batch_data_gb,
        'model_weights': model_weights_gb,
        'activations': activations_gb,
        'gradients': gradients_gb,
        'optimizer': optimizer_gb,
        'total': total_gb
    }

# Example: batch_size=2
# batch_data: 2.72 GB
# model_weights: 2.5 GB
# activations: 10.0 GB
# gradients: 2.5 GB
# optimizer: 5.0 GB
# total: 22.72 GB
```

### Section 2: Gradient Checkpointing

From [Hugging Face - Methods and tools for efficient training](https://huggingface.co/docs/transformers/v4.47.1/perf_train_gpu_one):

> "While gradient checkpointing may improve memory efficiency, it slows training by approximately 20%."

**Implementation for SA-1B training**:

```python
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class MemoryEfficientSAM(torch.nn.Module):
    """
    SAM with gradient checkpointing for memory efficiency.
    """

    def __init__(self, sam_model, checkpoint_ratio=0.5):
        super().__init__()
        self.sam = sam_model
        self.checkpoint_ratio = checkpoint_ratio

    def forward(self, images, point_coords, point_labels):
        # Checkpoint image encoder (largest memory consumer)
        if self.training:
            image_embeddings = checkpoint(
                self.sam.image_encoder,
                images,
                use_reentrant=False
            )
        else:
            image_embeddings = self.sam.image_encoder(images)

        # Prompt encoder (small, no checkpointing needed)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )

        # Checkpoint mask decoder if needed
        if self.training:
            low_res_masks, iou_predictions = checkpoint(
                self._decode_masks,
                image_embeddings,
                sparse_embeddings,
                dense_embeddings,
                use_reentrant=False
            )
        else:
            low_res_masks, iou_predictions = self._decode_masks(
                image_embeddings, sparse_embeddings, dense_embeddings
            )

        return low_res_masks, iou_predictions

    def _decode_masks(self, image_embeddings, sparse_embeddings, dense_embeddings):
        return self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )

# Usage
def train_with_checkpointing(model, dataloader, optimizer):
    """Train with gradient checkpointing enabled."""

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    for batch in dataloader:
        optimizer.zero_grad()

        images = batch['image'].cuda()
        masks = batch['masks'].cuda()

        # Forward pass with checkpointing
        pred_masks, iou_pred = model(images)

        # Compute loss
        loss = compute_loss(pred_masks, masks, iou_pred)

        # Backward pass (recomputes checkpointed activations)
        loss.backward()

        optimizer.step()
```

**Selective checkpointing for transformer blocks**:

```python
class CheckpointedViT(torch.nn.Module):
    """
    Vision Transformer with selective gradient checkpointing.
    """

    def __init__(self, vit_model, checkpoint_every_n=2):
        super().__init__()
        self.patch_embed = vit_model.patch_embed
        self.blocks = vit_model.blocks
        self.norm = vit_model.norm
        self.checkpoint_every_n = checkpoint_every_n

    def forward(self, x):
        x = self.patch_embed(x)

        for i, block in enumerate(self.blocks):
            if self.training and i % self.checkpoint_every_n == 0:
                # Checkpoint every N blocks
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        return x

# Memory savings calculation
def estimate_checkpoint_savings(num_blocks, checkpoint_every_n):
    """
    Estimate memory savings from checkpointing.

    Without checkpointing: Store all activations
    With checkpointing: Store 1/N activations, recompute others
    """
    # Activations stored
    activations_stored = num_blocks / checkpoint_every_n

    # Memory reduction factor
    reduction = 1 - (activations_stored / num_blocks)

    # Time overhead (recomputation)
    time_overhead = 0.2 * reduction  # ~20% for full checkpointing

    return {
        'memory_reduction': f"{reduction * 100:.0f}%",
        'time_overhead': f"{time_overhead * 100:.0f}%"
    }

# checkpoint_every_n=2: 50% memory reduction, ~10% time overhead
# checkpoint_every_n=4: 75% memory reduction, ~15% time overhead
```

### Section 3: Mixed Precision Training (FP16/BF16)

From [PyTorch - What Every User Should Know About Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/):

> "PyTorch's torch.amp module makes it easy to get started with mixed precision, and we highly recommend using it to train faster and reduce memory usage."

**FP16 vs BF16 comparison**:

From [Reddit Discussion on Mixed Precision](https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/):

> "FP16 has 5 bits for the exponent, meaning it can encode numbers between -65K and +65K. BF16 has 8 bits in exponent like FP32, meaning it can encode the same range."

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """
    Mixed precision training for SA-1B.
    """

    def __init__(self, model, optimizer, precision='fp16'):
        self.model = model
        self.optimizer = optimizer
        self.precision = precision

        # GradScaler only needed for FP16
        if precision == 'fp16':
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Determine dtype
        self.dtype = torch.float16 if precision == 'fp16' else torch.bfloat16

    def train_step(self, batch):
        """
        Single training step with mixed precision.
        """
        images = batch['image'].cuda()
        masks = batch['masks'].cuda()

        self.optimizer.zero_grad()

        # Forward pass in mixed precision
        with autocast(dtype=self.dtype):
            pred_masks, iou_pred = self.model(images)
            loss = self.compute_loss(pred_masks, masks, iou_pred)

        # Backward pass
        if self.scaler:
            # FP16: Scale loss to prevent underflow
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BF16: No scaling needed
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def compute_loss(self, pred_masks, gt_masks, iou_pred):
        """Compute combined segmentation loss."""
        # Dice loss
        dice = self.dice_loss(pred_masks, gt_masks)

        # Focal loss
        focal = self.focal_loss(pred_masks, gt_masks)

        # IoU prediction loss
        with torch.no_grad():
            true_iou = self.calculate_iou(pred_masks, gt_masks)
        iou_loss = torch.nn.functional.mse_loss(iou_pred, true_iou)

        return dice + focal + iou_loss

    @staticmethod
    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    @staticmethod
    def focal_loss(pred, target, alpha=0.8, gamma=2):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

# Memory comparison
def compare_precision_memory():
    """
    Compare memory usage across precisions.
    """
    precisions = {
        'FP32': {'bytes_per_param': 4, 'memory_factor': 1.0},
        'FP16': {'bytes_per_param': 2, 'memory_factor': 0.5},
        'BF16': {'bytes_per_param': 2, 'memory_factor': 0.5}
    }

    # SAM ViT-H: ~600M parameters
    num_params = 600_000_000

    print("Memory Usage Comparison")
    print("=" * 50)
    for name, info in precisions.items():
        memory_gb = (num_params * info['bytes_per_param']) / 1e9
        print(f"{name}: {memory_gb:.2f} GB weights, "
              f"{info['memory_factor']*100:.0f}% of FP32")

# BF16 advantages for SA-1B:
# - Same dynamic range as FP32 (no overflow issues)
# - 50% memory reduction
# - Faster computation on modern GPUs (A100, H100)
```

### Section 4: Mask Subsampling Strategies

**Reducing memory by sampling masks per image**:

```python
import numpy as np
import torch

class MaskSubsampler:
    """
    Subsample masks from SA-1B images to reduce memory.
    """

    def __init__(self, max_masks=32, strategy='diverse'):
        """
        Args:
            max_masks: Maximum masks to keep per image
            strategy: 'random', 'diverse', 'quality', 'size_balanced'
        """
        self.max_masks = max_masks
        self.strategy = strategy

    def subsample(self, masks, annotations):
        """
        Subsample masks based on strategy.

        Args:
            masks: (N, H, W) array of binary masks
            annotations: List of annotation dicts

        Returns:
            Subsampled masks and annotations
        """
        num_masks = len(masks)

        if num_masks <= self.max_masks:
            return masks, annotations

        if self.strategy == 'random':
            indices = self._random_sample(num_masks)
        elif self.strategy == 'diverse':
            indices = self._diverse_sample(masks, annotations)
        elif self.strategy == 'quality':
            indices = self._quality_sample(annotations)
        elif self.strategy == 'size_balanced':
            indices = self._size_balanced_sample(annotations)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return masks[indices], [annotations[i] for i in indices]

    def _random_sample(self, num_masks):
        """Random sampling."""
        return np.random.choice(num_masks, self.max_masks, replace=False)

    def _diverse_sample(self, masks, annotations):
        """
        Sample diverse masks based on size and location.
        Uses k-means clustering on mask properties.
        """
        from sklearn.cluster import KMeans

        # Extract features for each mask
        features = []
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            area = ann['area']

            # Centroid
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2

            # Normalized features
            features.append([
                cx / masks.shape[2],  # Normalized x
                cy / masks.shape[1],  # Normalized y
                np.log(area + 1) / 15,  # Log-normalized area
                bbox[2] / bbox[3] if bbox[3] > 0 else 1  # Aspect ratio
            ])

        features = np.array(features)

        # Cluster masks
        kmeans = KMeans(n_clusters=self.max_masks, random_state=42)
        clusters = kmeans.fit_predict(features)

        # Select one mask from each cluster (highest quality)
        indices = []
        for cluster_id in range(self.max_masks):
            cluster_masks = np.where(clusters == cluster_id)[0]
            if len(cluster_masks) > 0:
                # Select highest quality mask in cluster
                best_idx = max(
                    cluster_masks,
                    key=lambda i: annotations[i].get('predicted_iou', 0)
                )
                indices.append(best_idx)

        return np.array(indices)

    def _quality_sample(self, annotations):
        """Sample masks with highest predicted IoU scores."""
        scores = [ann.get('predicted_iou', 0) for ann in annotations]
        indices = np.argsort(scores)[-self.max_masks:]
        return indices

    def _size_balanced_sample(self, annotations):
        """
        Sample masks with balanced size distribution.
        """
        areas = np.array([ann['area'] for ann in annotations])
        log_areas = np.log(areas + 1)

        # Bin masks by size
        bins = np.linspace(log_areas.min(), log_areas.max(), 5)
        bin_indices = np.digitize(log_areas, bins)

        # Sample from each bin
        indices = []
        masks_per_bin = self.max_masks // 4

        for bin_id in range(1, 5):
            bin_masks = np.where(bin_indices == bin_id)[0]
            if len(bin_masks) > 0:
                n_sample = min(masks_per_bin, len(bin_masks))
                indices.extend(
                    np.random.choice(bin_masks, n_sample, replace=False)
                )

        # Fill remaining slots randomly
        remaining = self.max_masks - len(indices)
        if remaining > 0:
            available = list(set(range(len(annotations))) - set(indices))
            indices.extend(
                np.random.choice(available, min(remaining, len(available)), replace=False)
            )

        return np.array(indices[:self.max_masks])

# Usage in dataset
class MemoryEfficientSA1BDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, max_masks=32, subsample_strategy='diverse'):
        self.root_dir = root_dir
        self.subsampler = MaskSubsampler(max_masks, subsample_strategy)
        # ... initialization

    def __getitem__(self, idx):
        # Load image and annotations
        image = self.load_image(idx)
        masks, annotations = self.load_masks(idx)

        # Subsample masks
        masks, annotations = self.subsampler.subsample(masks, annotations)

        return {
            'image': image,
            'masks': masks,
            'annotations': annotations
        }
```

### Section 5: Streaming Data Loading

**Memory-efficient streaming for SA-1B**:

```python
import torch
from torch.utils.data import IterableDataset
import tarfile
import json
from PIL import Image
import io

class StreamingSA1BDataset(IterableDataset):
    """
    Stream SA-1B data directly from tar files without full extraction.
    Dramatically reduces memory footprint.
    """

    def __init__(
        self,
        tar_files,
        transform=None,
        max_masks=32,
        buffer_size=100
    ):
        self.tar_files = tar_files
        self.transform = transform
        self.max_masks = max_masks
        self.buffer_size = buffer_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            tar_files = self.tar_files
        else:
            # Multi-process: split tar files among workers
            per_worker = len(self.tar_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.tar_files)
            tar_files = self.tar_files[start:end]

        # Stream from tar files
        for tar_path in tar_files:
            yield from self._stream_tar(tar_path)

    def _stream_tar(self, tar_path):
        """
        Stream samples from a single tar file.

        Memory efficient: Only loads one sample at a time.
        """
        buffer = {}

        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if member.isfile():
                    # Extract image ID
                    name = member.name
                    if name.endswith('.jpg'):
                        image_id = name.replace('.jpg', '')
                        buffer[image_id] = {'image_member': member}
                    elif name.endswith('.json'):
                        image_id = name.replace('.json', '')
                        if image_id in buffer:
                            buffer[image_id]['json_member'] = member

                # Process complete samples
                complete = [
                    k for k, v in buffer.items()
                    if 'image_member' in v and 'json_member' in v
                ]

                for image_id in complete:
                    sample = buffer.pop(image_id)

                    # Load image
                    image_file = tar.extractfile(sample['image_member'])
                    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

                    # Load annotations
                    json_file = tar.extractfile(sample['json_member'])
                    annotations = json.load(json_file)

                    # Decode and subsample masks
                    masks = self._decode_masks(annotations, image.size)

                    # Apply transforms
                    if self.transform:
                        image, masks = self.transform(image, masks)

                    yield {
                        'image': image,
                        'masks': masks,
                        'image_id': image_id
                    }

    def _decode_masks(self, annotations, image_size):
        """Decode RLE masks and subsample."""
        from pycocotools import mask as mask_utils

        masks = []
        for ann in annotations['annotations'][:self.max_masks]:
            rle = ann['segmentation']
            mask = mask_utils.decode(rle)
            masks.append(mask)

        if masks:
            return np.stack(masks)
        else:
            return np.zeros((1, image_size[1], image_size[0]), dtype=np.uint8)

# Memory-mapped dataset alternative
class MemmapSA1BDataset(torch.utils.data.Dataset):
    """
    Use memory-mapped files for large mask arrays.
    """

    def __init__(self, index_file, masks_memmap_path):
        with open(index_file) as f:
            self.index = json.load(f)

        # Memory-mapped mask array
        # Shape: (total_masks, H, W), dtype: uint8
        self.masks = np.memmap(
            masks_memmap_path,
            dtype=np.uint8,
            mode='r',
            shape=self._get_memmap_shape()
        )

    def __getitem__(self, idx):
        sample_info = self.index[idx]

        # Load image (small memory footprint)
        image = Image.open(sample_info['image_path']).convert('RGB')

        # Load masks from memory-mapped array (lazy loading)
        mask_start = sample_info['mask_start_idx']
        mask_end = sample_info['mask_end_idx']
        masks = self.masks[mask_start:mask_end]  # Only loads requested data

        return {'image': image, 'masks': masks}
```

### Section 6: Memory Profiling Tools

From [Ohio Supercomputer Center - Estimating and Profiling GPU Memory](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai):

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

class MemoryProfiler:
    """
    Profile GPU memory usage during SA-1B training.
    """

    @staticmethod
    def profile_training_step(model, dataloader, num_steps=10):
        """
        Profile memory usage during training.
        """
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler('./log/profiler')
        ) as prof:

            for i, batch in enumerate(dataloader):
                if i >= num_steps:
                    break

                # Training step
                images = batch['image'].cuda()
                masks = batch['masks'].cuda()

                output = model(images)
                loss = compute_loss(output, masks)
                loss.backward()

                prof.step()

        # Print memory summary
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=20
        ))

    @staticmethod
    def get_memory_stats():
        """Get current GPU memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
                'max_reserved': torch.cuda.max_memory_reserved() / 1e9
            }
        return {}

    @staticmethod
    def memory_snapshot(filename='memory_snapshot.pkl'):
        """
        Record detailed memory snapshot for debugging.
        """
        torch.cuda.memory._record_memory_history()

        # ... run training code ...

        torch.cuda.memory._dump_snapshot(filename)
        torch.cuda.memory._record_memory_history(enabled=None)

    @staticmethod
    def find_memory_leaks(model, dataloader, num_iterations=100):
        """
        Detect memory leaks during training.
        """
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        memory_history = []

        for i, batch in enumerate(dataloader):
            if i >= num_iterations:
                break

            # Training step
            images = batch['image'].cuda()
            with torch.no_grad():
                output = model(images)

            # Clear cache
            del images, output
            torch.cuda.empty_cache()

            # Record memory
            current_memory = torch.cuda.memory_allocated()
            memory_history.append(current_memory)

            if i % 10 == 0:
                print(f"Iter {i}: {current_memory / 1e9:.3f} GB")

        # Check for leak
        final_memory = memory_history[-1]
        leak = final_memory - initial_memory

        if leak > 1e8:  # 100 MB threshold
            print(f"WARNING: Potential memory leak detected: {leak / 1e9:.3f} GB")
        else:
            print(f"No significant memory leak detected")

        return memory_history

# Usage
profiler = MemoryProfiler()
profiler.profile_training_step(model, dataloader)
stats = profiler.get_memory_stats()
print(f"Current GPU memory: {stats['allocated']:.2f} GB allocated, "
      f"{stats['reserved']:.2f} GB reserved")
```

### Section 7: Optimization Patterns for Large Datasets

**Combined optimization strategy**:

```python
class OptimizedSA1BTrainer:
    """
    Combines all memory optimization techniques for SA-1B training.
    """

    def __init__(
        self,
        model,
        config
    ):
        self.config = config

        # 1. Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        self.model = model

        # 2. Set up mixed precision
        self.dtype = torch.bfloat16 if config.use_bf16 else torch.float16
        self.scaler = GradScaler() if not config.use_bf16 else None

        # 3. Configure optimizer with memory efficiency
        self.optimizer = self._create_optimizer()

        # 4. Set up gradient accumulation
        self.accumulation_steps = config.gradient_accumulation_steps

    def _create_optimizer(self):
        """Create memory-efficient optimizer."""
        # Use 8-bit Adam for additional memory savings
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        except ImportError:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

    def create_dataloader(self, dataset):
        """Create memory-optimized dataloader."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            # Custom collate for variable mask counts
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Custom collate that handles variable mask counts."""
        images = torch.stack([item['image'] for item in batch])

        # Pad masks to max in batch
        max_masks = max(item['masks'].shape[0] for item in batch)

        masks = []
        mask_counts = []
        for item in batch:
            m = item['masks']
            mask_counts.append(m.shape[0])
            if m.shape[0] < max_masks:
                padding = torch.zeros(
                    (max_masks - m.shape[0], m.shape[1], m.shape[2]),
                    dtype=m.dtype
                )
                m = torch.cat([m, padding], dim=0)
            masks.append(m)

        return {
            'image': images,
            'masks': torch.stack(masks),
            'mask_counts': mask_counts
        }

    def train_epoch(self, dataloader):
        """Train one epoch with all optimizations."""
        self.model.train()
        total_loss = 0

        for i, batch in enumerate(dataloader):
            # Move to GPU
            images = batch['image'].cuda()
            masks = batch['masks'].cuda()

            # Forward with mixed precision
            with autocast(dtype=self.dtype):
                pred_masks, iou_pred = self.model(images)
                loss = self.compute_loss(pred_masks, masks, iou_pred)
                loss = loss / self.accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (i + 1) % self.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

            # Periodic memory cleanup
            if i % 100 == 0:
                torch.cuda.empty_cache()

        return total_loss / len(dataloader)

# Configuration
config = {
    'batch_size': 2,
    'gradient_accumulation_steps': 8,  # Effective batch size: 16
    'use_bf16': True,
    'learning_rate': 1e-4,
    'num_workers': 4,
    'max_masks_per_image': 32
}
```

### Section 8: ARR-COC-0-1 (10%): Memory-Efficient VLM Training with Dense Annotations

**Relevance to ARR-COC-0-1**:

Memory optimization is critical for ARR-COC-0-1 VLM training with spatial grounding:

```python
class ARRCOCMemoryOptimizedTrainer:
    """
    Memory-optimized VLM training with SA-1B spatial grounding.

    ARR-COC requires:
    - Vision encoder (frozen or fine-tuned)
    - Language model
    - Cross-modal attention
    - Dense spatial annotations (SA-1B masks)

    Memory challenges:
    - Vision: ~2-4GB
    - Language: ~7-14GB (7B parameter model)
    - Cross-modal: ~1-2GB
    - Masks: ~300MB per sample

    Total: 10-20GB+ per sample
    """

    def __init__(self, vision_encoder, language_model, config):
        # Freeze vision encoder (save gradient memory)
        for param in vision_encoder.parameters():
            param.requires_grad = False

        self.vision_encoder = vision_encoder
        self.language_model = language_model

        # Enable gradient checkpointing on language model
        if hasattr(language_model, 'gradient_checkpointing_enable'):
            language_model.gradient_checkpointing_enable()

        # Use 4-bit quantization for language model
        # (from bitsandbytes)
        self.quantize_language_model()

    def quantize_language_model(self):
        """
        Apply 4-bit quantization to reduce memory.

        7B model: 14GB FP16 -> 3.5GB 4-bit
        """
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Apply quantization
        self.language_model = bnb.nn.LinearNF4(self.language_model)

    def process_spatial_grounding(self, images, masks, text_queries):
        """
        Process spatial grounding task with memory efficiency.

        Task: Given image + text query, predict relevant mask
        """
        # Encode images (no gradients needed)
        with torch.no_grad():
            image_features = self.vision_encoder(images)

        # Subsample masks for memory
        subsampled_masks = self.subsample_masks(masks, max_masks=16)

        # Generate mask embeddings
        mask_embeddings = self.encode_masks(subsampled_masks, image_features)

        # Text processing with gradient
        text_embeddings = self.language_model.encode_text(text_queries)

        # Cross-modal matching (gradient flows here)
        similarity = torch.einsum('bmd,btd->bmt', mask_embeddings, text_embeddings)

        return similarity, subsampled_masks

    def training_step(self, batch):
        """
        Memory-efficient training step for spatial grounding.
        """
        images = batch['images'].cuda()
        masks = batch['masks'].cuda()
        text_queries = batch['text']
        target_mask_idx = batch['target_mask_idx']

        # Forward pass with mixed precision
        with autocast(dtype=torch.bfloat16):
            similarity, _ = self.process_spatial_grounding(
                images, masks, text_queries
            )

            # Contrastive loss for mask-text matching
            loss = self.contrastive_loss(similarity, target_mask_idx)

        return loss

# Memory budget for ARR-COC training
def calculate_arrcoc_memory_budget():
    """
    Calculate memory requirements for ARR-COC VLM training.
    """
    components = {
        'vision_encoder_frozen': 2.0,  # GB (no gradients)
        'language_model_4bit': 3.5,    # GB (quantized)
        'lm_gradients': 3.5,           # GB (gradients in fp16)
        'lm_optimizer': 1.0,           # GB (8-bit Adam)
        'cross_modal': 1.0,            # GB
        'masks_subsampled': 0.3,       # GB (16 masks @ float16)
        'activations': 2.0,            # GB (with checkpointing)
    }

    total = sum(components.values())

    print("ARR-COC Memory Budget")
    print("=" * 50)
    for name, gb in components.items():
        print(f"{name:30s}: {gb:.1f} GB")
    print("-" * 50)
    print(f"{'Total':30s}: {total:.1f} GB")

    return total

# Output:
# ARR-COC Memory Budget
# ==================================================
# vision_encoder_frozen         : 2.0 GB
# language_model_4bit           : 3.5 GB
# lm_gradients                  : 3.5 GB
# lm_optimizer                  : 1.0 GB
# cross_modal                   : 1.0 GB
# masks_subsampled              : 0.3 GB
# activations                   : 2.0 GB
# --------------------------------------------------
# Total                         : 13.3 GB

# Fits on single 16GB GPU (V100) or 24GB GPU (A10/3090)
```

---

## Sources

**Web Research:**
- [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Meta AI (accessed 2025-11-20)
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/v4.47.1/perf_train_gpu_one) - Hugging Face (accessed 2025-11-20)
- [What Every User Should Know About Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) - PyTorch Blog (accessed 2025-11-20)
- [Mixed Precision Training: BF16 vs FP16](https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/) - Reddit r/MachineLearning (accessed 2025-11-20)
- [Estimating and Profiling GPU Memory Usage](https://www.osc.edu/resources/getting_started/howto/howto_estimating_and_profiling_gpu_memory_usage_for_generative_ai) - Ohio Supercomputer Center (accessed 2025-11-20)
- [GPU Memory Allocation Training](https://www.reddit.com/r/MachineLearning/comments/1878lat/d_understanding_gpu_memory_allocation_when/) - Reddit r/MachineLearning (accessed 2025-11-20)
- [Automatic Mixed Precision Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) - PyTorch Documentation (accessed 2025-11-20)

**Source Documents:**
- SA-1B Dataset documentation
- SAM paper (Segment Anything, Kirillov et al., 2023)
