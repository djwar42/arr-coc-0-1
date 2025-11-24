# KNOWLEDGE DROP: SA-1B Python Dataset Class Implementation

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 19)
**File Created**: `sa1b-dataset/18-python-dataset-class.md`

---

## What Was Created

**Knowledge File**: Python Dataset Class Implementation (~700 lines)

**8 Sections**:
1. PyTorch Dataset Class Fundamentals
2. SA-1B Dataset Class Structure
3. `__init__` Method: Loading and Indexing
4. `__getitem__` Method: Sample Retrieval
5. `__len__` Method: Dataset Size
6. Lazy Loading and Memory Efficiency
7. Caching and Memory Mapping Strategies
8. **ARR-COC-0-1** (10%): Custom dataset for spatial relevance training

---

## Key Insights

### PyTorch Dataset Class Requirements

**Two required methods**:
```python
from torch.utils.data import Dataset

class SA1BDataset(Dataset):
    def __len__(self):
        """Return total number of samples"""
        return self.num_samples

    def __getitem__(self, idx):
        """Load and return sample at index idx"""
        return image, masks, metadata
```

From [PyTorch Datasets & DataLoaders Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html):
- `__getitem__` loads and returns a sample from the dataset at the given index
- `__len__` returns the size of the dataset
- These methods are called by DataLoader during training

### Complete SA-1B Dataset Implementation

```python
import os
import json
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils

class SA1BDataset(Dataset):
    """
    PyTorch Dataset for SA-1B (Segment Anything 1 Billion masks).

    Supports:
    - Direct file access (extracted tar files)
    - Streaming from tar files (memory efficient)
    - Lazy loading with caching
    - Memory mapping for large datasets
    """

    def __init__(
        self,
        root_dir: str,
        tar_indices: list = None,
        transform=None,
        mask_transform=None,
        max_masks: int = 100,
        use_tar_streaming: bool = False,
        cache_size: int = 1000,
        preload_index: bool = True
    ):
        """
        Initialize SA-1B Dataset.

        Args:
            root_dir: Path to SA-1B data directory
            tar_indices: List of tar file indices to use (0-999)
            transform: Transforms for images
            mask_transform: Transforms for masks
            max_masks: Maximum masks per image to load
            use_tar_streaming: Stream from tar without extraction
            cache_size: Number of samples to cache in memory
            preload_index: Build sample index on initialization
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_masks = max_masks
        self.use_tar_streaming = use_tar_streaming
        self.cache_size = cache_size

        # Sample index: list of (tar_idx, image_id) tuples
        self.samples = []

        # LRU cache for loaded samples
        self._cache = {}
        self._cache_order = []

        # Determine which tar files to use
        if tar_indices is None:
            # Find all available tar files
            tar_indices = self._find_tar_indices()
        self.tar_indices = tar_indices

        # Build sample index
        if preload_index:
            self._build_index()

    def _find_tar_indices(self) -> list:
        """Find all available tar file indices in root_dir."""
        indices = []
        for i in range(1000):
            tar_path = self.root_dir / f"sa_{i:06d}.tar"
            if tar_path.exists():
                indices.append(i)
        return indices

    def _build_index(self):
        """Build index of all samples across tar files."""
        print(f"Building index for {len(self.tar_indices)} tar files...")

        for tar_idx in self.tar_indices:
            if self.use_tar_streaming:
                # Index from tar file directly
                tar_path = self.root_dir / f"sa_{tar_idx:06d}.tar"
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('.jpg'):
                            # Extract image ID from filename
                            image_id = Path(member.name).stem
                            self.samples.append((tar_idx, image_id))
            else:
                # Index from extracted directory
                extract_dir = self.root_dir / f"sa_{tar_idx:06d}"
                if extract_dir.exists():
                    for img_path in extract_dir.glob("*.jpg"):
                        image_id = img_path.stem
                        self.samples.append((tar_idx, image_id))

        print(f"Indexed {len(self.samples)} samples")

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return sample at index idx.

        Returns:
            dict with keys:
                - 'image': torch.Tensor (C, H, W)
                - 'masks': torch.Tensor (N, H, W)
                - 'bboxes': torch.Tensor (N, 4)
                - 'areas': torch.Tensor (N,)
                - 'predicted_ious': torch.Tensor (N,)
                - 'stability_scores': torch.Tensor (N,)
                - 'image_id': str
        """
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        tar_idx, image_id = self.samples[idx]

        # Load image and annotations
        if self.use_tar_streaming:
            image, annotation = self._load_from_tar(tar_idx, image_id)
        else:
            image, annotation = self._load_from_disk(tar_idx, image_id)

        # Process masks
        masks, bboxes, areas, ious, stabilities = self._process_annotation(
            annotation, image.size
        )

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.mask_transform:
            masks = self.mask_transform(masks)

        sample = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            'areas': areas,
            'predicted_ious': ious,
            'stability_scores': stabilities,
            'image_id': image_id
        }

        # Update cache
        self._update_cache(idx, sample)

        return sample

    def _load_from_disk(self, tar_idx: int, image_id: str):
        """Load image and annotation from extracted files."""
        extract_dir = self.root_dir / f"sa_{tar_idx:06d}"

        img_path = extract_dir / f"{image_id}.jpg"
        ann_path = extract_dir / f"{image_id}.json"

        image = Image.open(img_path).convert('RGB')
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        return image, annotation

    def _load_from_tar(self, tar_idx: int, image_id: str):
        """Load image and annotation directly from tar file."""
        tar_path = self.root_dir / f"sa_{tar_idx:06d}.tar"

        with tarfile.open(tar_path, 'r') as tar:
            # Load image
            img_member = tar.getmember(f"sa_{tar_idx:06d}/{image_id}.jpg")
            img_file = tar.extractfile(img_member)
            image = Image.open(img_file).convert('RGB')

            # Load annotation
            ann_member = tar.getmember(f"sa_{tar_idx:06d}/{image_id}.json")
            ann_file = tar.extractfile(ann_member)
            annotation = json.load(ann_file)

        return image, annotation

    def _process_annotation(self, annotation: dict, image_size: tuple):
        """
        Process SA-1B annotation to extract masks and metadata.

        Args:
            annotation: SA-1B JSON annotation
            image_size: (width, height) of image

        Returns:
            masks, bboxes, areas, predicted_ious, stability_scores
        """
        annotations = annotation.get('annotations', [])

        # Limit number of masks
        if len(annotations) > self.max_masks:
            # Sort by area (descending) and take top max_masks
            annotations = sorted(
                annotations,
                key=lambda x: x.get('area', 0),
                reverse=True
            )[:self.max_masks]

        masks = []
        bboxes = []
        areas = []
        ious = []
        stabilities = []

        for ann in annotations:
            # Decode RLE mask
            segmentation = ann['segmentation']
            if isinstance(segmentation, dict):
                # COCO RLE format
                mask = mask_utils.decode(segmentation)
            else:
                # Already decoded
                mask = np.array(segmentation)

            masks.append(mask)
            bboxes.append(ann.get('bbox', [0, 0, 0, 0]))
            areas.append(ann.get('area', 0))
            ious.append(ann.get('predicted_iou', 0.0))
            stabilities.append(ann.get('stability_score', 0.0))

        # Handle empty annotations
        if not masks:
            h, w = image_size[1], image_size[0]
            return (
                torch.zeros(1, h, w),
                torch.zeros(1, 4),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1)
            )

        # Convert to tensors
        masks = torch.from_numpy(np.stack(masks)).float()
        bboxes = torch.tensor(bboxes).float()
        areas = torch.tensor(areas).float()
        ious = torch.tensor(ious).float()
        stabilities = torch.tensor(stabilities).float()

        return masks, bboxes, areas, ious, stabilities

    def _update_cache(self, idx: int, sample: dict):
        """Update LRU cache with new sample."""
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return

        # Add to cache
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[idx] = sample
        self._cache_order.append(idx)

    def get_image_info(self, idx: int) -> dict:
        """Get metadata for image without loading full sample."""
        tar_idx, image_id = self.samples[idx]
        return {
            'tar_index': tar_idx,
            'image_id': image_id,
            'tar_file': f"sa_{tar_idx:06d}.tar"
        }


# Extended dataset with memory mapping for very large scale
class SA1BMemoryMappedDataset(SA1BDataset):
    """
    Memory-mapped version for datasets too large to index in RAM.

    Uses numpy memmap for sample index and metadata.
    """

    def __init__(self, *args, memmap_path: str = None, **kwargs):
        self.memmap_path = memmap_path
        super().__init__(*args, **kwargs)

    def _build_index(self):
        """Build memory-mapped index."""
        if self.memmap_path and os.path.exists(self.memmap_path):
            # Load existing memmap
            self._load_memmap()
        else:
            # Build and save memmap
            super()._build_index()
            if self.memmap_path:
                self._save_memmap()

    def _save_memmap(self):
        """Save sample index to memory-mapped file."""
        # Convert samples to numpy array
        data = np.array(
            [(tar_idx, int(img_id.replace('sa_', '')))
             for tar_idx, img_id in self.samples],
            dtype=[('tar_idx', 'i4'), ('image_id', 'i8')]
        )

        # Create memory-mapped file
        mm = np.memmap(
            self.memmap_path,
            dtype=data.dtype,
            mode='w+',
            shape=data.shape
        )
        mm[:] = data
        mm.flush()
        print(f"Saved memmap index to {self.memmap_path}")

    def _load_memmap(self):
        """Load sample index from memory-mapped file."""
        mm = np.memmap(
            self.memmap_path,
            dtype=[('tar_idx', 'i4'), ('image_id', 'i8')],
            mode='r'
        )
        self.samples = [
            (int(row['tar_idx']), f"sa_{row['image_id']}")
            for row in mm
        ]
        print(f"Loaded {len(self.samples)} samples from memmap")
```

---

## Lazy Loading Strategies

### Why Lazy Loading for SA-1B?

**Scale challenges**:
- 11M images = ~10TB data
- 1.1B masks = massive memory requirement
- Loading all at once = impossible

**Lazy loading benefits**:
- Load samples only when accessed
- Memory usage = O(batch_size), not O(dataset_size)
- Enables training on consumer hardware

### Implementation Patterns

```python
class LazyLoadingSA1BDataset(Dataset):
    """
    Ultra-lazy: Don't even build full index upfront.
    """

    def __init__(self, root_dir, tar_indices):
        self.root_dir = Path(root_dir)
        self.tar_indices = tar_indices

        # Only count samples, don't enumerate
        self._count_samples()

        # Index built on-demand per tar file
        self._tar_indices_cache = {}

    def _count_samples(self):
        """Count samples without full enumeration."""
        self.samples_per_tar = {}
        self.total_samples = 0

        for tar_idx in self.tar_indices:
            # Read manifest or estimate from tar size
            tar_path = self.root_dir / f"sa_{tar_idx:06d}.tar"
            # Approximately 11,000 images per tar
            count = 11000  # Or read from manifest
            self.samples_per_tar[tar_idx] = count
            self.total_samples += count

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which tar file contains this index
        tar_idx, local_idx = self._global_to_local(idx)

        # Load tar index if not cached
        if tar_idx not in self._tar_indices_cache:
            self._index_tar(tar_idx)

        image_id = self._tar_indices_cache[tar_idx][local_idx]
        return self._load_sample(tar_idx, image_id)

    def _global_to_local(self, global_idx):
        """Convert global index to (tar_idx, local_idx)."""
        cumsum = 0
        for tar_idx in self.tar_indices:
            count = self.samples_per_tar[tar_idx]
            if global_idx < cumsum + count:
                return tar_idx, global_idx - cumsum
            cumsum += count
        raise IndexError(f"Index {global_idx} out of range")
```

---

## Caching Strategies

### LRU Cache for Frequently Accessed Samples

```python
from collections import OrderedDict

class CachedDataset(Dataset):
    def __init__(self, *args, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = OrderedDict()
        self.cache_size = cache_size

    def __getitem__(self, idx):
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # Load sample
        sample = self._load_sample(idx)

        # Add to cache
        self._cache[idx] = sample

        # Evict if over capacity
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return sample
```

### Disk-Based Caching

```python
import pickle
import hashlib

class DiskCachedDataset(Dataset):
    def __init__(self, *args, cache_dir='/tmp/sa1b_cache', **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, idx):
        """Generate cache file path for sample."""
        return self.cache_dir / f"sample_{idx}.pkl"

    def __getitem__(self, idx):
        cache_path = self._cache_path(idx)

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        sample = self._load_sample(idx)

        # Cache to disk
        with open(cache_path, 'wb') as f:
            pickle.dump(sample, f)

        return sample
```

---

## Research Performed

**Web sources consulted**:
1. [PyTorch Datasets & DataLoaders Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) - Official documentation
2. [torch.utils.data.DataLoader](https://docs.pytorch.org/docs/stable/data.html) - API reference
3. [Writing Custom Datasets](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) - Custom dataset guide
4. [StackOverflow: __getitem__ idx behavior](https://stackoverflow.com/questions/58834338) - Index handling
5. [PyTorch Discuss: Custom Dataset patterns](https://discuss.pytorch.org/t/custom-dataset-getitem-is-receiving-a-list-from-dataloader/217592)

**Source document**:
- SAM_DATASET_SA1B.md (lines 150-200: JSON structure, file organization)

---

## ARR-COC-0-1 Integration (10%)

### Custom Dataset for Spatial Relevance Training

**Training pipeline**:
```python
class ARRCOCDataset(SA1BDataset):
    """
    Extended SA-1B dataset for ARR-COC spatial relevance training.

    Adds:
    - Multi-scale mask selection
    - Spatial relationship encoding
    - Relevance score computation
    """

    def __init__(self, *args, relevance_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.relevance_config = relevance_config or {}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Add spatial relevance features
        sample['spatial_features'] = self._compute_spatial_features(
            sample['masks'],
            sample['bboxes']
        )

        # Multi-scale mask hierarchy
        sample['mask_hierarchy'] = self._build_mask_hierarchy(
            sample['masks'],
            sample['areas']
        )

        return sample

    def _compute_spatial_features(self, masks, bboxes):
        """Compute spatial relationships between masks."""
        n_masks = masks.shape[0]

        # Pairwise IoU matrix
        iou_matrix = torch.zeros(n_masks, n_masks)
        for i in range(n_masks):
            for j in range(n_masks):
                intersection = (masks[i] * masks[j]).sum()
                union = ((masks[i] + masks[j]) > 0).sum()
                iou_matrix[i, j] = intersection / (union + 1e-6)

        # Containment relationships
        containment = torch.zeros(n_masks, n_masks)
        for i in range(n_masks):
            for j in range(n_masks):
                if i != j:
                    overlap = (masks[i] * masks[j]).sum()
                    containment[i, j] = overlap / (masks[j].sum() + 1e-6)

        return {
            'iou_matrix': iou_matrix,
            'containment': containment,
            'centroids': self._compute_centroids(masks)
        }

    def _build_mask_hierarchy(self, masks, areas):
        """Build hierarchical mask structure by scale."""
        # Sort by area
        sorted_indices = torch.argsort(areas, descending=True)

        # Group into scale levels
        n_masks = len(masks)
        levels = {
            'large': sorted_indices[:n_masks//3],
            'medium': sorted_indices[n_masks//3:2*n_masks//3],
            'small': sorted_indices[2*n_masks//3:]
        }

        return levels
```

**Benefits for relevance realization**:
- **100+ masks/image**: Rich spatial supervision
- **Multi-granularity**: Learn scale-aware attention
- **Class-agnostic**: Pure spatial relevance without semantic bias
- **Hierarchical structure**: Natural curriculum from coarse to fine

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: Complete dataset class + extensions
- **Sections**: 8 (7 technical + 1 ARR-COC integration at 10%)
- **Web sources**: 5 cited with URLs
- **Source document**: 1 cited with line numbers
- **Completion time**: ~45 minutes

---

## Next Steps

**PART 20** will cover: PyTorch DataLoader Integration
- num_workers optimization
- prefetch_factor tuning
- pin_memory for GPU transfer
- Custom collate_fn for variable masks
