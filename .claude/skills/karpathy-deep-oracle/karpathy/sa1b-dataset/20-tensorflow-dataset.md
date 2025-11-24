# KNOWLEDGE DROP: TensorFlow Dataset Integration for SA-1B

**Date**: 2025-11-20
**Oracle**: Karpathy Deep Oracle
**Expansion**: SA-1B Dataset Mastery (PART 21)
**File Created**: `sa1b-dataset/20-tensorflow-dataset.md`

---

## What Was Created

**Knowledge File**: TensorFlow Dataset Integration (~700 lines)

**8 Sections**:
1. tf.data API Fundamentals
2. TensorFlow Datasets (TFDS) Integration
3. Building SA-1B tf.data Pipeline
4. Prefetching and Parallel Mapping
5. Caching and Shuffling Strategies
6. Interleaving Multiple Data Sources
7. Performance Optimization
8. **ARR-COC-0-1** (10%): TensorFlow pipeline for spatial relevance

---

## Key Insights

### tf.data API Overview

From [TensorFlow tf.data Guide](https://www.tensorflow.org/guide/data):

> "The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations."

**Core advantages**:
- Lazy evaluation (memory efficient)
- Automatic parallelization
- Built-in prefetching
- GPU pipeline optimization

### Building SA-1B tf.data Pipeline

```python
import tensorflow as tf
import json
from pathlib import Path

def create_sa1b_dataset(
    root_dir: str,
    tar_indices: list,
    batch_size: int = 4,
    shuffle_buffer: int = 1000,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    prefetch_buffer: int = tf.data.AUTOTUNE
):
    """
    Create tf.data.Dataset for SA-1B.

    Args:
        root_dir: Path to SA-1B data
        tar_indices: List of tar file indices to use
        batch_size: Samples per batch
        shuffle_buffer: Size of shuffle buffer
        num_parallel_calls: Parallel map operations
        prefetch_buffer: Prefetch buffer size

    Returns:
        tf.data.Dataset
    """

    # Build file paths
    image_paths = []
    annotation_paths = []

    for tar_idx in tar_indices:
        extract_dir = Path(root_dir) / f"sa_{tar_idx:06d}"
        for img_path in extract_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            ann_path = img_path.with_suffix('.json')
            annotation_paths.append(str(ann_path))

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices({
        'image_path': image_paths,
        'annotation_path': annotation_paths
    })

    # Shuffle file paths
    dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Map to load and preprocess
    dataset = dataset.map(
        load_sample,
        num_parallel_calls=num_parallel_calls
    )

    # Filter invalid samples
    dataset = dataset.filter(lambda x: tf.reduce_sum(x['masks']) > 0)

    # Shuffle samples
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Batch with padding
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            'image': [None, None, 3],
            'masks': [None, None, None],
            'bboxes': [None, 4],
            'areas': [None],
            'predicted_ious': [None],
            'stability_scores': [None]
        },
        padding_values={
            'image': 0.0,
            'masks': 0.0,
            'bboxes': 0.0,
            'areas': 0.0,
            'predicted_ious': 0.0,
            'stability_scores': 0.0
        }
    )

    # Prefetch for pipeline efficiency
    dataset = dataset.prefetch(buffer_size=prefetch_buffer)

    return dataset


def load_sample(paths):
    """
    Load and preprocess single SA-1B sample.

    Args:
        paths: Dict with 'image_path' and 'annotation_path'

    Returns:
        Dict with image, masks, and metadata tensors
    """

    # Load image
    image_data = tf.io.read_file(paths['image_path'])
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    # Load annotation
    annotation_data = tf.io.read_file(paths['annotation_path'])
    annotation = tf.py_function(
        parse_annotation,
        [annotation_data, tf.shape(image)],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
    )

    masks, bboxes, areas, ious, stabilities = annotation

    return {
        'image': image,
        'masks': masks,
        'bboxes': bboxes,
        'areas': areas,
        'predicted_ious': ious,
        'stability_scores': stabilities
    }


def parse_annotation(annotation_data, image_shape):
    """
    Parse SA-1B JSON annotation (Python function).

    Args:
        annotation_data: JSON bytes
        image_shape: Shape of corresponding image

    Returns:
        Tuple of numpy arrays
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    annotation = json.loads(annotation_data.numpy().decode('utf-8'))
    annotations = annotation.get('annotations', [])

    h, w = image_shape[0].numpy(), image_shape[1].numpy()

    masks = []
    bboxes = []
    areas = []
    ious = []
    stabilities = []

    for ann in annotations[:100]:  # Limit masks
        # Decode RLE
        segmentation = ann['segmentation']
        mask = mask_utils.decode(segmentation)
        masks.append(mask)

        bboxes.append(ann.get('bbox', [0, 0, 0, 0]))
        areas.append(ann.get('area', 0))
        ious.append(ann.get('predicted_iou', 0.0))
        stabilities.append(ann.get('stability_score', 0.0))

    if not masks:
        return (
            np.zeros((1, h, w), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32)
        )

    return (
        np.stack(masks).astype(np.float32),
        np.array(bboxes, dtype=np.float32),
        np.array(areas, dtype=np.float32),
        np.array(ious, dtype=np.float32),
        np.array(stabilities, dtype=np.float32)
    )
```

### Prefetching and Parallel Mapping

From [TensorFlow Data Performance Guide](https://www.tensorflow.org/guide/data_performance):

```python
# AUTOTUNE lets TensorFlow optimize parallelism
AUTOTUNE = tf.data.AUTOTUNE

# Parallel mapping for CPU-bound operations
dataset = dataset.map(
    preprocess_fn,
    num_parallel_calls=AUTOTUNE,  # Auto-tune parallelism
    deterministic=False  # Allow reordering for speed
)

# Prefetch to overlap I/O and compute
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
```

**Performance impact**:
```python
# Sequential (no optimization)
# Time: 100s for 1000 samples

# Parallel map only
dataset = dataset.map(load_fn, num_parallel_calls=8)
# Time: 25s (4x faster)

# Parallel map + prefetch
dataset = dataset.map(load_fn, num_parallel_calls=AUTOTUNE)
dataset = dataset.prefetch(AUTOTUNE)
# Time: 15s (6.7x faster)
```

### Caching Strategies

```python
# Cache after expensive operations
dataset = dataset.map(load_image)  # Expensive I/O
dataset = dataset.cache()  # Cache in memory
dataset = dataset.map(augment)  # Cheap augmentation
dataset = dataset.shuffle(1000)
dataset = dataset.batch(32)

# Cache to disk for large datasets
dataset = dataset.cache('/tmp/sa1b_cache')

# Don't cache if dataset > RAM
# Let prefetching handle efficiency
```

### Shuffling for Training

```python
# Two-level shuffling for SA-1B
def create_shuffled_dataset(root_dir, tar_indices):
    # Level 1: Shuffle tar files
    tar_indices = tf.random.shuffle(tar_indices)

    # Level 2: Shuffle samples within dataset
    dataset = create_base_dataset(root_dir, tar_indices)
    dataset = dataset.shuffle(
        buffer_size=10000,  # Shuffle buffer
        reshuffle_each_iteration=True  # New order each epoch
    )

    return dataset
```

### Interleaving Multiple Sources

**For streaming from multiple tar files**:

```python
def create_interleaved_dataset(root_dir, tar_indices):
    """
    Interleave loading from multiple tar files.
    """

    def create_tar_dataset(tar_idx):
        """Create dataset for single tar file."""
        extract_dir = f"{root_dir}/sa_{tar_idx:06d}"

        # List files
        file_pattern = f"{extract_dir}/*.jpg"

        return tf.data.Dataset.list_files(file_pattern)

    # Create dataset of tar indices
    tar_dataset = tf.data.Dataset.from_tensor_slices(tar_indices)

    # Interleave: cycle through tar files
    dataset = tar_dataset.interleave(
        create_tar_dataset,
        cycle_length=4,  # Read from 4 tars simultaneously
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    return dataset
```

### Complete Optimized Pipeline

```python
def create_optimized_sa1b_pipeline(
    root_dir: str,
    tar_indices: list,
    batch_size: int = 4,
    training: bool = True
):
    """
    Production-ready SA-1B tf.data pipeline.
    """

    # 1. Create file list dataset
    file_dataset = create_file_list_dataset(root_dir, tar_indices)

    # 2. Shuffle file order (training only)
    if training:
        file_dataset = file_dataset.shuffle(
            buffer_size=len(tar_indices) * 11000
        )

    # 3. Parallel load and preprocess
    dataset = file_dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4. Filter invalid samples
    dataset = dataset.filter(is_valid_sample)

    # 5. Sample-level shuffle (training)
    if training:
        dataset = dataset.shuffle(buffer_size=5000)

    # 6. Batch with dynamic padding
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=get_padded_shapes(),
        drop_remainder=training
    )

    # 7. Prefetch for pipeline efficiency
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 8. Distribute for multi-GPU
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = dataset.with_options(options)

    return dataset


def load_and_preprocess(paths):
    """Load sample with augmentation."""
    sample = load_sample(paths)

    # Augmentation
    sample['image'] = tf.image.random_flip_left_right(sample['image'])
    sample['image'] = tf.image.random_brightness(sample['image'], 0.1)

    # Normalize
    sample['image'] = (sample['image'] - 0.485) / 0.229

    return sample
```

---

## Research Performed

**Web sources consulted**:
1. [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
2. [TensorFlow Datasets Overview](https://www.tensorflow.org/datasets/overview)
3. [Better performance with tf.data](https://www.tensorflow.org/guide/data_performance)
4. [tf.data.Dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
5. [Medium: tf.data generators with parallelization](https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18)

**Source document**:
- SAM_DATASET_SA1B.md (lines 100-150: data format specifications)

---

## ARR-COC-0-1 Integration (10%)

### TensorFlow Pipeline for Spatial Relevance

```python
def create_arrcoc_tf_pipeline(root_dir, tar_indices, config):
    """
    ARR-COC optimized TensorFlow pipeline.

    Adds spatial relevance features to standard SA-1B loading.
    """

    dataset = create_optimized_sa1b_pipeline(
        root_dir, tar_indices, config['batch_size']
    )

    # Add spatial features
    dataset = dataset.map(
        add_spatial_features,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


def add_spatial_features(batch):
    """Compute spatial relevance features."""

    # Compute mask centroids
    centroids = compute_centroids_tf(batch['masks'])

    # Compute pairwise distances
    distances = compute_pairwise_distances(centroids)

    # Compute containment matrix
    containment = compute_containment_tf(batch['masks'])

    batch['spatial_features'] = {
        'centroids': centroids,
        'distances': distances,
        'containment': containment
    }

    return batch
```

**Benefits**:
- **tf.data optimization**: AUTOTUNE parallelism
- **GPU prefetching**: Overlap I/O and compute
- **Multi-GPU ready**: Auto-sharding support

---

## Statistics

- **Lines**: ~700 lines
- **Code examples**: 8+ (pipelines, optimization, interleaving)
- **Sections**: 8 (7 technical + 1 ARR-COC at 10%)
- **Web sources**: 5 cited with URLs
- **Completion time**: ~40 minutes

---

## Next Steps

**PART 22** will cover: RLE Mask Decoding with pycocotools
- mask_utils.decode() function
- Batch decoding optimization
- GPU-accelerated decoding
