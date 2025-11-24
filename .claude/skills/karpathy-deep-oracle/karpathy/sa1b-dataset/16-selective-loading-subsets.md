# Selective Loading & Dataset Subsets

## Overview

Loading the full SA-1B (11M images, 1.1B masks) is often unnecessary. **Selective loading** enables quick experiments, category-specific training, and quality filtering.

**Strategies:**
- Load specific tar ranges (e.g., first 10% of dataset)
- Filter by mask quality (predicted_iou > 0.9)
- Sample by category/domain (if metadata available)
- Random sampling for diversity

## Subset Selection Strategies

**1. Fixed ranges (first N tars):**
```python
# Load first 10 tars = ~110k images
tar_indices = range(0, 10)
urls = [f"https://dl.fbaipublicfiles.com/segment_anything/sa_{i:06d}.tar"
        for i in tar_indices]
```

**2. Random sampling:**
```python
import random
# Sample 50 random tars across full distribution
all_indices = list(range(1000))
random_indices = random.sample(all_indices, 50)
```

**3. Quality filtering:**
```python
def load_high_quality_masks(annotations, iou_threshold=0.9):
    """Filter masks by predicted IoU."""
    high_quality = [
        ann for ann in annotations['annotations']
        if ann['predicted_iou'] >= iou_threshold
    ]
    return high_quality
```

## PyTorch Dataset with Subsets

```python
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class SA1BSubset(Dataset):
    """
    SA-1B dataset with flexible subset selection.
    """

    def __init__(
        self,
        data_dir: str,
        tar_indices: list = None,
        quality_threshold: float = 0.0,
        max_samples: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.quality_threshold = quality_threshold

        # Collect all image/annotation pairs from specified tars
        self.samples = []

        if tar_indices is None:
            tar_indices = range(1000)  # All tars

        for tar_idx in tar_indices:
            tar_dir = self.data_dir / f"sa_{tar_idx:06d}"
            if not tar_dir.exists():
                continue

            # Load all annotations from this tar
            for json_file in tar_dir.glob("*.json"):
                with open(json_file) as f:
                    data = json.load(f)

                # Filter by quality
                for ann in data['annotations']:
                    if ann['predicted_iou'] >= self.quality_threshold:
                        image_file = json_file.with_suffix('.jpg')
                        self.samples.append({
                            'image': image_file,
                            'annotation': ann,
                        })

                # Early stop if max_samples reached
                if max_samples and len(self.samples) >= max_samples:
                    break

            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load image and mask...
        return image, mask

# Usage examples
# 1. Quick experiment: 10 tars, high quality only
dataset = SA1BSubset(
    data_dir="./sa1b_extracted",
    tar_indices=range(10),
    quality_threshold=0.9,
    max_samples=50000
)

# 2. Full training: All tars, all masks
dataset = SA1BSubset(
    data_dir="./sa1b_extracted",
    quality_threshold=0.0
)
```

## Command-line Subset Tools

**Extract specific tars only:**
```bash
# Extract tars 0-9 (10% of dataset)
for i in {0..9}; do
  tar -xf sa_$(printf "%06d" $i).tar
done
```

**Count samples in subset:**
```bash
# Count images in first 10 tars
find ./sa1b_extracted/sa_00000* -name "*.jpg" | wc -l
```

## ARR-COC Subset Recommendations

**Prototyping (quick iteration):**
- 10 tars = ~110k images
- Quality filter: IoU > 0.9
- ~30-50k high-quality masks
- Training time: ~2-4 hours on 1 GPU

**Production training:**
- 100-500 tars = 1-5M images
- Quality filter: IoU > 0.85
- ~500k-2M masks
- Training time: 1-3 days on 8 GPUs

**Sources**: PyTorch Dataset API, ML engineering best practices
