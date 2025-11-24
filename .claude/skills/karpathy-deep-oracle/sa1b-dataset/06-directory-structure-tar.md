# SA-1B Directory Structure & Tar Organization

## Overview

SA-1B uses a **highly organized tar-based distribution system** designed for efficient download, storage, and processing of the massive 11M image, 1.1B mask dataset. The dataset is split into **1,000 tar files**, each containing approximately 11,000 images with their annotations.

From [How to effectively download the SA-1B dataset · Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) (accessed 2025-11-20):
- "1000 tar file to download, each seems to be 11Gb"
- "That's 10Tb of data in total, probably 12Tb uncompressed"

From [Problems about loading and managing extreme large datasets like sa1b](https://discuss.huggingface.co/t/problems-about-loading-and-managing-extreme-large-datasets-like-sa1b/46520) (accessed 2025-11-20):
- "10G per tar file, 1000 tar files, 10T in total"
- Each tar file is standalone and self-contained

---

## 1. Hierarchical Directory Structure

### Top-Level Organization

```
SA-1B/
├── sa_000000.tar          # Tar 0 (~10-11 GB compressed)
├── sa_000001.tar          # Tar 1 (~10-11 GB compressed)
├── sa_000002.tar          # Tar 2 (~10-11 GB compressed)
├── ...
├── sa_000500.tar          # Tar 500 (mid-point)
├── ...
├── sa_000998.tar          # Tar 998
└── sa_000999.tar          # Tar 999 (final tar file)
```

**Key characteristics:**
- **Sequential naming:** `sa_000000.tar` to `sa_000999.tar`
- **6-digit zero-padded IDs:** Ensures proper alphabetical sorting
- **1,000 total files:** Evenly distributing 11M images
- **~10TB total:** Compressed tar files (~11GB each)

---

## 2. Individual Tar File Structure

### Internal Directory Layout

Each tar file (e.g., `sa_000000.tar`) contains:

```
sa_000000.tar (extracted) →
└── sa_000000/                    # Directory named same as tar file
    ├── sa_1.jpg                  # Image 1
    ├── sa_1.json                 # Annotations for image 1
    ├── sa_2.jpg                  # Image 2
    ├── sa_2.json                 # Annotations for image 2
    ├── sa_3.jpg                  # Image 3
    ├── sa_3.json                 # Annotations for image 3
    ├── ...
    ├── sa_10999.jpg              # Image ~11,000
    └── sa_10999.json             # Annotations for image ~11,000
```

**Image-annotation pairing:**
- Each `.jpg` image has a matching `.json` annotation file
- Same base filename: `sa_<image_id>.jpg` + `sa_<image_id>.json`
- One JSON file per image (not one JSON for all images in tar)

From [Segment Anything Dataset](https://ai.meta.com/datasets/segment-anything-downloads/) (accessed 2025-11-20):
- "annotations, masks and metadata"
- "approximate size is 10.5GB" per tar file
- Each tar is self-contained with both images and their corresponding JSON annotations

---

## 3. File Naming Conventions

### Tar File Naming

**Pattern:** `sa_NNNNNN.tar`

- **Prefix:** `sa_` (Segment Anything)
- **Index:** 6-digit zero-padded number (`000000` to `000999`)
- **Extension:** `.tar` (uncompressed tar archive)

**Examples:**
- `sa_000000.tar` (first tar)
- `sa_000042.tar` (42nd tar)
- `sa_000500.tar` (500th tar)
- `sa_000999.tar` (last tar)

### Image File Naming

**Pattern:** `sa_<image_id>.jpg`

- **Prefix:** `sa_`
- **Image ID:** Variable-length integer (1 to ~11,000 per tar)
- **Extension:** `.jpg`

**Examples:**
- `sa_1.jpg` (first image in tar)
- `sa_42.jpg` (42nd image)
- `sa_1000.jpg` (1000th image)
- `sa_10999.jpg` (typical last image in tar)

### Annotation File Naming

**Pattern:** `sa_<image_id>.json`

- **Prefix:** `sa_`
- **Image ID:** Matches corresponding image file exactly
- **Extension:** `.json`

**Examples:**
- `sa_1.json` → annotations for `sa_1.jpg`
- `sa_42.json` → annotations for `sa_42.jpg`
- `sa_10999.json` → annotations for `sa_10999.jpg`

---

## 4. Tar File Contents Breakdown

### Approximate Contents Per Tar

| Component | Count | Size |
|-----------|-------|------|
| **Images (.jpg)** | ~11,000 | ~8-9 GB |
| **Annotations (.json)** | ~11,000 | ~1-2 GB |
| **Total files** | ~22,000 | ~10-11 GB compressed |

**Note:** Exact counts vary slightly across tar files due to image distribution.

### File Size Distribution

**Image files:**
- **Format:** JPEG (lossy compression)
- **Average resolution:** 1500×2250 pixels
- **Average size:** ~700 KB - 1 MB per image
- **Total per tar:** ~8-9 GB for ~11,000 images

**Annotation files:**
- **Format:** JSON (text-based)
- **Content:** RLE-encoded masks + metadata
- **Average size:** ~100-200 KB per JSON
- **Total per tar:** ~1-2 GB for ~11,000 annotations

From [SAM_DATASET_SA1B.md](../../../PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md):
- Each tar: "~10 GB compressed"
- Total dataset: "~10 TB uncompressed"
- Images per tar: "~11,000 images"

---

## 5. Standalone Tar Design Philosophy

### Why Standalone Tar Files?

SA-1B uses a **fully standalone tar architecture** where each tar file is:

1. **Self-contained:** All images + annotations in one file
2. **Independently usable:** Can download and use single tar files
3. **Parallel-friendly:** Multiple tar files can be processed simultaneously
4. **Resumable:** Failed downloads only affect one tar, not entire dataset

From [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) (accessed 2025-11-20):
- "Fortunately, each tar file is standalone, with approx 10k image per tar file, with one json file per image"
- This design enables efficient parallel downloading and extraction

### Benefits of Standalone Design

**For downloading:**
- Download subset of tars (e.g., first 10 for experiments)
- Parallel downloads across multiple tar files
- Resume interrupted downloads (only re-download failed tars)

**For storage:**
- Extract only needed tar files (not entire 10TB)
- Delete extracted tar after processing to save space
- Distribute tar files across multiple disks

**For processing:**
- Process tar files independently in parallel
- No dependencies between tar files
- Stream data from single tar without loading entire dataset

---

## 6. Extraction and Storage Patterns

### Full Extraction

```bash
# Extract all 1,000 tar files (requires ~10-12 TB disk space)
for tar_file in sa_*.tar; do
    tar -xf "$tar_file"
done

# Result:
SA-1B/
├── sa_000000/    # ~11,000 images + JSONs
├── sa_000001/    # ~11,000 images + JSONs
├── ...
└── sa_000999/    # ~11,000 images + JSONs
```

**Disk space required:** ~10-12 TB (uncompressed)

### Selective Extraction

```bash
# Extract only first 10 tar files (for experimentation)
for i in $(seq 0 9); do
    tar_name=$(printf "sa_%06d.tar" $i)
    tar -xf "$tar_name"
done

# Result: ~100,000 images, ~100 GB disk space
```

### Extract-Process-Delete Pattern

```bash
# Extract, process, and delete to save disk space
for tar_file in sa_*.tar; do
    tar -xf "$tar_file"               # Extract (~10 GB → ~11 GB)
    python process_dataset.py "$tar_file"  # Process images/annotations
    rm -rf "${tar_file%.tar}"         # Delete extracted directory
    # Keep tar file for future use, or delete it too
done
```

**Disk space required:** Only ~11 GB at a time (vs. 10TB for full extraction)

---

## 7. Directory Traversal Patterns

### Accessing All Images

**Pattern 1: Iterate through tar files**

```python
import os
import glob

# All tar directories
tar_dirs = sorted(glob.glob("sa_*/"))  # ['sa_000000/', 'sa_000001/', ...]

for tar_dir in tar_dirs:
    # All images in this tar
    images = sorted(glob.glob(os.path.join(tar_dir, "*.jpg")))

    for image_path in images:
        # Process image
        process_image(image_path)
```

**Pattern 2: Flat iteration (all images)**

```python
import glob

# All images across all tar directories
all_images = sorted(glob.glob("sa_*/*.jpg"))  # ~11 million files

for image_path in all_images:
    process_image(image_path)
```

### Accessing Image-Annotation Pairs

```python
import os

def get_annotation_path(image_path):
    """Convert image path to annotation path."""
    return image_path.replace(".jpg", ".json")

# Iterate through images and load corresponding annotations
for image_path in all_images:
    annotation_path = get_annotation_path(image_path)

    image = load_image(image_path)
    annotations = load_json(annotation_path)

    # Process pair
    process_pair(image, annotations)
```

---

## 8. ARR-COC-0-1: Efficient Tar-Based Data Pipeline for Relevance Realization (10%)

### Spatial Grounding with Tar Streaming

For **ARR-COC-0-1 training on SA-1B**, the tar-based structure enables efficient spatial relevance realization:

**Stream-based loading:**
```python
# Process tar files sequentially without full extraction
import tarfile

for tar_idx in range(1000):  # sa_000000 to sa_000999
    tar_path = f"sa_{tar_idx:06d}.tar"

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar:
            if member.name.endswith('.jpg'):
                # Extract image in-memory
                image_data = tar.extractfile(member).read()

                # Load corresponding JSON (next member)
                json_member = tar.getmember(member.name.replace('.jpg', '.json'))
                annotation_data = tar.extractfile(json_member).read()

                # Train spatial grounding for relevance realization
                train_spatial_relevance(image_data, annotation_data)
```

**Benefits for ARR-COC:**
- **Memory efficient:** Stream from tar without extracting all 10TB
- **Subset training:** Use first 50 tars (~500K images) for initial experiments
- **Parallel training:** Different GPUs process different tar files
- **Scalable:** Incrementally add more tar files as training progresses

**Relevance realization use case:**
- Train model to identify **salient spatial regions** (high-relevance masks)
- Use mask granularity (100+ masks/image) for multi-scale relevance
- SA-1B's class-agnostic masks enable **pure spatial attention** learning
- Tar-based organization allows **progressive curriculum:** start with subset, scale up

From [SAM_DATASET_SA1B.md](../../../PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md):
- "1,000 tar files total"
- "Each tar: ~10 GB compressed"
- "~11,000 images per tar"

This structure perfectly suits **ARR-COC's incremental training approach**: start small (10 tars = 110K images), validate spatial grounding, then scale to full 1,000 tars.

---

## Sources

**Source Documents:**
- [SAM_DATASET_SA1B.md](../../../PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md) - Lines 94-123 (Directory Layout, Tar File Organization)

**Web Research:**
- [How to effectively download the SA-1B dataset · Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - GitHub, accessed 2025-11-20 (1000 tar files, 11GB each, standalone design)
- [Problems about loading and managing extreme large datasets like sa1b](https://discuss.huggingface.co/t/problems-about-loading-and-managing-extreme-large-datasets-like-sa1b/46520) - Hugging Face Forums, accessed 2025-11-20 (10G per tar, 1000 tars, 10T total)
- [Segment Anything Dataset Downloads](https://ai.meta.com/datasets/segment-anything-downloads/) - Meta AI, accessed 2025-11-20 (approximate size 10.5GB per tar)

**Additional References:**
- [segment_anything | TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/segment_anything) - TensorFlow documentation (dataset structure)
- [erow/SA-1B: download and load SA-1B](https://github.com/erow/SA-1B) - GitHub (SA-1B downloader implementation)
