# KNOWLEDGE DROP: SA-1B Partial Download & Subsets

**Date**: 2025-11-20
**Runner**: PART 16 (Partial Download Subsets)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers strategies for downloading subsets of SA-1B for experimentation, including selection strategies, representative sampling, and subset configurations for different use cases.

---

## 1. Why Download Subsets?

### Use Cases

| Scenario | Recommended Subset | Size | Time (100 Mbps) |
|----------|-------------------|------|-----------------|
| Quick testing | 1-3 tars | ~10-30 GB | 15-45 min |
| Development | 10 tars | ~100 GB | 2-3 hours |
| Small experiments | 50 tars | ~500 GB | 12 hours |
| Medium experiments | 100 tars | ~1 TB | 1 day |
| Large experiments | 500 tars | ~5 TB | 4-5 days |
| Full dataset | 1000 tars | ~10 TB | 9+ days |

### Benefits of Subsets

1. **Faster iteration**: Test code before full download
2. **Storage constraints**: Limited disk space
3. **Network costs**: Pay-per-GB cloud transfers
4. **Initial validation**: Verify pipeline works
5. **Incremental expansion**: Start small, grow as needed

---

## 2. Subset Selection Strategies

### Random Sampling

```python
"""
Random subset selection for representative sampling.
"""

import random
from typing import List, Tuple


def select_random_subset(
    total_files: int,
    subset_size: int,
    seed: int = 42
) -> List[int]:
    """
    Select random subset of file indices.

    Args:
        total_files: Total number of files (1000 for SA-1B)
        subset_size: Number of files to select
        seed: Random seed for reproducibility

    Returns:
        Sorted list of selected indices
    """
    random.seed(seed)
    indices = random.sample(range(total_files), subset_size)
    return sorted(indices)


def generate_subset_links(
    links_file: str,
    indices: List[int],
    output_file: str
):
    """
    Generate links file for subset.

    Args:
        links_file: Full links file
        indices: Selected indices
        output_file: Output subset links file
    """
    # Load all links
    all_links = []
    with open(links_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                all_links.append(line.strip())

    # Select subset
    subset_links = [all_links[i] for i in indices if i < len(all_links)]

    # Write subset file
    with open(output_file, 'w') as f:
        f.write('\n'.join(subset_links))

    print(f"Created subset with {len(subset_links)} files: {output_file}")


# Example: 100-file random subset
if __name__ == "__main__":
    indices = select_random_subset(1000, 100, seed=42)
    generate_subset_links('sa1b_links.txt', indices, 'sa1b_subset_100.txt')
```

### Sequential Selection

```python
"""
Sequential subset selection for contiguous ranges.
"""

def select_sequential_subset(
    start: int,
    end: int,
    total_files: int = 1000
) -> List[int]:
    """
    Select sequential range of files.

    Args:
        start: Starting index (inclusive)
        end: Ending index (exclusive)
        total_files: Total files available

    Returns:
        List of indices
    """
    end = min(end, total_files)
    return list(range(start, end))


def create_sequential_links(
    start: int,
    count: int,
    output_file: str
):
    """
    Create links file for sequential subset.

    Args:
        start: Starting index
        count: Number of files
        output_file: Output file path
    """
    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"

    with open(output_file, 'w') as f:
        for i in range(start, start + count):
            if i >= 1000:
                break
            filename = f"sa_{i:06d}.tar"
            url = f"{BASE_URL}/{filename}"
            f.write(f"{filename}\t{url}\n")

    print(f"Created sequential subset: {output_file}")


# Example subsets
if __name__ == "__main__":
    # First 10 files (for testing)
    create_sequential_links(0, 10, 'sa1b_first_10.txt')

    # Middle 100 files
    create_sequential_links(450, 100, 'sa1b_middle_100.txt')

    # Last 50 files
    create_sequential_links(950, 50, 'sa1b_last_50.txt')
```

### Stratified Sampling

```python
"""
Stratified sampling for diverse representation.
"""

def select_stratified_subset(
    total_files: int,
    subset_size: int,
    num_strata: int = 10
) -> List[int]:
    """
    Select stratified subset (evenly distributed across dataset).

    Args:
        total_files: Total number of files
        subset_size: Desired subset size
        num_strata: Number of strata to sample from

    Returns:
        List of selected indices
    """
    indices = []
    files_per_stratum = subset_size // num_strata
    stratum_size = total_files // num_strata

    for stratum in range(num_strata):
        start = stratum * stratum_size
        end = start + stratum_size

        # Select evenly from each stratum
        step = stratum_size // files_per_stratum
        for i in range(files_per_stratum):
            idx = start + (i * step)
            if idx < total_files:
                indices.append(idx)

    # Add remaining if needed
    while len(indices) < subset_size and len(indices) < total_files:
        idx = len(indices)
        if idx not in indices:
            indices.append(idx)

    return sorted(indices[:subset_size])


# Example: 100 files stratified across 10 regions
if __name__ == "__main__":
    indices = select_stratified_subset(1000, 100, num_strata=10)
    print(f"Stratified indices: {indices[:20]}...")  # Show first 20
```

---

## 3. Predefined Subset Configurations

### Starter Subsets

```python
"""
Predefined subset configurations for common use cases.
"""

SUBSET_CONFIGS = {
    # Quick testing
    'tiny': {
        'size': 3,
        'indices': [0, 500, 999],  # First, middle, last
        'description': 'Minimal test set (3 files, ~30 GB)'
    },

    # Development
    'small': {
        'size': 10,
        'indices': list(range(0, 100, 10)),  # Every 10th from first 100
        'description': 'Development set (10 files, ~100 GB)'
    },

    # Experimentation
    'medium': {
        'size': 50,
        'indices': list(range(0, 1000, 20)),  # Every 20th
        'description': 'Small experiment set (50 files, ~500 GB)'
    },

    # Serious experiments
    'large': {
        'size': 100,
        'indices': list(range(0, 1000, 10)),  # Every 10th
        'description': 'Large experiment set (100 files, ~1 TB)'
    },

    # Half dataset
    'half': {
        'size': 500,
        'indices': list(range(0, 1000, 2)),  # Every 2nd
        'description': 'Half dataset (500 files, ~5 TB)'
    },

    # First N (contiguous)
    'first_10': {
        'size': 10,
        'indices': list(range(10)),
        'description': 'First 10 files (contiguous)'
    },

    'first_100': {
        'size': 100,
        'indices': list(range(100)),
        'description': 'First 100 files (contiguous)'
    }
}


def get_subset_config(name: str) -> dict:
    """
    Get predefined subset configuration.

    Args:
        name: Configuration name

    Returns:
        Configuration dictionary
    """
    if name not in SUBSET_CONFIGS:
        available = ', '.join(SUBSET_CONFIGS.keys())
        raise ValueError(f"Unknown subset: {name}. Available: {available}")

    return SUBSET_CONFIGS[name]


def create_subset_from_config(
    config_name: str,
    output_file: str
):
    """
    Create links file from predefined configuration.

    Args:
        config_name: Configuration name
        output_file: Output file path
    """
    config = get_subset_config(config_name)
    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"

    with open(output_file, 'w') as f:
        for idx in config['indices']:
            filename = f"sa_{idx:06d}.tar"
            url = f"{BASE_URL}/{filename}"
            f.write(f"{filename}\t{url}\n")

    print(f"Created '{config_name}' subset: {config['description']}")
    print(f"  Files: {config['size']}")
    print(f"  Output: {output_file}")


# Example usage
if __name__ == "__main__":
    # List available configurations
    print("Available subset configurations:")
    for name, config in SUBSET_CONFIGS.items():
        print(f"  {name}: {config['description']}")

    # Create development subset
    create_subset_from_config('small', 'sa1b_dev.txt')
```

---

## 4. Subset Download Scripts

### Bash Script for Subsets

```bash
#!/bin/bash
# download_subset.sh - Download SA-1B subset

set -e

# Configuration
SUBSET_TYPE="${1:-small}"
OUTPUT_DIR="${2:-./sa1b_subset}"
WORKERS="${3:-4}"

# Define subsets
case "$SUBSET_TYPE" in
    "tiny")
        START=0; END=3
        ;;
    "small")
        START=0; END=10
        ;;
    "medium")
        START=0; END=50
        ;;
    "large")
        START=0; END=100
        ;;
    "custom")
        START="${4:-0}"; END="${5:-10}"
        ;;
    *)
        echo "Unknown subset type: $SUBSET_TYPE"
        echo "Available: tiny, small, medium, large, custom"
        exit 1
        ;;
esac

echo "Downloading SA-1B subset: $SUBSET_TYPE"
echo "  Range: $START to $END"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate links for subset
LINKS_FILE="$OUTPUT_DIR/subset_links.txt"
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"

for i in $(seq $START $((END - 1))); do
    printf "sa_%06d.tar\t${BASE_URL}/sa_%06d.tar\n" $i $i >> "$LINKS_FILE"
done

echo "Created links file with $((END - START)) entries"

# Download using aria2c if available
if command -v aria2c &> /dev/null; then
    aria2c \
        --input-file="$LINKS_FILE" \
        --dir="$OUTPUT_DIR" \
        --max-concurrent-downloads="$WORKERS" \
        --max-connection-per-server=8 \
        --split=8 \
        --continue=true
else
    # Fallback to wget
    while IFS=$'\t' read -r filename url; do
        wget -c -P "$OUTPUT_DIR" "$url" &

        # Limit parallel downloads
        while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
            sleep 1
        done
    done < "$LINKS_FILE"
    wait
fi

echo "Download complete!"
```

### Python Subset Downloader

```python
#!/usr/bin/env python3
"""
SA-1B subset downloader with progress tracking.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm


def download_file(url: str, output_path: str) -> bool:
    """Download single file with resume support."""
    temp_path = output_path + '.partial'

    try:
        # Check for partial download
        start_byte = 0
        if os.path.exists(temp_path):
            start_byte = os.path.getsize(temp_path)

        # Check if already complete
        if os.path.exists(output_path):
            return True

        headers = {}
        if start_byte > 0:
            headers['Range'] = f'bytes={start_byte}-'

        response = requests.get(url, headers=headers, stream=True, timeout=600)
        response.raise_for_status()

        mode = 'ab' if start_byte > 0 else 'wb'
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

        os.rename(temp_path, output_path)
        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_subset(
    start: int,
    end: int,
    output_dir: str,
    workers: int = 4
):
    """
    Download subset of SA-1B.

    Args:
        start: Starting index
        end: Ending index
        output_dir: Output directory
        workers: Number of parallel workers
    """
    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"

    os.makedirs(output_dir, exist_ok=True)

    # Generate download list
    downloads = []
    for i in range(start, min(end, 1000)):
        filename = f"sa_{i:06d}.tar"
        url = f"{BASE_URL}/{filename}"
        output_path = os.path.join(output_dir, filename)

        # Skip if exists
        if os.path.exists(output_path):
            continue

        downloads.append((url, output_path))

    if not downloads:
        print("All files already downloaded!")
        return

    print(f"Downloading {len(downloads)} files...")

    # Download in parallel
    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, url, path): (url, path)
            for url, path in downloads
        }

        with tqdm(total=len(downloads), desc="Progress") as pbar:
            for future in as_completed(futures):
                url, path = futures[future]
                try:
                    if future.result():
                        completed += 1
                    else:
                        failed.append(path)
                except Exception:
                    failed.append(path)
                pbar.update(1)

    print(f"\nComplete: {completed}/{len(downloads)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {os.path.basename(f)}")


def main():
    parser = argparse.ArgumentParser(description='Download SA-1B subset')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=10, help='End index')
    parser.add_argument('--output', type=str, default='./sa1b_subset',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Parallel workers')

    # Preset options
    parser.add_argument('--preset', type=str, choices=['tiny', 'small', 'medium', 'large'],
                       help='Use preset configuration')

    args = parser.parse_args()

    # Apply presets
    if args.preset:
        presets = {
            'tiny': (0, 3),
            'small': (0, 10),
            'medium': (0, 50),
            'large': (0, 100)
        }
        args.start, args.end = presets[args.preset]

    print(f"SA-1B Subset Downloader")
    print(f"  Range: {args.start} to {args.end}")
    print(f"  Files: {args.end - args.start}")
    print(f"  Workers: {args.workers}")
    print()

    download_subset(args.start, args.end, args.output, args.workers)


if __name__ == "__main__":
    main()
```

---

## 5. Representative Sampling Analysis

### Image Diversity Check

```python
"""
Analyze subset for representativeness.
"""

import os
import json
from collections import Counter


def analyze_subset(data_dir: str) -> dict:
    """
    Analyze subset for diversity metrics.

    Args:
        data_dir: Directory with extracted data

    Returns:
        Analysis results
    """
    stats = {
        'total_images': 0,
        'total_masks': 0,
        'masks_per_image': [],
        'image_sizes': [],
    }

    # Find all JSON files (mask annotations)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)

                with open(json_path, 'r') as f:
                    data = json.load(f)

                stats['total_images'] += 1
                num_masks = len(data.get('annotations', []))
                stats['total_masks'] += num_masks
                stats['masks_per_image'].append(num_masks)

                # Get image dimensions
                image_info = data.get('image', {})
                if 'height' in image_info and 'width' in image_info:
                    area = image_info['height'] * image_info['width']
                    stats['image_sizes'].append(area)

    # Calculate statistics
    if stats['masks_per_image']:
        masks = stats['masks_per_image']
        stats['avg_masks_per_image'] = sum(masks) / len(masks)
        stats['min_masks'] = min(masks)
        stats['max_masks'] = max(masks)

    if stats['image_sizes']:
        sizes = stats['image_sizes']
        stats['avg_image_size'] = sum(sizes) / len(sizes)
        stats['min_image_size'] = min(sizes)
        stats['max_image_size'] = max(sizes)

    return stats


def compare_to_full_dataset(subset_stats: dict) -> dict:
    """
    Compare subset statistics to full dataset.

    Full SA-1B statistics:
    - 11M images
    - 1.1B masks
    - ~100 masks/image average
    """
    full_stats = {
        'total_images': 11_000_000,
        'total_masks': 1_100_000_000,
        'avg_masks_per_image': 100
    }

    comparison = {
        'subset_fraction': subset_stats['total_images'] / full_stats['total_images'],
        'mask_density_ratio': (
            subset_stats.get('avg_masks_per_image', 0) /
            full_stats['avg_masks_per_image']
        )
    }

    return comparison
```

---

## 6. Incremental Download Strategy

```python
"""
Incremental download strategy for growing datasets.
"""

import os
import json


class IncrementalDownloader:
    """
    Incrementally download and expand SA-1B subset.
    """

    def __init__(self, data_dir: str):
        """
        Initialize incremental downloader.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = data_dir
        self.state_file = os.path.join(data_dir, 'download_state.json')
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load download state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'downloaded_indices': [],
            'target_size': 0
        }

    def _save_state(self):
        """Save download state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_next_batch(self, batch_size: int) -> list:
        """
        Get next batch of indices to download.

        Args:
            batch_size: Number of files in batch

        Returns:
            List of indices to download
        """
        downloaded = set(self.state['downloaded_indices'])
        next_batch = []

        for i in range(1000):
            if i not in downloaded:
                next_batch.append(i)
                if len(next_batch) >= batch_size:
                    break

        return next_batch

    def mark_downloaded(self, indices: list):
        """Mark indices as downloaded."""
        self.state['downloaded_indices'].extend(indices)
        self.state['downloaded_indices'] = sorted(
            set(self.state['downloaded_indices'])
        )
        self._save_state()

    def get_progress(self) -> dict:
        """Get download progress."""
        return {
            'downloaded': len(self.state['downloaded_indices']),
            'total': 1000,
            'percent': len(self.state['downloaded_indices']) / 10
        }


# Usage example
if __name__ == "__main__":
    downloader = IncrementalDownloader('./sa1b_data')

    # Download in batches of 10
    while True:
        batch = downloader.get_next_batch(10)
        if not batch:
            print("Download complete!")
            break

        print(f"Downloading batch: {batch}")
        # ... download files ...

        downloader.mark_downloaded(batch)

        progress = downloader.get_progress()
        print(f"Progress: {progress['percent']:.1f}%")

        # Check if we have enough for now
        if progress['downloaded'] >= 100:
            print("Reached target size (100 files)")
            break
```

---

## 7. Subset Quality Validation

```python
"""
Validate subset quality and completeness.
"""

import os
import tarfile


def validate_subset(
    data_dir: str,
    expected_files: int
) -> dict:
    """
    Validate downloaded subset.

    Args:
        data_dir: Directory with tar files
        expected_files: Expected number of files

    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'total_files': 0,
        'valid_files': 0,
        'corrupt_files': [],
        'missing_files': [],
        'total_size_gb': 0
    }

    # Check for expected files
    for i in range(expected_files):
        filename = f"sa_{i:06d}.tar"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            results['missing_files'].append(filename)
            results['valid'] = False
            continue

        results['total_files'] += 1
        file_size = os.path.getsize(filepath)
        results['total_size_gb'] += file_size / (1024**3)

        # Validate tar integrity
        try:
            with tarfile.open(filepath, 'r') as tar:
                # Check that we can read member list
                members = tar.getmembers()
                if len(members) == 0:
                    raise ValueError("Empty tar file")
                results['valid_files'] += 1
        except Exception as e:
            results['corrupt_files'].append((filename, str(e)))
            results['valid'] = False

    return results


# Example usage
if __name__ == "__main__":
    results = validate_subset('./sa1b_subset', expected_files=10)

    print("Subset Validation Results")
    print("=" * 40)
    print(f"Valid: {results['valid']}")
    print(f"Files: {results['valid_files']}/{results['total_files']}")
    print(f"Size: {results['total_size_gb']:.1f} GB")

    if results['missing_files']:
        print(f"\nMissing files: {len(results['missing_files'])}")

    if results['corrupt_files']:
        print(f"\nCorrupt files: {len(results['corrupt_files'])}")
        for filename, error in results['corrupt_files']:
            print(f"  {filename}: {error}")
```

---

## Sources

**GitHub Resources:**
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official repository
- [erow/SA-1B](https://github.com/erow/SA-1B) - Download scripts (accessed 2025-11-20)

**Community Discussions:**
- [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - Download strategies
- [Hugging Face Forums](https://discuss.huggingface.co/t/problems-about-loading-and-managing-extreme-large-datasets-like-sa1b/46520) - Dataset management

**Research:**
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - arXiv:2304.02643 (Kirillov et al., 2023)

---

## Next Steps

- PART 17: Tar extraction strategies
- PART 18: Verification and integrity checking
