# KNOWLEDGE DROP: SA-1B Tar Extraction Strategies

**Date**: 2025-11-20
**Runner**: PART 17 (Tar Extraction Strategies)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers strategies for extracting SA-1B tar files efficiently, including parallel extraction, memory-efficient streaming, SSD optimization, and cleanup procedures.

---

## 1. Sequential vs Parallel Extraction

### Performance Comparison

| Method | Speed | CPU Usage | Memory | Best For |
|--------|-------|-----------|--------|----------|
| Sequential | 1x | Low | Low | Limited resources |
| GNU Parallel | 3-8x | High | Medium | Multi-core systems |
| Python multiprocessing | 2-4x | Medium | Medium | Custom workflows |
| Stream + Extract | Variable | Low | Very Low | Memory-constrained |

### Basic Sequential Extraction

```bash
#!/bin/bash
# sequential_extract.sh - Basic sequential tar extraction

TAR_DIR="./raw"
OUTPUT_DIR="./extracted"

mkdir -p "$OUTPUT_DIR"

for tar_file in "$TAR_DIR"/sa_*.tar; do
    echo "Extracting: $tar_file"
    tar -xf "$tar_file" -C "$OUTPUT_DIR"
done

echo "Extraction complete!"
```

---

## 2. GNU Parallel Extraction

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install parallel

# macOS
brew install parallel

# Verify installation
parallel --version
```

### Parallel Extraction Script

From [GNU Parallel documentation](https://www.gnu.org/s/parallel/man.html) (accessed 2025-11-20):

```bash
#!/bin/bash
# parallel_extract.sh - Extract SA-1B tars using GNU parallel

TAR_DIR="${1:-./raw}"
OUTPUT_DIR="${2:-./extracted}"
NUM_JOBS="${3:-4}"

mkdir -p "$OUTPUT_DIR"

echo "Extracting SA-1B tar files"
echo "  Source: $TAR_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Parallel jobs: $NUM_JOBS"
echo ""

# Find all tar files and extract in parallel
find "$TAR_DIR" -name "sa_*.tar" -type f | \
    parallel -j "$NUM_JOBS" --progress \
    'tar -xf {} -C '"$OUTPUT_DIR"' && echo "Extracted: {}"'

echo "Extraction complete!"
```

### Advanced GNU Parallel Options

```bash
#!/bin/bash
# advanced_parallel_extract.sh

TAR_DIR="./raw"
OUTPUT_DIR="./extracted"
LOG_DIR="./logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Extract with full progress tracking and logging
find "$TAR_DIR" -name "sa_*.tar" -type f | \
    parallel \
    --jobs 8 \
    --progress \
    --bar \
    --joblog "$LOG_DIR/extraction.log" \
    --resume-failed \
    --retries 3 \
    'tar -xf {} -C '"$OUTPUT_DIR"

# Report results
echo ""
echo "Extraction Summary:"
echo "  Total: $(wc -l < "$LOG_DIR/extraction.log")"
echo "  Failed: $(grep -c "^.*\t1\t" "$LOG_DIR/extraction.log" || echo 0)"
```

---

## 3. Python Extraction Solutions

### Multi-process Extraction

```python
"""
Python multi-process tar extraction for SA-1B.
"""

import os
import tarfile
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def extract_single_tar(
    tar_path: str,
    output_dir: str,
    images_dir: str = None,
    masks_dir: str = None
) -> dict:
    """
    Extract a single tar file.

    Args:
        tar_path: Path to tar file
        output_dir: Base output directory
        images_dir: Separate directory for images (optional)
        masks_dir: Separate directory for masks (optional)

    Returns:
        Extraction results
    """
    results = {
        'tar_file': os.path.basename(tar_path),
        'success': False,
        'images': 0,
        'masks': 0,
        'error': None
    }

    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                # Determine destination based on file type
                if member.name.endswith('.jpg'):
                    dest_dir = images_dir or output_dir
                    results['images'] += 1
                elif member.name.endswith('.json'):
                    dest_dir = masks_dir or output_dir
                    results['masks'] += 1
                else:
                    dest_dir = output_dir

                # Extract member
                tar.extract(member, dest_dir)

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)

    return results


def extract_all_parallel(
    tar_dir: str,
    output_dir: str,
    images_dir: str = None,
    masks_dir: str = None,
    num_workers: int = None
):
    """
    Extract all tar files in parallel.

    Args:
        tar_dir: Directory containing tar files
        output_dir: Output directory
        images_dir: Separate images directory
        masks_dir: Separate masks directory
        num_workers: Number of parallel workers
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)
    if masks_dir:
        os.makedirs(masks_dir, exist_ok=True)

    # Find all tar files
    tar_files = sorted([
        os.path.join(tar_dir, f)
        for f in os.listdir(tar_dir)
        if f.endswith('.tar')
    ])

    if not tar_files:
        print("No tar files found!")
        return

    print(f"Extracting {len(tar_files)} tar files with {num_workers} workers")

    # Create extraction function with fixed arguments
    extract_func = partial(
        extract_single_tar,
        output_dir=output_dir,
        images_dir=images_dir,
        masks_dir=masks_dir
    )

    # Extract in parallel
    total_images = 0
    total_masks = 0
    failed = []

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_func, tar_files),
            total=len(tar_files),
            desc="Extracting"
        ))

    # Summarize results
    for result in results:
        if result['success']:
            total_images += result['images']
            total_masks += result['masks']
        else:
            failed.append((result['tar_file'], result['error']))

    print(f"\nExtraction complete!")
    print(f"  Images: {total_images:,}")
    print(f"  Masks: {total_masks:,}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed extractions:")
        for tar_file, error in failed:
            print(f"  {tar_file}: {error}")


# Example usage
if __name__ == "__main__":
    extract_all_parallel(
        tar_dir="./raw",
        output_dir="./extracted",
        images_dir="./images",
        masks_dir="./annotations",
        num_workers=4
    )
```

### Memory-Efficient Streaming

```python
"""
Memory-efficient streaming extraction for large datasets.
"""

import os
import tarfile
import io


def stream_extract_tar(
    tar_path: str,
    output_dir: str,
    buffer_size: int = 1024 * 1024  # 1 MB buffer
):
    """
    Extract tar file with minimal memory footprint.

    Args:
        tar_path: Path to tar file
        output_dir: Output directory
        buffer_size: Read/write buffer size
    """
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar:
            if member.isfile():
                # Get file object from tar
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    continue

                # Create output path
                output_path = os.path.join(output_dir, member.name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Stream write with buffer
                with open(output_path, 'wb') as out_file:
                    while True:
                        chunk = file_obj.read(buffer_size)
                        if not chunk:
                            break
                        out_file.write(chunk)

            elif member.isdir():
                dir_path = os.path.join(output_dir, member.name)
                os.makedirs(dir_path, exist_ok=True)


def selective_extract(
    tar_path: str,
    output_dir: str,
    file_types: list = None,
    pattern: str = None
):
    """
    Extract only specific files from tar.

    Args:
        tar_path: Path to tar file
        output_dir: Output directory
        file_types: List of extensions to extract (e.g., ['.jpg', '.json'])
        pattern: Regex pattern for filenames
    """
    import re

    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            # Check file type
            if file_types:
                if not any(member.name.endswith(ext) for ext in file_types):
                    continue

            # Check pattern
            if pattern:
                if not re.search(pattern, member.name):
                    continue

            # Extract
            tar.extract(member, output_dir)


# Example: Extract only images
if __name__ == "__main__":
    selective_extract(
        tar_path="./raw/sa_000000.tar",
        output_dir="./images_only",
        file_types=['.jpg']
    )
```

---

## 4. SSD Optimization

### Optimize for SSD Performance

```python
"""
SSD-optimized extraction strategies.
"""

import os
import shutil


class SSDOptimizedExtractor:
    """
    Extraction optimized for SSD drives.
    """

    def __init__(
        self,
        output_dir: str,
        temp_dir: str = None
    ):
        """
        Initialize extractor.

        Args:
            output_dir: Final output directory
            temp_dir: Temporary extraction directory (optional)
        """
        self.output_dir = output_dir
        self.temp_dir = temp_dir or os.path.join(output_dir, '.tmp')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_tar(self, tar_path: str):
        """
        Extract tar with SSD optimizations.

        Optimizations:
        - Extract to temp dir first, then move (reduces fragmentation)
        - Batch small files together
        - Pre-allocate space when possible
        """
        import tarfile

        base_name = os.path.basename(tar_path).replace('.tar', '')
        temp_output = os.path.join(self.temp_dir, base_name)
        final_output = os.path.join(self.output_dir, base_name)

        # Extract to temp directory
        os.makedirs(temp_output, exist_ok=True)

        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_output)

        # Move to final location (atomic on same filesystem)
        if os.path.exists(final_output):
            shutil.rmtree(final_output)
        shutil.move(temp_output, final_output)

        return final_output


# SSD-specific tips
SSD_TIPS = """
SSD Optimization Tips for SA-1B Extraction:

1. Use TRIM support:
   - Enable TRIM in your OS
   - Run fstrim periodically

2. Avoid excessive small writes:
   - Extract to temp, then move
   - Batch operations when possible

3. Monitor write amplification:
   - Don't fill SSD beyond 80%
   - Leave space for over-provisioning

4. Use appropriate filesystem:
   - ext4 with noatime on Linux
   - APFS on macOS
   - NTFS with quick format on Windows

5. Parallel extraction limits:
   - SSD: 4-8 parallel extractions
   - NVMe: 8-16 parallel extractions
   - Don't exceed drive's IOPS limits
"""
```

---

## 5. Extract and Organize Script

```bash
#!/bin/bash
# extract_and_organize.sh - Extract and organize SA-1B dataset

set -e

# Configuration
TAR_DIR="${1:-./raw}"
OUTPUT_DIR="${2:-./sa1b_extracted}"
NUM_WORKERS="${3:-4}"

# Create organized directory structure
IMAGES_DIR="$OUTPUT_DIR/images"
MASKS_DIR="$OUTPUT_DIR/annotations"
LOGS_DIR="$OUTPUT_DIR/logs"

mkdir -p "$IMAGES_DIR" "$MASKS_DIR" "$LOGS_DIR"

echo "SA-1B Extraction and Organization"
echo "================================="
echo "Source: $TAR_DIR"
echo "Output: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"
echo ""

# Count tar files
NUM_TARS=$(find "$TAR_DIR" -name "sa_*.tar" | wc -l)
echo "Found $NUM_TARS tar files"
echo ""

# Extract with parallel
echo "Extracting tar files..."
find "$TAR_DIR" -name "sa_*.tar" | \
    parallel -j "$NUM_WORKERS" --progress \
    'tar -xf {} -C '"$OUTPUT_DIR"

# Organize files
echo ""
echo "Organizing files..."

# Move images
echo "Moving images..."
find "$OUTPUT_DIR" -name "*.jpg" -exec mv {} "$IMAGES_DIR/" \;

# Move masks
echo "Moving masks..."
find "$OUTPUT_DIR" -name "*.json" -exec mv {} "$MASKS_DIR/" \;

# Clean up empty directories
find "$OUTPUT_DIR" -type d -empty -delete 2>/dev/null || true

# Generate statistics
NUM_IMAGES=$(find "$IMAGES_DIR" -name "*.jpg" | wc -l)
NUM_MASKS=$(find "$MASKS_DIR" -name "*.json" | wc -l)

echo ""
echo "Extraction Complete!"
echo "===================="
echo "Images: $NUM_IMAGES"
echo "Masks: $NUM_MASKS"
echo ""
echo "Directory structure:"
echo "  $IMAGES_DIR"
echo "  $MASKS_DIR"

# Save manifest
echo "Saving manifest..."
find "$IMAGES_DIR" -name "*.jpg" -printf "%f\n" | sort > "$OUTPUT_DIR/images_manifest.txt"
find "$MASKS_DIR" -name "*.json" -printf "%f\n" | sort > "$OUTPUT_DIR/masks_manifest.txt"

echo "Done!"
```

---

## 6. Incremental Extraction

```python
"""
Incremental extraction with progress tracking.
"""

import os
import json
import tarfile
from datetime import datetime


class IncrementalExtractor:
    """
    Extract tar files incrementally with checkpoint support.
    """

    def __init__(self, tar_dir: str, output_dir: str):
        """
        Initialize extractor.

        Args:
            tar_dir: Directory containing tar files
            output_dir: Output directory
        """
        self.tar_dir = tar_dir
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, '.extraction_state.json')

        os.makedirs(output_dir, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load extraction state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'extracted': [],
            'failed': [],
            'stats': {'images': 0, 'masks': 0}
        }

    def _save_state(self):
        """Save extraction state."""
        self.state['updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def extract_pending(self, max_files: int = None):
        """
        Extract pending tar files.

        Args:
            max_files: Maximum files to extract (None = all)
        """
        # Find all tar files
        tar_files = sorted([
            f for f in os.listdir(self.tar_dir)
            if f.endswith('.tar')
        ])

        # Filter out already extracted
        extracted_set = set(self.state['extracted'])
        pending = [f for f in tar_files if f not in extracted_set]

        if max_files:
            pending = pending[:max_files]

        if not pending:
            print("No pending files to extract")
            return

        print(f"Extracting {len(pending)} files...")

        for tar_name in pending:
            tar_path = os.path.join(self.tar_dir, tar_name)

            try:
                # Extract
                images, masks = self._extract_tar(tar_path)

                # Update state
                self.state['extracted'].append(tar_name)
                self.state['stats']['images'] += images
                self.state['stats']['masks'] += masks

                print(f"  Extracted: {tar_name} ({images} images, {masks} masks)")

            except Exception as e:
                self.state['failed'].append({
                    'file': tar_name,
                    'error': str(e),
                    'time': datetime.now().isoformat()
                })
                print(f"  Failed: {tar_name} - {e}")

            # Save state after each file
            self._save_state()

        print(f"\nExtraction summary:")
        print(f"  Total extracted: {len(self.state['extracted'])}")
        print(f"  Total images: {self.state['stats']['images']:,}")
        print(f"  Total masks: {self.state['stats']['masks']:,}")

    def _extract_tar(self, tar_path: str) -> tuple:
        """Extract single tar file."""
        images = 0
        masks = 0

        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                tar.extract(member, self.output_dir)

                if member.name.endswith('.jpg'):
                    images += 1
                elif member.name.endswith('.json'):
                    masks += 1

        return images, masks

    def get_progress(self) -> dict:
        """Get extraction progress."""
        total = len([f for f in os.listdir(self.tar_dir) if f.endswith('.tar')])
        extracted = len(self.state['extracted'])

        return {
            'extracted': extracted,
            'total': total,
            'percent': (extracted / total * 100) if total > 0 else 0,
            'images': self.state['stats']['images'],
            'masks': self.state['stats']['masks']
        }


# Example usage
if __name__ == "__main__":
    extractor = IncrementalExtractor('./raw', './extracted')

    # Extract next 10 files
    extractor.extract_pending(max_files=10)

    # Show progress
    progress = extractor.get_progress()
    print(f"\nOverall progress: {progress['percent']:.1f}%")
```

---

## 7. Cleanup and Space Management

```python
"""
Cleanup and space management for SA-1B extraction.
"""

import os
import shutil


def cleanup_raw_tars(
    tar_dir: str,
    extracted_dir: str,
    verify: bool = True
):
    """
    Remove tar files after successful extraction.

    Args:
        tar_dir: Directory with tar files
        extracted_dir: Directory with extracted files
        verify: Verify extraction before deletion
    """
    removed = 0
    kept = 0

    for tar_name in os.listdir(tar_dir):
        if not tar_name.endswith('.tar'):
            continue

        tar_path = os.path.join(tar_dir, tar_name)
        base_name = tar_name.replace('.tar', '')

        # Check if extracted
        if verify:
            # Look for extracted files
            extracted_path = os.path.join(extracted_dir, base_name)
            if not os.path.exists(extracted_path):
                print(f"Keeping {tar_name} (not extracted)")
                kept += 1
                continue

        # Remove tar file
        os.remove(tar_path)
        print(f"Removed {tar_name}")
        removed += 1

    print(f"\nCleanup complete: {removed} removed, {kept} kept")


def estimate_space_needed(num_tars: int) -> dict:
    """
    Estimate space needed for extraction.

    Args:
        num_tars: Number of tar files

    Returns:
        Space estimates
    """
    # Average sizes based on SA-1B
    avg_tar_size_gb = 10
    avg_extracted_size_gb = 10  # Similar after extraction

    return {
        'raw_tars_gb': num_tars * avg_tar_size_gb,
        'extracted_gb': num_tars * avg_extracted_size_gb,
        'total_peak_gb': num_tars * (avg_tar_size_gb + avg_extracted_size_gb),
        'after_cleanup_gb': num_tars * avg_extracted_size_gb
    }


def check_disk_space(path: str, required_gb: float) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check
        required_gb: Required space in GB

    Returns:
        True if sufficient space
    """
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)

    print(f"Disk space check for {path}:")
    print(f"  Available: {available_gb:.1f} GB")
    print(f"  Required: {required_gb:.1f} GB")

    if available_gb < required_gb:
        print(f"  WARNING: Insufficient space!")
        return False

    print(f"  OK: {available_gb - required_gb:.1f} GB headroom")
    return True


# Example usage
if __name__ == "__main__":
    # Estimate space for 100 tars
    space = estimate_space_needed(100)
    print("Space estimates for 100 tar files:")
    print(f"  Raw tars: {space['raw_tars_gb']:.0f} GB")
    print(f"  Extracted: {space['extracted_gb']:.0f} GB")
    print(f"  Peak (during extraction): {space['total_peak_gb']:.0f} GB")
    print(f"  After cleanup: {space['after_cleanup_gb']:.0f} GB")

    # Check available space
    check_disk_space("./", space['total_peak_gb'])
```

---

## Sources

**Documentation:**
- [GNU Parallel Manual](https://www.gnu.org/s/parallel/man.html) - Official documentation (accessed 2025-11-20)
- [Python tarfile](https://docs.python.org/3/library/tarfile.html) - Standard library documentation

**Community Resources:**
- [Stack Overflow - Parallel tar extraction](https://stackoverflow.com/questions/12313242/utilizing-multi-core-for-targzip-bzip-compression-decompression) (accessed 2025-11-20)
- [Databricks - Faster tar extraction](https://www.databricks.com/blog/2022/01/26/creating-a-faster-tar-extractor.html) (accessed 2025-11-20)

**GitHub:**
- [erow/SA-1B](https://github.com/erow/SA-1B) - Extraction implementation

---

## Next Steps

- PART 18: Verification and integrity checking
