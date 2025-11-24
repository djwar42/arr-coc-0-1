# KNOWLEDGE DROP: SA-1B Parallel Downloader Tools

**Date**: 2025-11-20
**Runner**: PART 14 (Parallel Downloader Tools)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers community tools and utilities for downloading SA-1B in parallel, including aria2c configurations, Python multi-threaded downloaders, and bandwidth optimization techniques.

---

## 1. Community Downloader Tools

### erow/SA-1B Downloader

From [erow/SA-1B](https://github.com/erow/SA-1B) (accessed 2025-11-20):

**Features**:
- Multi-process downloading and extraction
- Progress tracking
- Skip existing files
- Automatic extraction after download

```python
"""
SA-1B Downloader - Based on erow/SA-1B

Installation:
    pip install requests pycocotools

Usage:
    python download.py --processes 4 --input_file sa1b_links.txt
"""

import os
import tarfile
import requests
from multiprocessing import Pool
from functools import partial
import argparse


def download_and_extract(args, raw_dir, images_dir, masks_dir, skip_existing):
    """
    Download and extract a single tar file.

    Args:
        args: Tuple of (filename, url)
        raw_dir: Directory for raw tar files
        images_dir: Directory for extracted images
        masks_dir: Directory for extracted masks
        skip_existing: Whether to skip existing files
    """
    filename, url = args

    tar_path = os.path.join(raw_dir, filename)
    base_name = filename.replace('.tar', '')

    # Check if already extracted
    if skip_existing:
        expected_dir = os.path.join(images_dir, base_name)
        if os.path.exists(expected_dir):
            print(f"Skipping {filename} (already extracted)")
            return

    # Download if not exists
    if not os.path.exists(tar_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)

        print(f"Downloaded {filename}")

    # Extract
    print(f"Extracting {filename}...")
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.jpg'):
                tar.extract(member, images_dir)
            elif member.name.endswith('.json'):
                tar.extract(member, masks_dir)

    print(f"Extracted {filename}")


def main():
    parser = argparse.ArgumentParser(description='Download SA-1B dataset')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes')
    parser.add_argument('--input_file', type=str, default='sa1b_links.txt',
                       help='Input file with download links')
    parser.add_argument('--raw_dir', type=str, default='raw',
                       help='Directory for raw tar files')
    parser.add_argument('--images_dir', type=str, default='images',
                       help='Directory for extracted images')
    parser.add_argument('--masks_dir', type=str, default='annotations',
                       help='Directory for extracted masks')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip already extracted files')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)

    # Parse links file
    downloads = []
    with open(args.input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                url = parts[-1]
                downloads.append((filename, url))

    print(f"Found {len(downloads)} files to download")
    print(f"Using {args.processes} parallel processes")

    # Download in parallel
    download_func = partial(
        download_and_extract,
        raw_dir=args.raw_dir,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        skip_existing=args.skip_existing
    )

    with Pool(args.processes) as pool:
        pool.map(download_func, downloads)

    print("Download complete!")


if __name__ == "__main__":
    main()
```

---

## 2. aria2c Multi-threaded Downloader

### Why aria2c?

From [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) and [aria2 documentation](https://aria2.github.io/manual/en/html/aria2c.html) (accessed 2025-11-20):

**Advantages**:
- Multi-connection downloads (splits file into segments)
- Automatic resume on failure
- Built-in retry logic
- Progress reporting
- Metalink support for mirrors

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install aria2

# macOS
brew install aria2

# Windows
# Download from https://aria2.github.io/
```

### Basic aria2c Usage

```bash
# Download single file with 16 connections
aria2c -x 16 -s 16 https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar

# Download from links file
aria2c -i sa1b_links.txt -x 16 -s 16 -j 4

# With all optimizations
aria2c \
    -i sa1b_links.txt \
    -x 16 \
    -s 16 \
    -j 4 \
    -k 1M \
    --max-overall-download-limit=0 \
    --file-allocation=falloc \
    --continue=true \
    --auto-file-renaming=false \
    -d ./raw
```

### aria2c Parameters Explained

From [aria2c manual](https://aria2.github.io/manual/en/html/aria2c.html) (accessed 2025-11-20):

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `-x N` | Max connections per server | 16 |
| `-s N` | Split file into N pieces | 16 |
| `-j N` | Max concurrent downloads | 2-4 |
| `-k SIZE` | Min split size | 1M |
| `--continue` | Resume partial downloads | true |
| `--file-allocation` | Pre-allocate space | falloc |
| `-d DIR` | Download directory | ./raw |

### Optimized aria2c Configuration

```bash
#!/bin/bash
# download_sa1b_aria2.sh - Optimized SA-1B download script

# Configuration
INPUT_FILE="sa1b_links.txt"
OUTPUT_DIR="./raw"
LOG_FILE="download.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run aria2c with optimized settings
aria2c \
    --input-file="$INPUT_FILE" \
    --dir="$OUTPUT_DIR" \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --max-concurrent-downloads=4 \
    --continue=true \
    --auto-file-renaming=false \
    --file-allocation=falloc \
    --max-tries=5 \
    --retry-wait=10 \
    --timeout=600 \
    --connect-timeout=60 \
    --max-overall-download-limit=0 \
    --max-download-limit=0 \
    --summary-interval=60 \
    --console-log-level=notice \
    --log="$LOG_FILE" \
    --log-level=info

echo "Download complete! Check $LOG_FILE for details."
```

---

## 3. Python Multi-threaded Downloader

### Advanced Downloader with Progress

```python
"""
Advanced multi-threaded SA-1B downloader with progress tracking.
"""

import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm
import time


class SA1BDownloader:
    """
    Multi-threaded SA-1B dataset downloader.
    """

    def __init__(
        self,
        output_dir: str,
        max_workers: int = 4,
        chunk_size: int = 1024 * 1024,  # 1 MB
        max_retries: int = 5
    ):
        """
        Initialize downloader.

        Args:
            output_dir: Directory for downloaded files
            max_workers: Number of parallel downloads
            chunk_size: Download chunk size in bytes
            max_retries: Maximum retry attempts per file
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_retries = max_retries

        os.makedirs(output_dir, exist_ok=True)

        # Progress tracking
        self.completed = 0
        self.failed = []
        self.lock = threading.Lock()

    def download_file(self, filename: str, url: str) -> bool:
        """
        Download a single file with resume support.

        Args:
            filename: Output filename
            url: Download URL

        Returns:
            True if successful
        """
        output_path = os.path.join(self.output_dir, filename)
        temp_path = output_path + '.partial'

        for attempt in range(self.max_retries):
            try:
                # Check for existing partial download
                start_byte = 0
                if os.path.exists(temp_path):
                    start_byte = os.path.getsize(temp_path)

                # Check if already complete
                if os.path.exists(output_path):
                    return True

                # Set up request headers for resume
                headers = {}
                if start_byte > 0:
                    headers['Range'] = f'bytes={start_byte}-'

                # Download
                response = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=600
                )
                response.raise_for_status()

                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                if start_byte > 0:
                    total_size += start_byte

                # Write to file
                mode = 'ab' if start_byte > 0 else 'wb'
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)

                # Rename to final filename
                os.rename(temp_path, output_path)
                return True

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Failed to download {filename}: {e}")
                    return False

        return False

    def download_all(self, downloads: list) -> dict:
        """
        Download all files in parallel.

        Args:
            downloads: List of (filename, url) tuples

        Returns:
            Dictionary with results
        """
        results = {
            'completed': [],
            'failed': [],
            'total': len(downloads)
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all downloads
            futures = {
                executor.submit(self.download_file, filename, url): filename
                for filename, url in downloads
            }

            # Track progress
            with tqdm(total=len(downloads), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        success = future.result()
                        if success:
                            results['completed'].append(filename)
                        else:
                            results['failed'].append(filename)
                    except Exception as e:
                        results['failed'].append(filename)
                        print(f"Error downloading {filename}: {e}")

                    pbar.update(1)

        return results


def download_with_progress(links_file: str, output_dir: str, workers: int = 4):
    """
    Download SA-1B dataset with progress tracking.

    Args:
        links_file: Path to links.txt
        output_dir: Output directory
        workers: Number of parallel workers
    """
    # Parse links file
    downloads = []
    with open(links_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                url = parts[-1]
                downloads.append((filename, url))

    print(f"Found {len(downloads)} files to download")
    print(f"Using {workers} parallel workers")
    print(f"Output directory: {output_dir}")

    # Download
    downloader = SA1BDownloader(output_dir, max_workers=workers)
    results = downloader.download_all(downloads)

    # Report
    print(f"\nDownload complete!")
    print(f"  Completed: {len(results['completed'])}/{results['total']}")
    print(f"  Failed: {len(results['failed'])}")

    if results['failed']:
        print(f"\nFailed files:")
        for f in results['failed']:
            print(f"  - {f}")

    return results


# Example usage
if __name__ == "__main__":
    results = download_with_progress(
        links_file="sa1b_links.txt",
        output_dir="./raw",
        workers=4
    )
```

---

## 4. Bandwidth Optimization

### Connection Pooling

```python
"""
Optimized connection pooling for maximum bandwidth utilization.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3 import PoolManager


def create_optimized_session() -> requests.Session:
    """
    Create a session optimized for high-bandwidth downloads.

    Returns:
        Configured requests Session
    """
    session = requests.Session()

    # Retry configuration
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )

    # Adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20,
        pool_block=False
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Optimize headers
    session.headers.update({
        'User-Agent': 'SA-1B-Downloader/1.0',
        'Accept-Encoding': 'identity',  # Disable compression for large files
        'Connection': 'keep-alive'
    })

    return session


def calculate_optimal_workers(bandwidth_mbps: float, file_size_gb: float = 10) -> int:
    """
    Calculate optimal number of parallel workers based on bandwidth.

    Args:
        bandwidth_mbps: Available bandwidth in Mbps
        file_size_gb: Average file size in GB

    Returns:
        Recommended number of workers
    """
    # Each worker should get at least 50 Mbps for efficiency
    min_bandwidth_per_worker = 50

    # Calculate based on bandwidth
    by_bandwidth = int(bandwidth_mbps / min_bandwidth_per_worker)

    # Cap based on practical limits
    # Too many workers can cause issues
    optimal = max(1, min(by_bandwidth, 8))

    return optimal
```

### Segmented Downloads

```python
"""
Segmented download for single large files.
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor
import tempfile


def download_segment(url: str, start: int, end: int, output_file: str) -> bool:
    """
    Download a segment of a file.

    Args:
        url: Download URL
        start: Start byte
        end: End byte
        output_file: Output file path

    Returns:
        True if successful
    """
    headers = {'Range': f'bytes={start}-{end}'}

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

    return True


def download_file_segmented(
    url: str,
    output_path: str,
    num_segments: int = 16
) -> bool:
    """
    Download a file using multiple parallel segments.

    Args:
        url: Download URL
        output_path: Output file path
        num_segments: Number of parallel segments

    Returns:
        True if successful
    """
    # Get file size
    response = requests.head(url)
    total_size = int(response.headers.get('content-length', 0))

    if total_size == 0:
        # Fallback to single download
        response = requests.get(url, stream=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        return True

    # Calculate segment sizes
    segment_size = total_size // num_segments
    segments = []

    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size - 1 if i < num_segments - 1 else total_size - 1
        segments.append((start, end))

    # Download segments in parallel
    temp_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with ThreadPoolExecutor(max_workers=num_segments) as executor:
            futures = []

            for i, (start, end) in enumerate(segments):
                temp_file = os.path.join(temp_dir, f"segment_{i}")
                temp_files.append(temp_file)
                futures.append(
                    executor.submit(download_segment, url, start, end, temp_file)
                )

            # Wait for all segments
            for future in futures:
                future.result()

        # Combine segments
        with open(output_path, 'wb') as output:
            for temp_file in temp_files:
                with open(temp_file, 'rb') as segment:
                    output.write(segment.read())

    return True
```

---

## 5. Resume Support Implementation

```python
"""
Robust resume support for interrupted downloads.
"""

import os
import json
import hashlib
from datetime import datetime


class ResumeManager:
    """
    Manage download resume state.
    """

    def __init__(self, state_file: str):
        """
        Initialize resume manager.

        Args:
            state_file: Path to state file
        """
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'completed': [],
            'in_progress': {},
            'failed': [],
            'started': datetime.now().isoformat()
        }

    def _save_state(self):
        """Save state to file."""
        self.state['updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def mark_completed(self, filename: str, checksum: str = None):
        """Mark file as completed."""
        if filename not in self.state['completed']:
            self.state['completed'].append(filename)

        if filename in self.state['in_progress']:
            del self.state['in_progress'][filename]

        if filename in self.state['failed']:
            self.state['failed'].remove(filename)

        self._save_state()

    def mark_in_progress(self, filename: str, bytes_downloaded: int):
        """Update progress for file."""
        self.state['in_progress'][filename] = {
            'bytes': bytes_downloaded,
            'updated': datetime.now().isoformat()
        }
        self._save_state()

    def mark_failed(self, filename: str, error: str):
        """Mark file as failed."""
        if filename not in self.state['failed']:
            self.state['failed'].append(filename)

        if filename in self.state['in_progress']:
            del self.state['in_progress'][filename]

        self._save_state()

    def is_completed(self, filename: str) -> bool:
        """Check if file is completed."""
        return filename in self.state['completed']

    def get_resume_byte(self, filename: str) -> int:
        """Get byte to resume from."""
        if filename in self.state['in_progress']:
            return self.state['in_progress'][filename]['bytes']
        return 0

    def get_summary(self) -> dict:
        """Get download summary."""
        return {
            'completed': len(self.state['completed']),
            'in_progress': len(self.state['in_progress']),
            'failed': len(self.state['failed']),
            'started': self.state.get('started'),
            'updated': self.state.get('updated')
        }
```

---

## 6. Progress Tracking

```python
"""
Real-time progress tracking for SA-1B downloads.
"""

import sys
import time
from datetime import timedelta


class ProgressTracker:
    """
    Track download progress with ETA calculation.
    """

    def __init__(self, total_files: int, total_bytes: int = 0):
        """
        Initialize tracker.

        Args:
            total_files: Total number of files
            total_bytes: Total bytes to download (0 if unknown)
        """
        self.total_files = total_files
        self.total_bytes = total_bytes

        self.completed_files = 0
        self.downloaded_bytes = 0
        self.start_time = time.time()

    def update(self, files: int = 0, bytes_: int = 0):
        """Update progress."""
        self.completed_files += files
        self.downloaded_bytes += bytes_

    def get_eta(self) -> str:
        """Calculate estimated time remaining."""
        elapsed = time.time() - self.start_time
        if elapsed == 0 or self.completed_files == 0:
            return "calculating..."

        # ETA based on files
        files_per_second = self.completed_files / elapsed
        remaining_files = self.total_files - self.completed_files

        if files_per_second > 0:
            remaining_seconds = remaining_files / files_per_second
            return str(timedelta(seconds=int(remaining_seconds)))

        return "unknown"

    def get_speed(self) -> str:
        """Calculate download speed."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return "0 MB/s"

        speed_mbps = (self.downloaded_bytes / (1024*1024)) / elapsed
        return f"{speed_mbps:.1f} MB/s"

    def print_status(self):
        """Print current status."""
        percent = (self.completed_files / self.total_files) * 100

        status = (
            f"\rProgress: {self.completed_files}/{self.total_files} "
            f"({percent:.1f}%) | "
            f"Speed: {self.get_speed()} | "
            f"ETA: {self.get_eta()}"
        )

        sys.stdout.write(status)
        sys.stdout.flush()
```

---

## 7. Complete Download Script

```bash
#!/bin/bash
# complete_sa1b_download.sh - Complete SA-1B download solution

set -e

# Configuration
LINKS_FILE="${1:-sa1b_links.txt}"
OUTPUT_DIR="${2:-./sa1b_raw}"
NUM_PROCESSES="${3:-4}"
LOG_DIR="./logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/download_$TIMESTAMP.log"

echo "SA-1B Dataset Downloader"
echo "========================"
echo "Links file: $LINKS_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Processes: $NUM_PROCESSES"
echo "Log file: $LOG_FILE"
echo ""

# Check if aria2c is available
if command -v aria2c &> /dev/null; then
    echo "Using aria2c for optimized downloads..."

    aria2c \
        --input-file="$LINKS_FILE" \
        --dir="$OUTPUT_DIR" \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --max-concurrent-downloads="$NUM_PROCESSES" \
        --continue=true \
        --auto-file-renaming=false \
        --file-allocation=falloc \
        --max-tries=5 \
        --retry-wait=10 \
        --timeout=600 \
        --log="$LOG_FILE" \
        --log-level=info \
        --summary-interval=60

else
    echo "aria2c not found. Using wget fallback..."

    # Parallel wget downloads
    cat "$LINKS_FILE" | while read filename url; do
        if [ -n "$filename" ] && [ -n "$url" ]; then
            echo "$url"
        fi
    done | xargs -P "$NUM_PROCESSES" -I {} wget -c -P "$OUTPUT_DIR" {}

fi

echo ""
echo "Download complete!"
echo "Files saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
```

---

## Sources

**GitHub Repositories:**
- [erow/SA-1B](https://github.com/erow/SA-1B) - SA-1B Downloader tool (accessed 2025-11-20)
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official SAM repository
- [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - Download discussion

**Documentation:**
- [aria2c Manual](https://aria2.github.io/manual/en/html/aria2c.html) - Official aria2 documentation (accessed 2025-11-20)
- [aria2 GitHub](https://github.com/aria2/aria2) - aria2 source repository

**Community Resources:**
- [Stack Overflow - aria2c parallel downloads](https://stackoverflow.com/questions/55166245/aria2c-parallel-download-parameters) (accessed 2025-11-20)
- [Reddit r/DataHoarder](https://www.reddit.com/r/DataHoarder/) - Large dataset download discussions

---

## Next Steps

- PART 15: Download time optimization strategies
- PART 16: Partial download and subset selection
- PART 17: Tar extraction strategies
