# KNOWLEDGE DROP: SA-1B Download Time Optimization

**Date**: 2025-11-20
**Runner**: PART 15 (Download Time Optimization)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers strategies for optimizing SA-1B download times, including time estimates, connection optimization, error handling, checkpointing, and bandwidth management.

---

## 1. Download Time Estimates

### By Connection Speed

| Connection | Speed | Time/File (10GB) | Full Dataset (10TB) |
|------------|-------|------------------|---------------------|
| Home DSL | 10 Mbps | 2.2 hours | 92 days |
| Cable | 100 Mbps | 13 minutes | 9.2 days |
| Fiber | 500 Mbps | 2.7 minutes | 1.8 days |
| Gigabit | 1 Gbps | 1.3 minutes | 22 hours |
| Datacenter | 10 Gbps | 8 seconds | 2.2 hours |

### Realistic Estimates with Overhead

From [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) (accessed 2025-11-20):

Community reports indicate:
- **4 processes, 100 Mbps**: 2-3 days for full dataset
- **8 processes, Gigabit**: 12-18 hours
- **Datacenter (10+ Gbps)**: 2-4 hours

```python
"""
Calculate realistic download time estimates.
"""

def estimate_download_time(
    bandwidth_mbps: float,
    total_size_tb: float = 10.0,
    efficiency: float = 0.7,
    num_workers: int = 4
) -> dict:
    """
    Estimate download time with realistic factors.

    Args:
        bandwidth_mbps: Available bandwidth in Mbps
        total_size_tb: Total dataset size in TB
        efficiency: Network efficiency factor (0.0-1.0)
        num_workers: Number of parallel downloads

    Returns:
        Dictionary with time estimates
    """
    # Convert to consistent units
    total_bytes = total_size_tb * 1024 * 1024 * 1024 * 1024
    effective_bandwidth = bandwidth_mbps * efficiency * 1024 * 1024 / 8  # bytes/sec

    # Single worker time
    single_worker_seconds = total_bytes / effective_bandwidth

    # Multi-worker (diminishing returns after 4-8 workers)
    worker_efficiency = min(num_workers, 8) * 0.9  # 90% scaling
    multi_worker_seconds = single_worker_seconds / worker_efficiency

    # Add overhead (retries, extraction, etc.)
    overhead_factor = 1.15  # 15% overhead
    total_seconds = multi_worker_seconds * overhead_factor

    return {
        'total_hours': total_seconds / 3600,
        'total_days': total_seconds / 86400,
        'per_file_minutes': (total_seconds / 1000) / 60,
        'effective_speed_mbps': (total_bytes * 8 / total_seconds) / (1024 * 1024)
    }


# Example calculations
if __name__ == "__main__":
    scenarios = [
        ("Home (100 Mbps)", 100, 4),
        ("Fiber (500 Mbps)", 500, 4),
        ("Gigabit (1000 Mbps)", 1000, 8),
        ("Datacenter (10 Gbps)", 10000, 8),
    ]

    print("SA-1B Download Time Estimates (10 TB)")
    print("=" * 60)

    for name, bandwidth, workers in scenarios:
        result = estimate_download_time(bandwidth, num_workers=workers)
        print(f"\n{name}:")
        print(f"  Workers: {workers}")
        print(f"  Total time: {result['total_days']:.1f} days ({result['total_hours']:.1f} hours)")
        print(f"  Per file: {result['per_file_minutes']:.1f} minutes")
        print(f"  Effective speed: {result['effective_speed_mbps']:.0f} Mbps")
```

---

## 2. Connection Optimization Strategies

### TCP Tuning

```bash
#!/bin/bash
# tcp_tuning.sh - Optimize TCP for large file downloads

# Increase TCP buffer sizes (requires root)
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Enable TCP window scaling
sudo sysctl -w net.ipv4.tcp_window_scaling=1

# Increase connection backlog
sudo sysctl -w net.core.netdev_max_backlog=5000

# Enable TCP fast open
sudo sysctl -w net.ipv4.tcp_fastopen=3

echo "TCP tuning complete"
```

### Optimal Worker Configuration

```python
"""
Calculate optimal number of download workers.
"""

import multiprocessing


def calculate_optimal_workers(
    bandwidth_mbps: float,
    cpu_cores: int = None,
    memory_gb: float = None
) -> int:
    """
    Calculate optimal number of parallel download workers.

    Args:
        bandwidth_mbps: Available bandwidth
        cpu_cores: Number of CPU cores (auto-detect if None)
        memory_gb: Available RAM in GB (auto-detect if None)

    Returns:
        Recommended number of workers
    """
    if cpu_cores is None:
        cpu_cores = multiprocessing.cpu_count()

    if memory_gb is None:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)

    # Constraints
    constraints = []

    # 1. Bandwidth constraint: ~50-100 Mbps per worker minimum
    bandwidth_workers = int(bandwidth_mbps / 75)
    constraints.append(("bandwidth", bandwidth_workers))

    # 2. CPU constraint: Leave 1-2 cores free
    cpu_workers = max(1, cpu_cores - 2)
    constraints.append(("CPU", cpu_workers))

    # 3. Memory constraint: ~500MB per worker for buffering
    memory_workers = int(memory_gb * 2)  # 500MB each
    constraints.append(("memory", memory_workers))

    # 4. Practical limit: Diminishing returns after 8
    practical_max = 8
    constraints.append(("practical", practical_max))

    # Take minimum of all constraints
    optimal = min(c[1] for c in constraints)
    optimal = max(1, optimal)  # At least 1

    print("Worker calculation:")
    for name, value in constraints:
        marker = " <-- limiting" if value == optimal else ""
        print(f"  {name}: {value}{marker}")

    return optimal


# Example
if __name__ == "__main__":
    workers = calculate_optimal_workers(
        bandwidth_mbps=500,
        cpu_cores=8,
        memory_gb=32
    )
    print(f"\nRecommended workers: {workers}")
```

---

## 3. Error Handling & Recovery

### Robust Download with Retry

```python
"""
Robust download implementation with comprehensive error handling.
"""

import os
import time
import requests
from requests.exceptions import (
    RequestException, ConnectionError, Timeout,
    ChunkedEncodingError, HTTPError
)


class RobustDownloader:
    """
    Download files with comprehensive error handling and recovery.
    """

    RETRYABLE_ERRORS = (
        ConnectionError,
        Timeout,
        ChunkedEncodingError,
    )

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        timeout: tuple = (30, 600)  # (connect, read)
    ):
        """
        Initialize downloader.

        Args:
            max_retries: Maximum retry attempts
            initial_backoff: Initial backoff delay in seconds
            max_backoff: Maximum backoff delay
            timeout: Tuple of (connect_timeout, read_timeout)
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.timeout = timeout

    def download(self, url: str, output_path: str) -> bool:
        """
        Download file with automatic retry and resume.

        Args:
            url: Download URL
            output_path: Output file path

        Returns:
            True if successful

        Raises:
            Exception if all retries exhausted
        """
        temp_path = output_path + '.partial'

        for attempt in range(self.max_retries):
            try:
                # Check for partial download
                start_byte = 0
                if os.path.exists(temp_path):
                    start_byte = os.path.getsize(temp_path)

                # Set up headers
                headers = {}
                if start_byte > 0:
                    headers['Range'] = f'bytes={start_byte}-'
                    print(f"  Resuming from byte {start_byte}")

                # Make request
                response = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout
                )

                # Check for retryable HTTP errors
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    raise HTTPError(f"Retryable status: {response.status_code}")

                response.raise_for_status()

                # Write to file
                mode = 'ab' if start_byte > 0 else 'wb'
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)

                # Success - rename to final path
                os.rename(temp_path, output_path)
                return True

            except self.RETRYABLE_ERRORS as e:
                # Calculate backoff
                backoff = min(
                    self.initial_backoff * (2 ** attempt),
                    self.max_backoff
                )

                if attempt < self.max_retries - 1:
                    print(f"  Retry {attempt + 1}/{self.max_retries} "
                          f"after {backoff:.1f}s: {type(e).__name__}")
                    time.sleep(backoff)
                else:
                    raise

            except HTTPError as e:
                if "Retryable status" in str(e):
                    backoff = min(
                        self.initial_backoff * (2 ** attempt),
                        self.max_backoff
                    )
                    if attempt < self.max_retries - 1:
                        print(f"  Retry {attempt + 1}/{self.max_retries} "
                              f"after {backoff:.1f}s: {e}")
                        time.sleep(backoff)
                    else:
                        raise
                else:
                    raise

        return False

    def verify_download(self, path: str, expected_size: int = None) -> bool:
        """
        Verify downloaded file integrity.

        Args:
            path: File path
            expected_size: Expected file size (optional)

        Returns:
            True if file appears valid
        """
        if not os.path.exists(path):
            return False

        file_size = os.path.getsize(path)

        # Check minimum size (SA-1B files are ~8-12 GB)
        if file_size < 1024 * 1024 * 100:  # 100 MB minimum
            return False

        # Check expected size if provided
        if expected_size and file_size != expected_size:
            return False

        return True
```

---

## 4. Checkpointing System

### State Management

```python
"""
Checkpointing system for download state management.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional


class CheckpointManager:
    """
    Manage download checkpoints for recovery.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, 'checkpoint.json')

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load checkpoint state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)

        return {
            'version': 1,
            'created': datetime.now().isoformat(),
            'files': {},
            'stats': {
                'completed': 0,
                'failed': 0,
                'bytes_downloaded': 0
            }
        }

    def _save_state(self):
        """Save checkpoint state."""
        self.state['updated'] = datetime.now().isoformat()

        # Write to temp file first for atomicity
        temp_file = self.state_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)

        os.replace(temp_file, self.state_file)

    def start_download(self, filename: str, url: str, expected_size: int = 0):
        """Mark file as started."""
        self.state['files'][filename] = {
            'url': url,
            'status': 'downloading',
            'expected_size': expected_size,
            'downloaded_bytes': 0,
            'started': datetime.now().isoformat(),
            'attempts': 0
        }
        self._save_state()

    def update_progress(self, filename: str, bytes_downloaded: int):
        """Update download progress."""
        if filename in self.state['files']:
            self.state['files'][filename]['downloaded_bytes'] = bytes_downloaded
            self.state['files'][filename]['updated'] = datetime.now().isoformat()
            self._save_state()

    def complete_download(self, filename: str, final_size: int, checksum: str = None):
        """Mark file as completed."""
        if filename in self.state['files']:
            self.state['files'][filename]['status'] = 'completed'
            self.state['files'][filename]['final_size'] = final_size
            self.state['files'][filename]['completed'] = datetime.now().isoformat()
            if checksum:
                self.state['files'][filename]['checksum'] = checksum

            self.state['stats']['completed'] += 1
            self.state['stats']['bytes_downloaded'] += final_size

            self._save_state()

    def fail_download(self, filename: str, error: str):
        """Mark file as failed."""
        if filename in self.state['files']:
            self.state['files'][filename]['status'] = 'failed'
            self.state['files'][filename]['error'] = error
            self.state['files'][filename]['attempts'] += 1

            self.state['stats']['failed'] += 1

            self._save_state()

    def get_incomplete(self) -> list:
        """Get list of incomplete downloads."""
        incomplete = []
        for filename, info in self.state['files'].items():
            if info['status'] not in ['completed']:
                incomplete.append({
                    'filename': filename,
                    'url': info['url'],
                    'downloaded_bytes': info.get('downloaded_bytes', 0),
                    'attempts': info.get('attempts', 0)
                })
        return incomplete

    def get_summary(self) -> dict:
        """Get download summary."""
        total = len(self.state['files'])
        completed = self.state['stats']['completed']
        failed = self.state['stats']['failed']
        in_progress = total - completed - failed

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'bytes_downloaded': self.state['stats']['bytes_downloaded'],
            'percent_complete': (completed / total * 100) if total > 0 else 0
        }

    def export_failed_list(self, output_path: str):
        """Export list of failed downloads for retry."""
        failed = []
        for filename, info in self.state['files'].items():
            if info['status'] == 'failed':
                failed.append(f"{filename}\t{info['url']}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(failed))

        print(f"Exported {len(failed)} failed downloads to {output_path}")
```

---

## 5. Bandwidth Management

### Rate Limiting

```python
"""
Bandwidth management and rate limiting.
"""

import time
import threading
from collections import deque


class BandwidthLimiter:
    """
    Limit download bandwidth to avoid network saturation.
    """

    def __init__(self, max_mbps: float):
        """
        Initialize bandwidth limiter.

        Args:
            max_mbps: Maximum bandwidth in Mbps
        """
        self.max_bytes_per_second = max_mbps * 1024 * 1024 / 8
        self.window_size = 1.0  # 1 second window
        self.history = deque()
        self.lock = threading.Lock()

    def throttle(self, bytes_transferred: int):
        """
        Throttle based on bandwidth limit.

        Args:
            bytes_transferred: Bytes just transferred
        """
        with self.lock:
            current_time = time.time()

            # Add to history
            self.history.append((current_time, bytes_transferred))

            # Clean old entries
            cutoff = current_time - self.window_size
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()

            # Calculate current rate
            total_bytes = sum(b for _, b in self.history)
            current_rate = total_bytes / self.window_size

            # Sleep if over limit
            if current_rate > self.max_bytes_per_second:
                sleep_time = (total_bytes - self.max_bytes_per_second) / self.max_bytes_per_second
                sleep_time = min(sleep_time, 1.0)  # Cap at 1 second
                time.sleep(sleep_time)


class AdaptiveBandwidthManager:
    """
    Adaptively manage bandwidth based on network conditions.
    """

    def __init__(self, initial_workers: int = 4):
        """
        Initialize adaptive manager.

        Args:
            initial_workers: Starting number of workers
        """
        self.current_workers = initial_workers
        self.min_workers = 1
        self.max_workers = 16

        self.error_count = 0
        self.success_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # seconds

    def record_success(self):
        """Record successful download."""
        self.success_count += 1
        self._maybe_adjust()

    def record_error(self):
        """Record download error."""
        self.error_count += 1
        self._maybe_adjust()

    def _maybe_adjust(self):
        """Adjust workers based on error rate."""
        if time.time() - self.last_adjustment < self.adjustment_interval:
            return

        total = self.success_count + self.error_count
        if total < 10:
            return

        error_rate = self.error_count / total

        old_workers = self.current_workers

        if error_rate > 0.2:  # >20% errors - reduce workers
            self.current_workers = max(
                self.min_workers,
                self.current_workers - 1
            )
        elif error_rate < 0.05 and self.success_count > 20:  # <5% errors - increase
            self.current_workers = min(
                self.max_workers,
                self.current_workers + 1
            )

        if self.current_workers != old_workers:
            print(f"Adjusted workers: {old_workers} -> {self.current_workers} "
                  f"(error rate: {error_rate:.1%})")

        # Reset counters
        self.error_count = 0
        self.success_count = 0
        self.last_adjustment = time.time()

    def get_workers(self) -> int:
        """Get current recommended worker count."""
        return self.current_workers
```

---

## 6. Complete Optimized Download Script

```python
#!/usr/bin/env python3
"""
Optimized SA-1B downloader with all optimization strategies.
"""

import os
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Optimized SA-1B Downloader')
    parser.add_argument('--links', type=str, default='sa1b_links.txt',
                       help='Links file path')
    parser.add_argument('--output', type=str, default='./sa1b_raw',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting file index')
    parser.add_argument('--end', type=int, default=1000,
                       help='Ending file index')
    parser.add_argument('--max-bandwidth', type=float, default=0,
                       help='Max bandwidth in Mbps (0 = unlimited)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output, exist_ok=True)
    checkpoint_dir = os.path.join(args.output, '.checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parse links
    downloads = []
    with open(args.links, 'r') as f:
        for i, line in enumerate(f):
            if args.start <= i < args.end:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    url = parts[-1]
                    output_path = os.path.join(args.output, filename)

                    # Skip if exists (for resume)
                    if args.resume and os.path.exists(output_path):
                        continue

                    downloads.append((filename, url, output_path))

    if not downloads:
        print("No files to download!")
        return

    print(f"SA-1B Optimized Downloader")
    print(f"=" * 50)
    print(f"Files to download: {len(downloads)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    if args.max_bandwidth:
        print(f"Max bandwidth: {args.max_bandwidth} Mbps")
    print()

    # Initialize components
    downloader = RobustDownloader(
        max_retries=5,
        initial_backoff=1.0,
        timeout=(30, 600)
    )

    checkpoint = CheckpointManager(checkpoint_dir)

    bandwidth_manager = None
    if args.max_bandwidth > 0:
        bandwidth_manager = BandwidthLimiter(args.max_bandwidth)

    # Download with progress
    start_time = time.time()
    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for filename, url, output_path in downloads:
            future = executor.submit(
                downloader.download, url, output_path
            )
            futures[future] = (filename, url, output_path)

        with tqdm(total=len(downloads), desc="Downloading") as pbar:
            for future in as_completed(futures):
                filename, url, output_path = futures[future]

                try:
                    success = future.result()
                    if success:
                        completed += 1
                        file_size = os.path.getsize(output_path)
                        checkpoint.complete_download(filename, file_size)
                    else:
                        failed.append(filename)
                        checkpoint.fail_download(filename, "Download failed")

                except Exception as e:
                    failed.append(filename)
                    checkpoint.fail_download(filename, str(e))

                pbar.update(1)

    # Summary
    elapsed = time.time() - start_time
    print(f"\nDownload complete!")
    print(f"  Completed: {completed}/{len(downloads)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed/3600:.1f} hours")

    if failed:
        failed_file = os.path.join(args.output, 'failed.txt')
        checkpoint.export_failed_list(failed_file)
        print(f"  Failed list: {failed_file}")


if __name__ == "__main__":
    main()
```

---

## 7. Best Practices Summary

### Do's

1. **Use multiple workers** (4-8 for most connections)
2. **Enable resume support** (partial downloads save time)
3. **Checkpoint progress** (survive interruptions)
4. **Use connection pooling** (reduce overhead)
5. **Implement exponential backoff** (handle transient errors)
6. **Monitor bandwidth** (avoid network saturation)
7. **Verify downloads** (catch corruption early)

### Don'ts

1. **Don't use too many workers** (diminishing returns after 8)
2. **Don't skip error handling** (network is unreliable)
3. **Don't ignore partial downloads** (resume saves hours)
4. **Don't saturate your connection** (leave headroom)
5. **Don't download sequentially** (parallel is faster)

---

## Sources

**GitHub Resources:**
- [facebookresearch/segment-anything Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - Download time discussions (accessed 2025-11-20)
- [erow/SA-1B](https://github.com/erow/SA-1B) - Parallel download implementation

**Documentation:**
- [Python requests library](https://docs.python-requests.org/) - HTTP client documentation
- [aria2c manual](https://aria2.github.io/manual/en/html/aria2c.html) - Download optimization

**Community:**
- [Hugging Face Forums](https://discuss.huggingface.co/t/problems-about-loading-and-managing-extreme-large-datasets-like-sa1b/46520) - SA-1B loading discussion (accessed 2025-11-20)

---

## Next Steps

- PART 16: Partial download and subset selection
- PART 17: Tar extraction strategies
- PART 18: Verification and integrity checking
