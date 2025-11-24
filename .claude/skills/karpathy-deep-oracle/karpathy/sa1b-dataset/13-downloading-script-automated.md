# Automated Download Script for SA-1B

## Overview

Automate SA-1B downloads with retry logic, progress tracking, and parallel execution for efficient large-scale data acquisition.

## Production Download Script

```python
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_tar(url: str, output_dir: Path, max_retries: int = 5) -> bool:
    """Download single tar with retry logic."""
    filename = url.split('/')[-1]
    output_path = output_dir / filename

    if output_path.exists():
        print(f"✓ {filename} already exists, skipping")
        return True

    for attempt in range(max_retries):
        try:
            subprocess.run(
                ['wget', '--continue', '--progress=bar',
                 '--timeout=60', '-O', str(output_path), url],
                check=True,
                timeout=3600  # 1 hour timeout
            )
            print(f"✓ Downloaded {filename}")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"✗ Attempt {attempt + 1}/{max_retries} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                import time; time.sleep(5)

    print(f"✗ Failed to download {filename} after {max_retries} attempts")
    return False

def download_sa1b(
    links_file: str = "segment_anything_links.txt",
    output_dir: str = "./sa1b_data",
    num_workers: int = 4,
    start_idx: int = 0,
    end_idx: int = None
):
    """
    Download SA-1B dataset with parallel execution.

    Args:
        links_file: Path to links file
        output_dir: Where to save tars
        num_workers: Parallel downloads
        start_idx: Start at this tar index
        end_idx: End at this tar index (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read URLs
    with open(links_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    # Slice if requested
    if end_idx:
        urls = urls[start_idx:end_idx]
    else:
        urls = urls[start_idx:]

    print(f"Downloading {len(urls)} tars with {num_workers} workers")

    # Parallel download
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_tar, url, output_path): url
            for url in urls
        }

        completed = 0
        failed = 0
        for future in as_completed(futures):
            completed += 1
            if future.result():
                pass  # Success
            else:
                failed += 1

            print(f"Progress: {completed}/{len(urls)} complete, {failed} failed")

    print(f"\\nDownload complete: {completed - failed}/{len(urls)} successful")

# Usage examples
if __name__ == "__main__":
    # Quick test: Download first 10 tars
    download_sa1b(end_idx=10, num_workers=2)

    # Production: Download all with 8 parallel workers
    # download_sa1b(num_workers=8)

    # Resume: Download starting from tar 500
    # download_sa1b(start_idx=500, num_workers=8)
```

## ARR-COC Integration

```python
# Download subset for quick training experiments
download_sa1b(
    end_idx=50,  # 50 tars = ~550k images
    num_workers=4,
    output_dir="/mnt/fast-ssd/sa1b-subset"
)
```

**Sources**: Best practices from ML engineering, wget documentation
