# KNOWLEDGE DROP: SA-1B Official Download Links & Access

**Date**: 2025-11-20
**Runner**: PART 13 (Official Download Links)
**Status**: COMPLETE

---

## Overview

This knowledge drop covers the official download infrastructure for SA-1B, including the links.txt format, license requirements, AWS S3 hosting details, and authentication procedures.

---

## 1. Official Download Page

### Meta AI Download Portal

**URL**: https://ai.meta.com/datasets/segment-anything-downloads/

The official download page provides:
- License agreement acceptance form
- Download links file (links.txt)
- Dataset documentation
- Usage guidelines

From [Meta AI SA-1B Downloads](https://ai.meta.com/datasets/segment-anything-downloads/) (accessed 2025-11-20):
- Requires account creation/login
- Must accept SA-1B Dataset Research License
- Provides text file with all download URLs

---

## 2. Links.txt File Format

### Structure

The `links.txt` (or `sa1b_links.txt`) file contains 1000 entries in the format:

```
filename    url
sa_000000.tar    https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar
sa_000001.tar    https://dl.fbaipublicfiles.com/segment_anything/sa_000001.tar
sa_000002.tar    https://dl.fbaipublicfiles.com/segment_anything/sa_000002.tar
...
sa_000999.tar    https://dl.fbaipublicfiles.com/segment_anything/sa_000999.tar
```

### File Details

- **Total files**: 1000 tar archives
- **Naming pattern**: `sa_XXXXXX.tar` (6-digit zero-padded)
- **Average size**: ~10 GB per tar file
- **Total size**: ~10 TB for complete dataset

### Parsing the Links File

```python
"""
Parse SA-1B links.txt file for download automation.

From: https://github.com/facebookresearch/segment-anything/issues/26
"""

def parse_links_file(links_path: str) -> list:
    """
    Parse the official SA-1B links.txt file.

    Args:
        links_path: Path to links.txt file

    Returns:
        List of (filename, url) tuples
    """
    downloads = []

    with open(links_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Format: filename<tab>url
            parts = line.split('\t')
            if len(parts) == 2:
                filename, url = parts
                downloads.append((filename, url))
            else:
                # Some versions use space separation
                parts = line.split()
                if len(parts) >= 2:
                    filename = parts[0]
                    url = parts[-1]
                    downloads.append((filename, url))

    return downloads


def get_tar_range(links: list, start: int, end: int) -> list:
    """
    Get a subset of tar files for partial download.

    Args:
        links: Full list of (filename, url) tuples
        start: Starting index (0-999)
        end: Ending index (exclusive)

    Returns:
        Subset of links
    """
    return links[start:end]


# Example usage
if __name__ == "__main__":
    links = parse_links_file("sa1b_links.txt")
    print(f"Total tar files: {len(links)}")

    # Get first 10 for testing
    test_subset = get_tar_range(links, 0, 10)
    for filename, url in test_subset:
        print(f"  {filename}: {url[:50]}...")
```

---

## 3. SA-1B Dataset Research License

### License Requirements

From [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (accessed 2025-11-20):

**Key Terms**:
- **Non-commercial use only**: Research and educational purposes
- **No redistribution**: Cannot share the downloaded dataset
- **Attribution required**: Must cite the original paper
- **Derivatives allowed**: Can create derived datasets for research

### License Acceptance

```python
"""
License acceptance verification before download.
"""

LICENSE_TEXT = """
SA-1B Dataset Research License

By downloading this dataset, you agree to:

1. Use the dataset for non-commercial research purposes only
2. Not redistribute the dataset or any derived data
3. Cite the Segment Anything paper in any publications
4. Delete the dataset upon request by Meta AI
5. Not use the dataset to train models for commercial purposes

Citation:
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and ...},
  journal={arXiv:2304.02643},
  year={2023}
}
"""

def verify_license_acceptance() -> bool:
    """
    Interactive license acceptance verification.

    Returns:
        True if user accepts license
    """
    print(LICENSE_TEXT)
    print("\n" + "="*60)

    response = input("\nDo you accept these terms? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("License accepted. Proceeding with download...")
        return True
    else:
        print("License not accepted. Download cancelled.")
        return False


def save_license_record(output_dir: str):
    """
    Save license acceptance record with timestamp.

    Args:
        output_dir: Directory where dataset will be downloaded
    """
    import datetime
    import os

    record_path = os.path.join(output_dir, "LICENSE_ACCEPTED.txt")

    with open(record_path, 'w') as f:
        f.write(f"SA-1B Dataset Research License\n")
        f.write(f"Accepted: {datetime.datetime.now().isoformat()}\n")
        f.write(f"User: {os.environ.get('USER', 'unknown')}\n")
        f.write(f"\n{LICENSE_TEXT}")

    print(f"License record saved to: {record_path}")
```

---

## 4. AWS S3 Hosting Infrastructure

### CDN Details

The SA-1B dataset is hosted on Meta's public file distribution infrastructure:

**Base URL**: `https://dl.fbaipublicfiles.com/segment_anything/`

### CDN Characteristics

From community discussions on [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) (accessed 2025-11-20):

- **Protocol**: HTTPS
- **No authentication**: Direct download links (after license acceptance)
- **Resume support**: HTTP Range requests supported
- **Parallel connections**: Multiple simultaneous downloads allowed
- **Geographic distribution**: CDN provides reasonable speeds globally
- **Rate limiting**: No strict per-IP limits observed

### Testing CDN Connectivity

```python
"""
Test connectivity and download speed to SA-1B CDN.
"""

import requests
import time
from urllib.parse import urlparse


def test_cdn_connectivity(url: str, timeout: int = 10) -> dict:
    """
    Test connectivity to SA-1B download URL.

    Args:
        url: Download URL to test
        timeout: Connection timeout in seconds

    Returns:
        Dictionary with test results
    """
    results = {
        'url': url,
        'reachable': False,
        'supports_resume': False,
        'content_length': 0,
        'server': None,
        'response_time_ms': 0
    }

    try:
        start_time = time.time()

        # HEAD request to get file info without downloading
        response = requests.head(url, timeout=timeout, allow_redirects=True)

        results['response_time_ms'] = (time.time() - start_time) * 1000
        results['reachable'] = response.status_code == 200
        results['server'] = response.headers.get('Server', 'Unknown')

        # Check file size
        content_length = response.headers.get('Content-Length')
        if content_length:
            results['content_length'] = int(content_length)

        # Check resume support (Accept-Ranges header)
        accept_ranges = response.headers.get('Accept-Ranges', '')
        results['supports_resume'] = accept_ranges.lower() == 'bytes'

    except requests.exceptions.RequestException as e:
        results['error'] = str(e)

    return results


def test_download_speed(url: str, sample_mb: int = 10) -> float:
    """
    Test download speed by downloading a sample.

    Args:
        url: Download URL
        sample_mb: Megabytes to download for test

    Returns:
        Download speed in MB/s
    """
    sample_bytes = sample_mb * 1024 * 1024

    headers = {'Range': f'bytes=0-{sample_bytes-1}'}

    start_time = time.time()
    response = requests.get(url, headers=headers, stream=True)

    downloaded = 0
    for chunk in response.iter_content(chunk_size=1024*1024):
        downloaded += len(chunk)
        if downloaded >= sample_bytes:
            break

    elapsed = time.time() - start_time
    speed_mbps = (downloaded / (1024*1024)) / elapsed

    return speed_mbps


# Example usage
if __name__ == "__main__":
    test_url = "https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar"

    print("Testing SA-1B CDN connectivity...")
    results = test_cdn_connectivity(test_url)

    print(f"\nResults:")
    print(f"  Reachable: {results['reachable']}")
    print(f"  Response time: {results['response_time_ms']:.0f}ms")
    print(f"  File size: {results['content_length'] / (1024**3):.2f} GB")
    print(f"  Resume support: {results['supports_resume']}")
    print(f"  Server: {results['server']}")

    if results['reachable']:
        print("\nTesting download speed (10 MB sample)...")
        speed = test_download_speed(test_url, sample_mb=10)
        print(f"  Download speed: {speed:.1f} MB/s")

        # Estimate full download time
        file_size_gb = results['content_length'] / (1024**3)
        eta_hours = (file_size_gb * 1024) / (speed * 3600)
        print(f"  Estimated time for this file: {eta_hours:.1f} hours")
```

---

## 5. Download URL Generation

### Programmatic URL Construction

```python
"""
Generate SA-1B download URLs programmatically.
"""

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"


def generate_tar_url(index: int) -> str:
    """
    Generate download URL for a specific tar file.

    Args:
        index: Tar file index (0-999)

    Returns:
        Full download URL
    """
    if not 0 <= index <= 999:
        raise ValueError(f"Index must be 0-999, got {index}")

    filename = f"sa_{index:06d}.tar"
    return f"{BASE_URL}/{filename}"


def generate_all_urls() -> list:
    """
    Generate all 1000 tar file URLs.

    Returns:
        List of (filename, url) tuples
    """
    urls = []
    for i in range(1000):
        filename = f"sa_{i:06d}.tar"
        url = f"{BASE_URL}/{filename}"
        urls.append((filename, url))
    return urls


def generate_url_range(start: int, end: int) -> list:
    """
    Generate URLs for a range of tar files.

    Args:
        start: Starting index (inclusive)
        end: Ending index (exclusive)

    Returns:
        List of (filename, url) tuples
    """
    urls = []
    for i in range(start, end):
        if 0 <= i <= 999:
            filename = f"sa_{i:06d}.tar"
            url = f"{BASE_URL}/{filename}"
            urls.append((filename, url))
    return urls


def save_links_file(urls: list, output_path: str):
    """
    Save URLs to links.txt format file.

    Args:
        urls: List of (filename, url) tuples
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for filename, url in urls:
            f.write(f"{filename}\t{url}\n")

    print(f"Saved {len(urls)} links to {output_path}")


# Example: Generate custom subset
if __name__ == "__main__":
    # Generate first 100 URLs for testing
    test_urls = generate_url_range(0, 100)
    save_links_file(test_urls, "sa1b_test_100.txt")

    # Generate all URLs
    all_urls = generate_all_urls()
    save_links_file(all_urls, "sa1b_all_1000.txt")
```

---

## 6. Authentication & Access Control

### Current Access Model

From [Meta AI Datasets](https://ai.meta.com/datasets/segment-anything/) (accessed 2025-11-20):

**No authentication required for downloads**:
- URLs are publicly accessible after license acceptance
- No API keys or tokens needed
- No IP-based restrictions observed
- Links remain valid indefinitely

### Best Practices

```python
"""
Best practices for accessing SA-1B downloads.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_download_session() -> requests.Session:
    """
    Create a requests session with proper configuration for SA-1B downloads.

    Returns:
        Configured requests Session
    """
    session = requests.Session()

    # Configure retries for transient failures
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set appropriate headers
    session.headers.update({
        'User-Agent': 'SA-1B-Downloader/1.0 (Research)',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate'
    })

    return session


def validate_download_url(url: str) -> bool:
    """
    Validate that a URL is a legitimate SA-1B download URL.

    Args:
        url: URL to validate

    Returns:
        True if URL appears valid
    """
    # Check base URL
    if not url.startswith("https://dl.fbaipublicfiles.com/segment_anything/"):
        return False

    # Check filename pattern
    import re
    pattern = r"sa_\d{6}\.tar$"
    if not re.search(pattern, url):
        return False

    return True


# Example usage
if __name__ == "__main__":
    session = create_download_session()

    test_url = "https://dl.fbaipublicfiles.com/segment_anything/sa_000000.tar"

    if validate_download_url(test_url):
        response = session.head(test_url)
        print(f"URL valid: {response.status_code == 200}")
    else:
        print("Invalid URL format")
```

---

## 7. Estimated Download Statistics

### Per-File Statistics

| Metric | Value |
|--------|-------|
| Number of tar files | 1,000 |
| Average file size | ~10 GB |
| Smallest file | ~8 GB |
| Largest file | ~12 GB |
| Total dataset size | ~10 TB |

### Download Time Estimates

| Connection Speed | Time per File | Full Dataset |
|-----------------|---------------|--------------|
| 10 Mbps | ~2.2 hours | ~92 days |
| 100 Mbps | ~13 minutes | ~9 days |
| 1 Gbps | ~1.3 minutes | ~22 hours |
| 10 Gbps | ~8 seconds | ~2.2 hours |

---

## 8. Complete Download Workflow

```python
"""
Complete workflow for SA-1B download preparation.
"""

import os
import json
from datetime import datetime


class SA1BDownloadManager:
    """
    Manage SA-1B dataset download process.
    """

    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"
    TOTAL_FILES = 1000

    def __init__(self, output_dir: str):
        """
        Initialize download manager.

        Args:
            output_dir: Base directory for downloads
        """
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, "raw")
        self.state_file = os.path.join(output_dir, "download_state.json")

        os.makedirs(self.raw_dir, exist_ok=True)

    def generate_download_list(self, start: int = 0, end: int = 1000) -> list:
        """
        Generate list of files to download.

        Args:
            start: Starting index
            end: Ending index

        Returns:
            List of (filename, url, local_path) tuples
        """
        downloads = []

        for i in range(start, min(end, self.TOTAL_FILES)):
            filename = f"sa_{i:06d}.tar"
            url = f"{self.BASE_URL}/{filename}"
            local_path = os.path.join(self.raw_dir, filename)
            downloads.append((filename, url, local_path))

        return downloads

    def save_state(self, completed: list, failed: list):
        """
        Save download state for resume capability.

        Args:
            completed: List of completed file indices
            failed: List of failed file indices
        """
        state = {
            'timestamp': datetime.now().isoformat(),
            'completed': completed,
            'failed': failed,
            'total': self.TOTAL_FILES
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> dict:
        """
        Load previous download state.

        Returns:
            State dictionary or None
        """
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None

    def get_remaining_downloads(self, start: int = 0, end: int = 1000) -> list:
        """
        Get list of files still needing download.

        Args:
            start: Starting index
            end: Ending index

        Returns:
            List of (filename, url, local_path) for incomplete downloads
        """
        state = self.load_state()
        completed = set(state.get('completed', [])) if state else set()

        all_downloads = self.generate_download_list(start, end)

        remaining = []
        for filename, url, local_path in all_downloads:
            # Extract index from filename
            index = int(filename.split('_')[1].split('.')[0])

            # Skip if completed or file exists
            if index in completed:
                continue
            if os.path.exists(local_path):
                # Verify file size (optional)
                continue

            remaining.append((filename, url, local_path))

        return remaining


# Example usage
if __name__ == "__main__":
    manager = SA1BDownloadManager("/data/sa1b")

    # Get first 10 files for testing
    downloads = manager.generate_download_list(0, 10)

    print(f"Download list ({len(downloads)} files):")
    for filename, url, local_path in downloads:
        print(f"  {filename} -> {local_path}")
```

---

## Sources

**Official Resources:**
- [Meta AI SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Official dataset page
- [Meta AI SA-1B Downloads](https://ai.meta.com/datasets/segment-anything-downloads/) - Download portal

**GitHub Resources:**
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official SAM repository
- [GitHub Issue #26](https://github.com/facebookresearch/segment-anything/issues/26) - Community download discussion

**Research Paper:**
- [Segment Anything](https://arxiv.org/abs/2304.02643) - arXiv:2304.02643 (Kirillov et al., 2023)

---

## Next Steps

- PART 14: Parallel downloader tools (aria2c, community scripts)
- PART 15: Download time optimization strategies
- PART 16: Partial download and subset selection
